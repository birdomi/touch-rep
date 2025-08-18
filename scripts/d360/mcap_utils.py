"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


Utils to read/extract D360 sensor topics from MCAP files and convert them to a pickle format.
This module also provides methods to preprocess the D360 time series data for training.
"""

import bisect
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from mcap.reader import make_reader
from mcap_ros2.reader import McapROS2Message, read_ros2_messages
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from torchaudio.compliance import kaldi
from tqdm import tqdm

imu_raw_types = {"acc": 6, "raw_acc": 1, "raw_gyro": 2, "raw_mag": 3}
imu_raw_scales = {
    "acc": 1 / 4096.0,
    "raw_acc": 1 / 4096.0,
    "raw_gyro": None,
    "raw_mag": 2000 / 32768.0,
}


def load_audio_data(file: str):
    with open(file, "rb") as f:
        reader = make_reader(f)
        topics = sorted(
            [
                channel.topic
                for _, channel in reader.get_summary().channels.items()
                if "mic" in channel.topic and "spec" not in channel.topic
            ]
        )

    devices = sorted(list(set([topic.rsplit("/", 1)[0] for topic in topics])))
    msgs = {device: {"mic_0": [], "mic_1": []} for device in devices}
    for msg in read_ros2_messages(file, topics=topics):
        topic = msg.channel.topic
        msgs[topic[:7]][topic[8:]].append(msg)
    return devices, topics, msgs


def load_imu_data(file: str):
    with open(file, "rb") as f:
        reader = make_reader(f)
        topics = sorted(
            [channel.topic for _, channel in reader.get_summary().channels.items() if "imu" in channel.topic]
        )

    devices = sorted(list(set([topic.rsplit("/", 1)[0] for topic in topics])))
    msgs = {device: {"imu_raw_topic": [], "imu_quat_topic": []} for device in devices}
    for msg in read_ros2_messages(file, topics=topics):
        topic = msg.channel.topic
        msgs[topic[:7]][topic[8:]].append(msg)
    return devices, topics, msgs


def load_image_data(file: str):
    with open(file, "rb") as f:
        reader = make_reader(f)
        topics = sorted(
            [
                channel.topic
                for _, channel in reader.get_summary().channels.items()
                if "image_raw" in channel.topic and "d360" in channel.topic
            ]
        )

    devices = sorted(list(set([topic.rsplit("/", 2)[0] for topic in topics])))
    msgs = {device: {"image_raw/compressed": []} for device in devices}
    for msg in read_ros2_messages(file, topics=topics):
        topic = msg.channel.topic
        msgs[topic[:7]][topic[8:]].append(msg)
    return devices, topics, msgs


def load_ros_data(
    file: str,
    image: bool = False,
    audio: bool = False,
    imu: bool = False,
    pressure: bool = False,
    allegro: Optional[List[str]] = None,
    franka: Optional[List[str]] = None,
    extra_topics: Optional[List[str]] = None,
):
    extra_topics = extra_topics if extra_topics is not None else []

    def check_image(topic: str):
        return image and "image_raw" in topic

    def check_audio(topic: str):
        return audio and "mic" in topic and "spec" not in topic

    def check_imu(topic: str):
        return imu and "imu" in topic

    def check_pressure(topic: str):
        return pressure and "pressure_topic" in topic

    def check_allegro(topic: str):
        return allegro and "allegroHand" in topic

    def check_franka(topic: str):
        return franka and "franka" in topic

    def check_extra_topics(topic: str):
        for extra_topic in extra_topics:
            if extra_topic in topic:
                return True
        return False

    def check(topic: str):
        return (
            check_image(topic)
            or check_audio(topic)
            or check_imu(topic)
            or check_pressure(topic)
            or check_extra_topics(topic)
            or check_allegro(topic)
            or check_franka(topic)
        )

    subtopics: List[str] = []
    if image:
        subtopics.append("image_raw/compressed")

    if audio:
        subtopics.append("mic_0")
        subtopics.append("mic_1")

    if imu:
        subtopics.append("imu_raw_topic")
        subtopics.append("imu_quat_topic")

    if pressure:
        subtopics.append("pressure_topic")

    list_topics = ["d360"]
    if allegro:
        list_topics.append("allegroHand")
    if franka:
        list_topics.append("franka")

    with open(file, "rb") as f:
        reader = make_reader(f)
        topics = sorted(
            [
                channel.topic
                for _, channel in reader.get_summary().channels.items()
                if any([topic in channel.topic for topic in list_topics]) and check(channel.topic)
            ]
        )

    devices = sorted(list(set([topic.split("/", 2)[1] for topic in topics])))
    msgs = {device: {subtopic: [] for subtopic in subtopics} for device in devices}

    if allegro:
        msgs["allegroHand"] = {topic: [] for topic in allegro}
    if franka:
        msgs["franka"] = {topic: [] for topic in franka}

    print("Reading ROS2 messages...")
    for msg in tqdm(read_ros2_messages(file, topics=topics)):
        topic = msg.channel.topic
        if "d360" in topic:
            msgs[topic[1:7]][topic[8:]].append(msg)
        elif "allegroHand" in topic:
            if topic[13:] in allegro:
                msgs["allegroHand"][topic[13:]].append(msg)
        elif "franka" in topic:
            if topic[7:] in franka:
                msgs["franka"][topic[7:]].append(msg)
        else:
            raise NotImplementedError

    topics_read = []
    for d in msgs.keys():
        for t in msgs[d].keys():
            if len(msgs[d][t]) > 0:
                topics_read.append(f"/{d}/{t}")

    return devices, topics_read, msgs


def save_to_pickle(
    file: str,
    file_out: str,
    image: bool = True,
    audio: bool = True,
    imu: bool = True,
    pressure: bool = True,
    allegro: Optional[List[str]] = None,
    franka: Optional[List[str]] = None,
    extra_topics: Optional[List[str]] = None,
):

    devices, topics, raw_msgs = load_ros_data(
        file,
        image=image,
        audio=audio,
        imu=imu,
        pressure=pressure,
        allegro=allegro,
        franka=franka,
        extra_topics=extra_topics,
    )
    raw_data = {device: {} for device in devices}

    print("Saving to pickle...")
    for topic in topics:
        device, topic = topic.split("/", 2)[1:]

        if topic not in raw_msgs[device]:
            continue
        if topic in raw_msgs[device]:
            print(f"Loading /{device}/{topic}")
            if "image_raw/compressed" in topic:
                stamp = [get_timestamp(msg) for msg in raw_msgs[device][topic]]
                format = [msg.ros_msg.format for msg in raw_msgs[device][topic]]
                data = [msg.ros_msg.data for msg in raw_msgs[device][topic]]
                raw_data[device].update({topic: {"stamp": stamp, "format": format, "data": data}})

                print(f"Saving d360 images to {file_out}/{device}/img/ ... ")
                path_save_imgs = f"{file_out}/{device}/img/"
                os.makedirs(path_save_imgs, exist_ok=True)
                for i, img_bytes in enumerate(data):
                    file_path = f"{path_save_imgs}/{i}"
                    with open(file_path, "wb") as file:
                        file.write(img_bytes)
                print("Saved d360 images")

            elif "mic" in topic:
                stamp = [get_timestamp(msg) for msg in raw_msgs[device][topic]]
                sample_rate = [msg.ros_msg.sample_rate for msg in raw_msgs[device][topic]]
                data = [msg.ros_msg.data for msg in raw_msgs[device][topic]]
                raw_data[device].update({topic: {"stamp": stamp, "sample_rate": sample_rate, "data": data}})
            elif "imu_quat" in topic:
                stamp = [get_timestamp(msg) for msg in raw_msgs[device][topic]]
                sensor_ts = [msg.ros_msg.sensor_ts for msg in raw_msgs[device][topic]]
                quat = [
                    [
                        msg.ros_msg.quat.x,
                        msg.ros_msg.quat.y,
                        msg.ros_msg.quat.z,
                        msg.ros_msg.quat.w,
                    ]
                    for msg in raw_msgs[device][topic]
                ]
                raw_data[device].update({topic: {"stamp": stamp, "sensor_ts": sensor_ts, "quat": quat}})
            elif "imu_raw" in topic:
                stamp = [get_timestamp(msg) for msg in raw_msgs[device][topic]]
                sensor_ts = [msg.ros_msg.sensor_ts for msg in raw_msgs[device][topic]]
                types = [msg.ros_msg.type for msg in raw_msgs[device][topic]]
                data = [[msg.ros_msg.x, msg.ros_msg.y, msg.ros_msg.z] for msg in raw_msgs[device][topic]]
                raw_data[device].update(
                    {
                        topic: {
                            "stamp": stamp,
                            "sensor_ts": sensor_ts,
                            "type": types,
                            "data": data,
                        }
                    }
                )
            elif "pressure" in topic:
                stamp = [get_timestamp(msg) for msg in raw_msgs[device][topic]]
                if hasattr(raw_msgs[device][topic][0], "d360_pressure"):
                    sensor_ts = [msg.ros_msg.ts.nanosec / 1000 for msg in raw_msgs[device][topic]]
                    pressure_data = [msg.ros_msg.d360_pressure for msg in raw_msgs[device][topic]]
                    temperature_data = [msg.ros_msg.d360_temperature for msg in raw_msgs[device][topic]]
                else:
                    sensor_ts = [msg.ros_msg.ts / 1000 for msg in raw_msgs[device][topic]]
                    pressure_data = [msg.ros_msg.pressure for msg in raw_msgs[device][topic]]
                    temperature_data = [msg.ros_msg.temperature for msg in raw_msgs[device][topic]]
                raw_data[device].update(
                    {
                        topic: {
                            "stamp": stamp,
                            "sensor_ts": sensor_ts,
                            "pressure": pressure_data,
                            "temperature": temperature_data,
                        }
                    }
                )
            elif ("joint_states" in topic) or ("joint_cmd" in topic) and ("allegroHand" in device):
                stamp = [get_timestamp(msg) for msg in raw_msgs[device][topic]]
                position = [msg.ros_msg.position for msg in raw_msgs[device][topic]]
                velocity = [msg.ros_msg.velocity for msg in raw_msgs[device][topic]]
                effort = [msg.ros_msg.effort for msg in raw_msgs[device][topic]]
                raw_data[device].update(
                    {
                        topic: {
                            "stamp": stamp,
                            "position": position,
                            "velocity": velocity,
                            "effort": effort,
                        }
                    }
                )
            else:
                raise NotImplementedError
        else:
            print(f"{device}/{topic} is dropped.")
    pd.to_pickle(raw_data, f"{file_out}/data.pickle")


def get_timestamp(msg: McapROS2Message):
    return msg.ros_msg.header.stamp.sec + msg.ros_msg.header.stamp.nanosec / 1e9


def extract_audio(msgs: Dict[str, Dict[str, List[McapROS2Message]]], device: str, mics: List[str]):
    times = []
    audio = []
    for mic in mics:
        ros_msgs = msgs[device][mic]
        times.append(np.array([get_timestamp(msg) for msg in ros_msgs]))
        audio.append(np.array([msg.ros_msg.data for msg in ros_msgs]))
    return times, audio


def find_intersections(raw_times, raw_data):
    i, j = 0, 0
    m, n = i, j
    times = []
    data = [[], []]
    M = 5000
    while m < len(raw_times[0]) and n < len(raw_times[1]):
        K = min([M, len(raw_times[0]) - m, len(raw_times[1]) - n])
        check = np.where(raw_times[0][m : m + K] != raw_times[1][n : n + K])[0]
        if check.any():
            k = check[0]
            assert np.all(raw_times[0][i : m + k] == raw_times[1][j : n + k])
            if k != 0:
                times.append(raw_times[0][i : m + k])
                data[0].append(raw_data[0][i : m + k])
                data[1].append(raw_data[1][j : n + k])
            if raw_times[0][m + k] < raw_times[1][n + k]:
                i = bisect.bisect_left(raw_times[0], raw_times[1][n + k], lo=k + 1)
                j = n + k
            else:
                i = m + k
                j = bisect.bisect_left(raw_times[1], raw_times[0][m + k], lo=k + 1)
            m, n = i, j
        else:
            m += K
            n += K

    times.append(raw_times[0][i:m])
    data[0].append(raw_data[0][i:m])
    data[1].append(raw_data[1][j:n])

    return times, data


def get_labels(path: str, datasets: List[str]):
    objects = set()
    surfaces = set()
    actions = set()
    for dataset in datasets:
        yaml_file = os.path.join(path, dataset, "metadata.yaml")
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        objects.add(yaml_data["object"])
        actions.add(yaml_data["action"])
        surfaces.add(yaml_data["surface"])
    return objects, actions, surfaces


def get_weights(cls: List[str], cnts: List[int], beta=0.05):
    assert len(cls) == len(cnts), "cls and cnts must be of the same size"
    num_cls = len(cls)
    cls_cnts = {cl: 0 for cl in set(cls)}
    for cl, cnt in zip(cls, cnts):
        cls_cnts[cl] += cnt
    total_cnts = sum(cnts)

    cls_weights = {cl: 1 / (cnt / total_cnts + beta / num_cls) for cl, cnt in sorted(cls_cnts.items())}
    total_weights = sum(cls_weights.values())
    cls_weights = {cl: weight / total_weights for cl, weight in cls_weights.items()}
    return cls_weights


def get_prefusion(use_prefusion: bool):
    return "use_prefusion" if use_prefusion else "no_prefusion"


def preprocess_image(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    backgrounds: Optional[Dict[str, np.array]],
    rects: Optional[Dict[str, Tuple[int, int, int, int]]],
    device: str,
    preload: bool = True,
    split: bool = True,
):
    assert len(topic) == 1, "Only 1 topic is expected for D360 image data."

    img_sr = 30
    raw_data = msgs[device][topic[0]]

    raw_times = np.array(raw_data["stamp"])
    tol = 2.0 / img_sr
    segs = np.concatenate([[0], np.where(np.diff(raw_times) > tol)[0] + 1, [len(raw_times)]])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)
    segs = segs[segs[:, 1] - segs[:, 0] > 5]

    raw_imgs = raw_data["data"] if preload else np.arange(len(raw_data["data"]))
    rect = rects[device]
    if backgrounds is not None:
        bg = backgrounds[device][None, rect[0] : rect[1], rect[2] : rect[3]].astype(int)
    else:
        bg = None

    times = [raw_times[seg[0] : seg[1]] for seg in segs]
    times = [np.min(time - np.arange(len(time)) / img_sr) + np.arange(len(time)) / img_sr for time in times]
    imgs = [raw_imgs[seg[0] : seg[1]] for seg in segs]

    if not split:
        times = np.concatenate(times, axis=0)
        imgs = [img for img_segs in imgs for img in img_segs]

    segs = np.concatenate([[0], np.cumsum(segs[:, 1] - segs[:, 0])])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)

    return times, imgs, segs, rect, bg


def preprocess_audio(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    split: bool = True,
):
    assert len(topic) == 2, "Only 2 topics are expected for D360 audio data."

    hop = 512
    sr = 48000
    dt = hop / sr

    mics = ["mic_0", "mic_1"]
    raw_times = [np.array(msgs[device][mic]["stamp"]) for mic in mics]
    raw_audio = [np.array(msgs[device][mic]["data"]) for mic in mics]

    times, audio = find_intersections(raw_times, raw_audio)

    raw_times = np.concatenate(times, axis=0)
    raw_audio = [np.concatenate(data, axis=0) for data in audio]

    segs = set(np.concatenate([[0], np.cumsum([time.shape[0] for time in times])]))
    segs = list(segs.union(set(np.where(np.diff(raw_times) > 5 * dt)[0] + 1)))
    segs = np.array(sorted(segs))
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=1)

    num_segs = segs.shape[0]
    times = [raw_times[segs[n][0] : segs[n][1]] for n in range(num_segs)]
    times = [np.min([time - np.arange(len(time)) * dt]) + np.arange(len(time)) * dt for time in times]
    audio = [[data[seg[0] : seg[1]] for seg in segs] for data in raw_audio]
    times = [time[0] + np.arange(len(time) * hop) / 48000 for time in times]
    audio = [[subdata.reshape(-1) for subdata in data] for data in audio]
    segs = segs * hop

    if not split:
        times = np.concatenate(times, axis=0)
        audio = [np.concatenate(mic, axis=0) for mic in audio]
    return times, audio, segs


def preprocess_imu(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    type: Literal["acc", "raw_acc", "raw_gyro", "raw_mag"],
    split: bool = True,
):
    assert len(topic) == 1, "Only 1 topic is expected for D360 IMU data."

    time_amplifier = 65536
    scale = imu_raw_scales[type]
    type = imu_raw_types[type]
    imu_raw_data = msgs[device][topic[0]]
    selected = np.array(imu_raw_data["type"]) == type
    raw_times = np.array(imu_raw_data["stamp"])[selected]
    raw_imu = np.array(imu_raw_data["data"])[selected] * scale
    raw_sensor_ts = np.array(imu_raw_data["sensor_ts"])[selected]
    raw_sensor_ts = raw_sensor_ts / time_amplifier
    raw_times = np.min(raw_times - raw_sensor_ts) + raw_sensor_ts
    tol = 2 / 400
    segs = np.concatenate([[0], np.where(np.diff(raw_sensor_ts) > tol)[0] + 1, [len(raw_sensor_ts)]])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)
    segs = segs[segs[:, 1] - segs[:, 0] > 32]

    times = [raw_times[seg[0] : seg[1]] for seg in segs]
    imu = [raw_imu[seg[0] : seg[1]] for seg in segs]

    if not split:
        times = np.concatenate(times, axis=0)
        imu = np.concatenate(imu, axis=0)

    segs = np.concatenate([[0], np.cumsum(segs[:, 1] - segs[:, 0])])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)

    return times, imu, segs


def preprocess_force(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    split: bool = True,
):
    assert len(topic) == 1, "Only 1 topic is expected for D360 force data."

    ft_sr = 400
    tol = 2.0 / ft_sr

    raw_data = msgs[device][topic[0]]
    raw_times = np.array(raw_data["stamp"])
    raw_force = np.array(raw_data["force"]) if "force" in raw_data else np.array(raw_data["data"])

    segs = np.concatenate([[0], np.where(np.diff(raw_times) > tol)[0] + 1, [len(raw_times)]])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)
    segs = segs[segs[:, 1] - segs[:, 0] > 5]

    times = [raw_times[seg[0] : seg[1]] for seg in segs]
    times = [np.min(time - np.arange(len(time)) / ft_sr) + np.arange(len(time)) / ft_sr for time in times]
    forces = [raw_force[seg[0] : seg[1]] for seg in segs]

    if not split:
        times = np.concatenate(times, axis=0)
        forces = np.concatenate(forces, axis=0)

    segs = np.concatenate([[0], np.cumsum(segs[:, 1] - segs[:, 0])])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)

    forces[:, -1] *= -1.0  # positive normal forces

    return times, forces, segs


def preprocess_pressure(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    split: bool = True,
):
    assert len(topic) == 1, "Only 1 topic is expected for D360 pressure data."
    sample_rate = 200
    tol = 2.0 / sample_rate

    raw_data = msgs[device][topic[0]]

    assert (np.array(raw_data["pressure"]) >= 1000).sum() / len(raw_data["pressure"]) >= 0.1, "Invalid pressure data."

    raw_times = np.array(raw_data["stamp"])
    raw_sensor_ts = np.array(raw_data["sensor_ts"])
    raw_times = np.min(raw_times - raw_sensor_ts) + raw_sensor_ts
    raw_pressure = np.array(raw_data["pressure"])

    segs = np.concatenate([[0], np.where(np.diff(raw_times) > tol)[0] + 1, [len(raw_times)]])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)
    segs = segs[segs[:, 1] - segs[:, 0] > 25]

    times = [raw_times[seg[0] : seg[1]] for seg in segs]
    pressure = [raw_pressure[seg[0] : seg[1]] for seg in segs]

    if not split:
        times = np.concatenate(times, axis=0)
        pressure = np.concatenate(pressure, axis=0)

    segs = np.concatenate([[0], np.cumsum(segs[:, 1] - segs[:, 0])])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)

    return times, pressure, segs


def preprocess_contact(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    type: Literal["angle", "height"],
    split: bool = True,
):
    assert len(topic) == 1, "Only 1 topic is expected for D360 contact data."
    sample_rate = 400
    tol = 2 / sample_rate

    raw_data = msgs[device][topic[0]]
    raw_times = np.array(raw_data["stamp"])

    raw_contacts = np.array(raw_data[type])
    if type == "angle":
        raw_contacts = np.where(raw_contacts < 0, raw_contacts + 360, raw_contacts)

    dts = np.diff(raw_times)
    segs = [
        seg + 1
        for seg in np.where(dts > tol)[0]
        if dts[seg : seg + np.ceil(dts[seg] * sample_rate).astype(int) + 1].mean() * sample_rate > 1.1
    ]
    segs = np.concatenate([[0], segs, [len(raw_times)]]).astype(int)
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)
    segs = segs[segs[:, 1] - segs[:, 0] > 5]

    times = [raw_times[seg[0] : seg[1]] for seg in segs]
    times = [np.min(time - np.arange(len(time)) / sample_rate) + np.arange(len(time)) / sample_rate for time in times]
    contacts = [raw_contacts[seg[0] : seg[1]] for seg in segs]

    if not split:
        times = np.concatenate(times, axis=0)
        contacts = np.concatenate(contacts, axis=0)

    segs = np.concatenate([[0], np.cumsum(segs[:, 1] - segs[:, 0])])
    segs = np.concatenate([segs[:-1, None], segs[1:, None]], axis=-1)

    return times, contacts, segs


def load_image(
    msgs: Dict[str, Dict[str, List[Any]]],
    sequence: str,
    devices: List[str],
    config: DictConfig,
    data: Dict[str, Any],
    preload: bool = True,
):
    bg_path = os.path.join(sequence, "background")
    if not os.path.exists(bg_path):
        bg_path = (Path(sequence).parents[2] / "background").as_posix()
    img_rects = {}
    for device in devices:
        with open(os.path.join(bg_path, f"{device}.txt"), "r") as f:
            img_rects[device] = [int(num) for num in f.readlines()[0].split(" ")]
    data["rect"].append(img_rects)

    if data["bg"] is not None:
        img_bgs = {}
        for device in devices:
            img_bg = cv2.imread(os.path.join(bg_path, f"{device}.png"))[:, :, ::-1]
            img_bgs[device] = np.ascontiguousarray(img_bg)
        data["bg"].append(img_bgs)
    else:
        img_bgs = None

    init_time = []
    init_data = []
    init_seg = []
    init_interp = []

    for device in devices:
        init_dev_time, init_dev_data, init_dev_seg, _, _ = preprocess_image(
            msgs=msgs,
            topic=config.topic,
            backgrounds=img_bgs,
            rects=img_rects,
            device=device,
            preload=preload,
            split=False,
        )
        init_time.append(init_dev_time)
        init_data.append(init_dev_data)
        init_seg.append([np.array([init_dev_time[seg[0]], init_dev_time[seg[1] - 1]]) for seg in init_dev_seg])
        init_interp.append(interp1d(init_dev_time, np.arange(0, len(init_dev_time)), axis=0, kind="nearest"))

    data["init_time"].append(init_time)
    data["init_data"].append(init_data)
    data["init_seg"].append(init_seg)
    data["interp"].append(init_interp)


def load_mic_wave(
    msgs: Dict[str, Dict[str, List[Any]]],
    sequence: str,
    devices: List[str],
    config: DictConfig,
    data: Dict[str, Any],
):
    init_time = []
    init_data = []
    init_seg = []
    init_interp = []
    for device in devices:
        init_dev_time, init_dev_data, init_dev_seg = preprocess_audio(
            msgs=msgs, topic=config.topic, device=device, split=False
        )
        init_time.append(init_dev_time)
        init_data.append(init_dev_data)
        init_seg.append([np.array([init_dev_time[seg[0]], init_dev_time[seg[1] - 1]]) for seg in init_dev_seg])
        init_interp.append([interp1d(init_dev_time, data, axis=0, kind="nearest") for data in init_dev_data])

    data["init_time"].append(init_time)
    data["init_data"].append(init_data)
    data["init_seg"].append(init_seg)
    data["interp"].append(init_interp)


def load_time_series(
    func: partial,
    msgs: Dict[str, Dict[str, List[Any]]],
    sequence: str,
    devices: Optional[List[str]],
    config: DictConfig,
    data: Dict[str, Any],
    interp: str = "linear",
):
    if devices is not None:
        init_time = []
        init_data = []
        init_seg = []
        init_interp = []
        for device in devices:
            init_dev_time, init_dev_data, init_dev_seg = func(msgs=msgs, topic=config.topic, device=device, split=False)
            init_time.append(init_dev_time)
            init_data.append(init_dev_data)
            init_seg.append([np.array([init_dev_time[seg[0]], init_dev_time[seg[1] - 1]]) for seg in init_dev_seg])
            init_interp.append(interp1d(init_dev_time, init_dev_data, axis=0, kind=interp))
    else:
        init_time, init_data, init_seg = func(msgs=msgs, device=device, split=False)
        init_seg = [np.array([init_time[seg[0]], init_time[seg[1] - 1]]) for seg in init_seg]
        init_interp = interp1d(init_dev_time, init_dev_data, axis=0, kind=interp)

    data["init_time"].append(init_time)
    data["init_data"].append(init_data)
    data["init_seg"].append(init_seg)
    data["interp"].append(init_interp)


def normalize_img(datasets, num_threads=8, backend="threads"):
    def stats(x):
        return [x.mean(axis=(0, 1)), (x**2).mean(axis=(0, 1))]

    def process(datasets, index):
        n, seq, dev, seg, idx = index
        dataset = datasets[n]
        return stats(dataset.d360.extract_image(seq, dev, seg, idx))

    offsets = [dataset.device_msgs.index("img") for dataset in datasets]

    img_idxs = [
        np.array([n, seq, dev, seg, idx[offset]])
        for n, (dataset, offset) in enumerate(zip(datasets, offsets))
        for (seq, dev, seg), idx, _ in dataset.data_idxs
    ]

    pixel_stats = Parallel(n_jobs=num_threads, prefer=backend)(
        delayed(process)(datasets, img_id) for img_id in img_idxs
    )
    pixel_avg, pixel_var = np.array(pixel_stats).astype(np.float64).mean(axis=0)
    pixel_std = np.sqrt(pixel_var - pixel_avg**2)

    return pixel_avg, pixel_std


def normalize_mic_fbank(datasets):
    fbanks = [
        np.concatenate(x, axis=-1) for dataset in datasets for xss in dataset.mic_fbank_data for xs in xss for x in xs
    ]
    fbanks = np.concatenate(fbanks, axis=0)
    fbank_avg = np.mean(fbanks, axis=0)
    fbank_std = np.std(fbanks, axis=0)
    return fbank_avg, fbank_std


def normalize_pressure(datasets):
    def filter_data(dataset, x):
        length = dataset.pressure_length
        stride = dataset.pressure_stride
        assert stride == 1

        x = np.stack([x[n : n + length * stride : stride] for n in range(len(x) - (length - 1) * stride)])
        return dataset.d360.device_cfg.pressure.filter(x)[:, :, None]

    pressures = [
        filter_data(dataset, x)
        for dataset in datasets
        for xss in dataset.pressure_data
        for xs in xss
        for x in xs
        if len(x) >= (dataset.pressure_length - 1) * dataset.pressure_stride
    ]
    pressures = np.concatenate(pressures, axis=0)
    pressure_avg = np.mean(pressures, axis=(0, 1))
    pressure_std = np.std(pressures, axis=(0, 1))
    return pressure_avg, pressure_std


def sync_image(
    config: DictConfig,
    data: Dict[str, Dict[str, Any]],
    synced: bool,
):
    img_data = data["img"]
    init_data = img_data["init_data"][-1]
    init_interp = img_data["interp"][-1]
    synced_time = img_data["time"][-1]
    if synced:
        init_idxs = [[interp(time).astype(int) for time in synced_time] for interp in init_interp]
    else:
        assert len(synced_time) == len(init_interp)
        init_idxs = [
            [interp(dev_time).astype(int) for dev_time in synced_dev_time]
            for interp, synced_dev_time in zip(init_interp, synced_time)
        ]
    synced_data = [[[init_data[dev][id] for id in ids] for ids in idxs] for dev, idxs in enumerate(init_idxs)]
    img_data["data"].append(synced_data)


def sync_audio(config: DictConfig, data: Dict[str, Dict[str, Any]], synced: bool):
    mic_wave_data = data["mic_wave"]
    synced_wave_time = mic_wave_data["time"][-1]
    init_wave_interp = mic_wave_data["interp"][-1]

    if synced:
        synced_wave_data = [
            [[interp(time) for interp in wave_interp] for time in synced_wave_time] for wave_interp in init_wave_interp
        ]
    else:
        assert len(synced_wave_time) == len(init_wave_interp)
        synced_wave_data = [
            [[interp(dev_time) for interp in wave_interp] for dev_time in synced_dev_time]
            for wave_interp, synced_dev_time in zip(init_wave_interp, synced_wave_time)
        ]
    mic_wave_data["data"].append(synced_wave_data)

    mic_wave_cfg = config.mic_wave
    mic_fbank_cfg = config.mic_fbank
    mic_fbank_data = data["mic_fbank"]
    synced_fbank_data = [
        [
            [
                kaldi.fbank(
                    torch.from_numpy(data[None]),
                    htk_compat=True,
                    sample_frequency=mic_wave_cfg.sample_rate,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=mic_fbank_cfg.num_mel_bins,
                    dither=0.0,
                    frame_length=mic_fbank_cfg.frame_length,
                    frame_shift=mic_fbank_cfg.frame_shift,
                    high_freq=mic_fbank_cfg.high_freq,
                    low_freq=mic_fbank_cfg.low_freq,
                )
                .numpy()
                .astype(np.float32)
                for data in wave
            ]
            for wave in synced_dev_wave
        ]
        for synced_dev_wave in synced_wave_data
    ]
    mic_fbank_data["data"].append(synced_fbank_data)

    if synced:
        mic_fbank_data["time"][-1] = [
            np.clip(
                np.arange(
                    interval[0] + mic_fbank_cfg.frame_length / 1000,
                    interval[1],
                    1 / mic_fbank_cfg.sample_rate,
                ),
                interval[0],
                interval[1],
            )
            for interval in mic_fbank_data["interval"][-1]
        ]
    else:
        mic_fbank_data["time"][-1] = [
            [
                np.clip(
                    np.arange(
                        interval[0] + mic_fbank_cfg.frame_length / 1000,
                        interval[1],
                        1 / mic_fbank_cfg.sample_rate,
                    ),
                    interval[0],
                    interval[1],
                )
                for interval in dev_interval
            ]
            for dev_interval in mic_fbank_data["interval"][-1]
        ]


def sync_time_series(config: DictConfig, data: Dict[str, Dict[str, Any]], synced: bool, topic: str):
    info = data[topic]
    init_interp = info["interp"][-1]
    synced_time = info["time"][-1]
    if synced:
        synced_data = [[interp(time).astype(np.float32) for time in synced_time] for interp in init_interp]
    else:
        assert len(synced_time) == len(init_interp)
        synced_data = [
            [interp(dev_time).astype(np.float32) for dev_time in synced_dev_time]
            for interp, synced_dev_time in zip(init_interp, synced_time)
        ]
    info["data"].append(synced_data)


def merge_intervals(segs0, time0, segs1, time1):
    intervals = []
    i, j = 0, 0
    while i < len(segs0) and j < len(segs1):
        l_time = max(time0[segs0[i][0]], time1[segs1[j][0]])
        r_time = min(time0[segs0[i][1] - 1], time1[segs1[j][1] - 1])
        if l_time < r_time:
            intervals.append([l_time, r_time])

        if time0[segs0[i][1] - 1] < time1[segs1[j][1] - 1]:
            i += 1
        else:
            j += 1

    intervals = np.array(intervals)
    return intervals


def intersect_intervals(intervals0, intervals1):
    intervals = []
    i, j = 0, 0
    while i < len(intervals0) and j < len(intervals1):
        l_time = max(intervals0[i][0], intervals1[j][0])
        r_time = min(intervals0[i][1], intervals1[j][1])

        if l_time < r_time:
            intervals.append(np.array([l_time, r_time]))

        if intervals0[i][1] < intervals1[j][1]:
            i += 1
        else:
            j += 1

    return intervals


def segment_data(init_intervals, init_times, init_data, step):
    intervals = []
    times = []
    data = []
    i, j = 0, 0
    while i < len(init_intervals) and j < len(init_times):
        assert len(init_times[j]) == len(init_data[j])
        l_time = max(init_intervals[i][0], init_times[j][0])
        r_time = min(init_intervals[i][1], init_times[j][-1])
        if l_time < r_time:
            start_index = np.ceil((l_time - init_times[j][0]) / step).astype(int)
            end_index = np.floor((r_time - init_times[j][0]) / step).astype(int)
            start = init_times[j][0] + start_index * step
            end = init_times[j][0] + end_index * step
            if start <= end:
                intervals.append([start, end])
                times.append(init_times[j][0] + np.arange(start_index, end_index + 1) * step)
                data.append(init_data[j][start_index : end_index + 1])

        if init_intervals[i][1] < init_times[j][-1]:
            i += 1
        else:
            j += 1
    return intervals, times, data


def select_by_interval(data, time, interval):
    return data[np.logical_and(time >= interval[0], time < interval[1])]


def get_experiment_name(
    use_img: bool,
    use_mic: bool,
    use_imu: bool,
    use_pressure: bool,
    model_size: str,
    attn_cls: str,
    fusion: str,
):
    modals = []
    if use_img:
        modals.append("Image")

    if use_mic:
        modals.append("Audio")

    if use_imu:
        modals.append("IMU")

    if use_pressure:
        modals.append("Pressure")

    assert modals, "No modalities are specified."

    modals = "_".join(modals)

    if attn_cls == "Attention" or attn_cls == "MemEffAttention":
        attn = "Attn"
    elif attn_cls == "DiffAttention" or attn_cls == "MemEffDiffAttention":
        attn = "DiffAttn"
    else:
        raise NotImplementedError

    return "-".join([modals, model_size.capitalize(), attn, fusion.capitalize()])


def get_modality_tag(use_img: bool, use_mic: bool, use_imu: bool, use_pressure: bool):
    modals = []
    if use_img:
        modals.append("image")

    if use_mic:
        modals.append("audio")

    if use_imu:
        modals.append("imu")

    if use_pressure:
        modals.append("pressure")

    assert modals, "No modalities are specified."

    modals = "_".join(modals)

    return modals


def get_modality_used_tag(modal_tag: str, used_modals: Optional[List[str]]):
    if used_modals is not None:
        modals = []
        if "img" in used_modals:
            modals.append("image")

        if "mic" in used_modals:
            modals.append("audio")

        if "imu" in used_modals:
            modals.append("imu")

        if "pressure" in used_modals:
            modals.append("pressure")
    else:
        modals = modal_tag.split("_")

    modals = ["acc" if modal == "imu" else modal for modal in modals]
    modals = "_".join(modals)

    return f"{modals}_used"


def high_pass_filter(cutoff: Any, fs: Any, order: int = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lambda signal: filtfilt(b, a, signal)
