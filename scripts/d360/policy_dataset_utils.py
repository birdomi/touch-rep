from typing import Optional, Dict, Any, List, Literal, Tuple
import os
import numpy as np
import bisect
import einops
import torch
from torchaudio.compliance import kaldi
from scipy.interpolate import interp1d
from PIL import Image as PILImage
import io

T_OFFSET = 0.5

imu_raw_types = {"acc": 6, "raw_acc": 1, "raw_gyro": 2, "raw_mag": 3}
imu_raw_scales = {
    "acc": 1 / 4096.0,
    "raw_acc": 1 / 4096.0,
    "raw_gyro": None,
    "raw_mag": 2000 / 32768.0,
}

modalities_config = {
    "img": {
        "topic": ["image_raw/compressed"],
        "sample_rate": 30,
        "length": 2,
        "stride": 5,
    },
    "mic_wave": {
        "topic": ["mic_0", "mic_1"],
        "sample_rate": 48000,
        "stride": 1,
    },
    "mic_fbank": {
        "topic": ["mic_fbank"],
        "sample_rate": 400,
        "length": 224,
        "stride": 1,
    },
    "imu_acc": {
        "topic": ["imu_raw_topic"],
        "sample_rate": 400,
        "length": 224,
        "stride": 1,
    },
    "pressure": {
        "topic": ["pressure_topic"],
        "sample_rate": 200,
        "length": 224,
        "stride": 1,
    },
}


def preprocess_imu(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    type: Literal["acc", "raw_acc", "raw_gyro", "raw_mag"],
    split: bool = False,
    start_end_timestamps: Optional[List[float]] = None,
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

    start_end_timestamps = [
        start_end_timestamps[0] - T_OFFSET,
        start_end_timestamps[1] + T_OFFSET,
    ]
    times_idx = np.searchsorted(raw_times, start_end_timestamps)
    times = [raw_times[times_idx[0] : times_idx[1]]]
    imu = [raw_imu[times_idx[0] : times_idx[1]]]

    if not split:
        times = np.concatenate(times, axis=0)
        imu = np.concatenate(imu, axis=0)

    return times, imu


def preprocess_pressure(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    split: bool = False,
    start_end_timestamps: Optional[List[float]] = None,
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

    start_end_timestamps = [
        start_end_timestamps[0] - T_OFFSET,
        start_end_timestamps[1] + T_OFFSET,
    ]

    times_idx = np.searchsorted(raw_times, start_end_timestamps)
    times = [raw_times[times_idx[0] : times_idx[1]]]
    pressure = [raw_pressure[times_idx[0] : times_idx[1]]]

    if not split:
        times = np.concatenate(times, axis=0)
        pressure = np.concatenate(pressure, axis=0)

    return times, pressure


def preprocess_allergro_joint_states(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str = None,
    split: bool = False,
    start_end_timestamps: Optional[List[float]] = None,
):
    ft_sr = 400
    tol = 2.0 / ft_sr

    if "joint_cmd" in topic[0]:
        ft_sr = 20
        tol = 2.0 / ft_sr

    device = topic[0].split("/")[0]
    topic = topic[0].split("/")[1]
    raw_data = msgs[device][topic]
    raw_times = np.array(raw_data["stamp"])
    raw_position = np.array(raw_data["position"])
    raw_velocity = np.array(raw_data["velocity"])
    raw_effort = np.array(raw_data["effort"])

    start_end_timestamps = [
        start_end_timestamps[0] - T_OFFSET,
        start_end_timestamps[1] + T_OFFSET,
    ]
    times_idx = np.searchsorted(raw_times, start_end_timestamps)
    times = [raw_times[times_idx[0] : times_idx[1]]]
    position = [raw_position[times_idx[0] : times_idx[1]]]

    if not split:
        times = np.concatenate(times, axis=0)
        position = np.concatenate(position, axis=0)
        # velocity = np.concatenate(velocity, axis=0)
        # effort = np.concatenate(effort, axis=0)

    return times, position


def preprocess_image(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    preload: bool = True,
    split: bool = False,
    start_end_timestamps: Optional[List[float]] = None,
):
    assert len(topic) == 1, "Only 1 topic is expected for D360 image data."

    img_sr = 30
    raw_data = msgs[device][topic[0]]

    raw_times = np.array(raw_data["stamp"])
    tol = 2.0 / img_sr

    raw_imgs = raw_data["data"] if preload else np.arange(len(raw_data["data"]))

    start_end_timestamps = [
        start_end_timestamps[0] - T_OFFSET,
        start_end_timestamps[1] + T_OFFSET,
    ]
    times_idx = np.searchsorted(raw_times, start_end_timestamps)
    times = [raw_times[times_idx[0] : times_idx[1]]]
    imgs = [raw_imgs[times_idx[0] : times_idx[1]]]

    if not split:
        times = np.concatenate(times, axis=0)
        imgs = [img for img_segs in imgs for img in img_segs]

    return times, imgs


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


def preprocess_audio(
    msgs: Dict[str, Dict[str, List[Any]]],
    topic: List[str],
    device: str,
    split: bool = False,
    start_end_timestamps: Optional[List[float]] = None,
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

    start_end_timestamps = [
        start_end_timestamps[0] - T_OFFSET,
        start_end_timestamps[1] + T_OFFSET,
    ]
    times_idx = np.searchsorted(raw_times, start_end_timestamps)
    times = [raw_times[times_idx[0] : times_idx[1]]]

    times = [np.min([time - np.arange(len(time)) * dt]) + np.arange(len(time)) * dt for time in times]
    audio = [[data[times_idx[0] : times_idx[1]]] for data in raw_audio]
    times = [time[0] + np.arange(len(time) * hop) / 48000 for time in times]
    audio = [[subdata.reshape(-1) for subdata in data] for data in audio]

    if not split:
        times = np.concatenate(times, axis=0)
        audio = [np.concatenate(mic, axis=0) for mic in audio]

    return times, audio


def get_pressure(
    ros_data: Dict[str, Dict[str, List[Any]]],
    device: str,
    start_end_timestamps: List[float],
    data_synced: Dict[str, Dict[str, Dict[str, Any]]],
):
    topic = "pressure_topic"
    sample_rate = 200
    time, data = preprocess_pressure(ros_data, [topic], device, start_end_timestamps=start_end_timestamps)
    data_interp = interp1d(time, data, axis=0, kind="linear")
    synced_time = np.clip(
        start_end_timestamps[0] + np.arange(0, start_end_timestamps[1] - start_end_timestamps[0], 1 / sample_rate),
        start_end_timestamps[0],
        start_end_timestamps[1],
    )
    data_sync_time = data_interp(synced_time).astype(np.float32)
    data_synced[device]["pressure"] = {"time": synced_time, "data": data_sync_time}


def get_imu_acc(
    ros_data: Dict[str, Dict[str, List[Any]]],
    device: str,
    start_end_timestamps: List[float],
    data_synced: Dict[str, Dict[str, Dict[str, Any]]],
):
    topic = "imu_raw_topic"
    sample_rate = 400
    time, data = preprocess_imu(ros_data, [topic], device, "acc", start_end_timestamps=start_end_timestamps)
    data_interp = interp1d(time, data, axis=0, kind="cubic")
    synced_time = np.clip(
        start_end_timestamps[0] + np.arange(0, start_end_timestamps[1] - start_end_timestamps[0], 1 / sample_rate),
        start_end_timestamps[0],
        start_end_timestamps[1],
    )

    data_sync_time = data_interp(synced_time).astype(np.float32)
    data_synced[device]["imu_acc"] = {"time": synced_time, "data": data_sync_time}


def get_img(
    ros_data: Dict[str, Dict[str, List[Any]]],
    device: str,
    start_end_timestamps: List[float],
    data_synced: Dict[str, Dict[str, Dict[str, Any]]],
):
    topic = "image_raw/compressed"
    sample_rate = 30
    time, data = preprocess_image(
        ros_data,
        [topic],
        device,
        preload=False,
        start_end_timestamps=start_end_timestamps,
    )
    data_interp = interp1d(time, data, axis=0, kind="nearest")
    synced_time = np.clip(
        start_end_timestamps[0] + np.arange(0, start_end_timestamps[1] - start_end_timestamps[0], 1 / sample_rate),
        start_end_timestamps[0],
        start_end_timestamps[1],
    )
    data_sync_time = data_interp(synced_time).astype(int)
    data_synced[device]["img"] = {"time": synced_time, "data": data_sync_time}


def get_mic_fbank(
    ros_data: Dict[str, Dict[str, List[Any]]],
    device: str,
    start_end_timestamps: List[float],
    data_synced: Dict[str, Dict[str, Dict[str, Any]]],
):
    topic = ["mic_0", "mic_1"]
    sample_rate = 48000
    time, data = preprocess_audio(ros_data, topic, device, start_end_timestamps=start_end_timestamps)
    data_interp = [interp1d(time, d_mic, axis=0, kind="nearest") for d_mic in data]
    synced_time = np.clip(
        start_end_timestamps[0] + np.arange(0, start_end_timestamps[1] - start_end_timestamps[0], 1 / sample_rate),
        start_end_timestamps[0],
        start_end_timestamps[1],
    )
    synced_time = np.clip(synced_time, time[0], time[-1])
    data_sync_time = [data_interp[i](synced_time).astype(np.float32) for i in range(2)]

    synced_fbank_data = [
        kaldi.fbank(
            torch.from_numpy(data[None]),
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_length=5,
            frame_shift=2.5,
            high_freq=16000,
            low_freq=1,
        )
        .numpy()
        .astype(np.float32)
        for data in data_sync_time
    ]

    mic_fbank_sample_rate = 400
    mic_fbank_frame_length = 5
    mic_fbank_data = synced_fbank_data
    mic_fbank_time = np.clip(
        np.arange(
            start_end_timestamps[0] + mic_fbank_frame_length / 1000,
            start_end_timestamps[1],
            1 / mic_fbank_sample_rate,
        ),
        start_end_timestamps[0],
        start_end_timestamps[1],
    )
    data_synced[device]["mic_fbank"] = {"time": mic_fbank_time, "data": mic_fbank_data}


def get_allegro_joint_cmd(
    ros_data: Dict[str, Dict[str, List[Any]]],
    device: str,
    start_end_timestamps: List[float],
    data_synced: Dict[str, Dict[str, Dict[str, Any]]],
):
    topic = "allegroHand/joint_cmd"
    sample_rate = 20
    time, data = preprocess_allergro_joint_states(ros_data, [topic], start_end_timestamps=start_end_timestamps)
    data_interp = interp1d(time, data, axis=0, kind="linear")
    device = topic.split("/")[0]
    topic = topic.split("/")[1]
    synced_time = np.clip(
        start_end_timestamps[0] + np.arange(0, start_end_timestamps[1] - start_end_timestamps[0], 1 / sample_rate),
        start_end_timestamps[0],
        start_end_timestamps[1],
    )
    data_sync_time = data_interp(synced_time).astype(np.float32)
    data_synced[device][topic] = {"time": synced_time, "data": data_sync_time}


def get_idxs_wrt_time(
    topic: str,
    device: str,
    reference_time: np.ndarray,
    synced_data: Dict[str, Dict[str, Dict[str, Any]]],
    dev_msg_starts: Dict[str, List[int]],
):
    sample_rate = modalities_config[topic]["sample_rate"]
    length = modalities_config[topic]["length"]
    stride = modalities_config[topic]["stride"]
    msg_time = synced_data[device][topic]["time"]
    msg_end = np.round((reference_time - msg_time[0]) * sample_rate).clip(0, len(msg_time) - 1).astype(int)
    msg_start = msg_end - (length - 1) * stride
    dev_msg_starts[device].append(msg_start.astype(int))


def get_sync_pressure(query_idxs, data_synced, dataset, device, episode_folder):
    topic = "pressure"
    length = modalities_config[topic]["length"]
    stride = modalities_config[topic]["stride"]
    sub_idx = query_idxs[3]

    pressure = data_synced[device][topic]["data"][sub_idx : sub_idx + length * stride : stride]
    # pressure_time = data_synced[device][topic]["time"][
    #     sub_idx : sub_idx + length * stride : stride
    # ]
    dataset[device][topic].append(pressure)


def get_sync_img(query_idxs, data_synced, dataset, device, episode_folder, rect=None):
    topic = "img"
    length = modalities_config[topic]["length"]
    stride = modalities_config[topic]["stride"]
    sub_idx = query_idxs[0]
    imgs = data_synced[device][topic]["data"][sub_idx : sub_idx + length * stride : stride]
    image = []
    for img in imgs:
        img_path = os.path.join(episode_folder, device, "img", f"{img}")
        with open(img_path, "rb") as file:
            img_data = file.read()
        sensor_img = PILImage.open(io.BytesIO(img_data))

        if device == "d360_0":
            rect = (425, 137, 865, 577)
        elif device == "d360_1":
            rect = (425, 154, 865, 594)
        elif device == "d360_2":
            rect = (434, 137, 874, 577)
        else:
            raise ValueError(f"Unknown device: {device}")
        sensor_img = sensor_img.crop(rect)
        sensor_img = np.array(sensor_img.resize((224, 224)), dtype=np.int16)
        sensor_img = sensor_img.astype(np.uint8)
        image.append(sensor_img)
    image = np.stack(image, axis=0)
    dataset[device][topic].append(image)


def get_sync_mic_fbank(query_idxs, data_synced, dataset, device, episode_folder):
    topic = "mic_fbank"
    length = modalities_config[topic]["length"]
    stride = modalities_config[topic]["stride"]
    sub_idx = query_idxs[1]

    mic_fbank_data = [d[sub_idx : sub_idx + length * stride : stride] for d in data_synced[device][topic]["data"]]
    # mic_fbank_time = data_synced[device][topic]["time"][
    #     sub_idx : sub_idx + length * stride : stride
    # ]
    dataset[device][topic].append(mic_fbank_data)


def get_sync_imu_acc(query_idxs, data_synced, dataset, device, episode_folder):
    topic = "imu_acc"
    length = modalities_config[topic]["length"]
    stride = modalities_config[topic]["stride"]
    sub_idx = query_idxs[2]

    imu_acc = data_synced[device][topic]["data"][sub_idx : sub_idx + length * stride : stride]
    # imu_acc_time = data_synced[device][topic]["time"][
    #     sub_idx : sub_idx + length * stride : stride
    # ]
    dataset[device][topic].append(imu_acc)
