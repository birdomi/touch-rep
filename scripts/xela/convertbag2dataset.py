# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# This script converts ROS bag files to a pkl dataset format that is used for Sparsh-skin pretraining
# Run this script in a ROS enabled environment
import argparse
import glob
import os
import pickle
from pathlib import Path
from typing import List

import cv2
import imageio
import numpy as np
import pandas as pd
import rosbag2_py
from cv_bridge import CvBridge
from mcap_ros2.reader import McapROS2Message, read_ros2_messages
from PIL import Image
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from torch import Value
from tqdm import tqdm

from tactile_ssl.data.d360.utils import load_ros_data, process_d360_rosmsgs

ALLEGRO_TOPIC = "/allegroHand/joint_states"
FT_TOPIC = "/netft_data2"
MECA_EE_POSE_TOPIC = "/robot1/MecademicRobot_pose_fb"
FRANKA_JOINT_TOPIC = "/franka/joint_states"
OBJECT_MARKER_TOPIC = "/object_marker"
XELA_TOPIC = "/xServTopic"
CAMERA_NAMES = ["left", "top", "right", "wrist"]


def save_image(img, save_path):
    img = Image.fromarray(img.astype("uint8"), "RGB")
    img.save(save_path)


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()


def save_depthmaps(depth, depth_scale, save_path):
    np.savez_compressed(save_path, depth=np.array(depth), depth_scale=depth_scale)


def compressed_img_msg_to_array(msg):
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return np.asarray(cv_image)


def img_msg_to_array(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return np.asarray(cv_image)


def extract_object_pose(msg_data):
    object_pose = []
    for i in tqdm(range(0, len(msg_data[OBJECT_MARKER_TOPIC]))):
        timestamp, data = (
            msg_data["/object_marker"][i][0],
            msg_data["/object_marker"][i][1],
        )
        object_pose_val = np.array(
            [
                timestamp,
                data.pose.position.x,
                data.pose.position.y,
                data.pose.position.z,
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w,
            ]
        )
        object_pose.append(object_pose_val)
    object_pose = np.array(object_pose)
    return object_pose


def extract_reskin_topics(msg_data):
    reskin_topics = [topic for topic in msg_data.keys() if "reskin" in topic]
    reskin_data = {}
    for topic in reskin_topics:
        reskin_data[f"{topic}"] = []
        for i in tqdm(range(0, len(msg_data[topic]))):
            timestamp, data = (
                msg_data[topic][i][0],
                msg_data[topic][i][1].magnetic_field,
            )
            reskin_val = [timestamp, data.x, data.y, data.z]

            reskin_data[f"{topic}"].append(reskin_val)
        reskin_data[f"{topic}"] = np.array(reskin_data[f"{topic}"])
    return reskin_data


def extract_xela_topics(msg_data):
    xela_data = []
    xela_forces = []
    for i in tqdm(range(0, len(msg_data[XELA_TOPIC]))):
        timestamp, data = msg_data[XELA_TOPIC][i][0], msg_data[XELA_TOPIC][i][1]
        serialized_data = []
        serialized_force = []
        for sensor in data.sensors:
            for i, taxel in enumerate(sensor.taxels):
                taxel_array = np.array([timestamp, taxel.x, taxel.y, taxel.z])
                serialized_data.append(taxel_array)
            for i, forces in enumerate(sensor.forces):
                force_array = np.array([timestamp, forces.x, forces.y, forces.z])
                serialized_force.append(force_array)
        serialized_force = np.array(serialized_force)
        serialized_data = np.array(serialized_data)
        xela_data.append(serialized_data)
        xela_forces.append(serialized_force)
    return xela_data, xela_forces


def extract_force_topic(msg_data, force_topic):
    force_data = []
    for i in tqdm(range(0, len(msg_data[force_topic]))):
        timestamp, data = (
            msg_data[force_topic][i][0],
            msg_data[force_topic][i][1],
        )
        force_val = np.array([timestamp, data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        force_data.append(force_val)
    force_data = np.array(force_data)
    return force_data


def extract_ee_pose_topic(msg_data, ee_pose_topic):
    robot_pose = []
    for i in tqdm(range(0, len(msg_data[ee_pose_topic]))):
        timestamp, data = (
            msg_data[ee_pose_topic][i][0],
            msg_data[ee_pose_topic][i][1],
        )
        ee_pose_val = np.array(
            [
                timestamp,
                data.position.x,
                data.position.y,
                data.position.z,
                data.orientation.x,
                data.orientation.y,
                data.orientation.z,
                data.orientation.w,
            ]
        )
        robot_pose.append(ee_pose_val)
    robot_pose = np.array(robot_pose)
    return robot_pose


def extract_color_images(msg_data, camera_topic, out_dir):
    color_timestamps = []
    camera_name = camera_topic.split("_")[0]
    is_compressed = "compressed" in camera_topic
    os.makedirs(f"{out_dir}/{camera_name}/color", exist_ok=True)
    for i in tqdm(range(0, len(msg_data[camera_topic]))):
        timestamp, data = msg_data[camera_topic][i][0], msg_data[camera_topic][i][1]
        rgb_image = compressed_img_msg_to_array(data) if is_compressed else img_msg_to_array(data)
        save_image(rgb_image, f"{out_dir}/{camera_name}/color/{i:06d}.jpg")
        color_timestamps.append(timestamp)
    color_timestamps = np.array(color_timestamps)
    np.savetxt(f"{out_dir}/{camera_name}/color/timestamps.txt", color_timestamps)


def extract_depth_images(msg_data, depth_topic, out_dir):
    depth_timestamps = []
    os.makedirs(f"{out_dir}/realsense/depth", exist_ok=True)
    if depth_topic in msg_data.keys():
        for i in tqdm(range(0, len(msg_data[depth_topic]))):
            timestamp, data = msg_data[depth_topic][i][0], msg_data[depth_topic][i][1]
            depth_image = img_msg_to_array(data)
            imageio.imwrite(f"{out_dir}/realsense/depth/{i:06d}.png", depth_image.astype(np.uint16))
            depth_timestamps.append(timestamp)
        depth_timestamps = np.array(depth_timestamps)
        np.savetxt(f"{out_dir}/realsense/depth/timestamps.txt", depth_timestamps)


def extract_digit_images(msg_data, topic, out_dir, finger_name):
    digit_timestamps = []
    os.makedirs(f"{out_dir}/digit_{finger_name}", exist_ok=True)
    for i in tqdm(range(0, len(msg_data[topic]))):
        timestamp, data = msg_data[topic][i][0], msg_data[topic][i][1]
        rgb_image = compressed_img_msg_to_array(data)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).astype("uint8")
        save_image(rgb_image, f"{out_dir}/digit_{finger_name}/{i:06d}.jpg")
        digit_timestamps.append(timestamp)
    digit_timestamps = np.array(digit_timestamps)
    np.savetxt(f"{out_dir}/digit_{finger_name}/timestamps.txt", digit_timestamps)


def extract_joint_states(msg_data, joint_states_topic):
    joint_states_np = []
    for i in tqdm(range(0, len(msg_data[joint_states_topic]))):
        timestamp, data = (
            msg_data[joint_states_topic][i][0],
            msg_data[joint_states_topic][i][1],
        )
        position = np.array(data.position)
        effort = np.array(data.effort)

        joint_state = [timestamp] + list(position) + list(effort)
        joint_states_np.append(np.array(joint_state))
    return np.array(joint_states_np)


def create_bag_reader(input_bag: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def read_messages(
    reader: rosbag2_py.SequentialReader,
    topics_to_extract: List[rosbag2_py.TopicMetadata],
):
    topics_in_bag = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topics_in_bag:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    def filter(topic_name):
        for topic in topics_to_extract:
            if topic_name == topic.name:
                return True
        return False

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        if not filter(topic):
            # print(f"topic skipped: {topic}")
            continue
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader


def sanity_check_requested_topics(
    args: argparse.Namespace,
    bag_path: str,
    topics_in_bag: List[rosbag2_py.TopicMetadata],
    out_dir: str,
) -> List[rosbag2_py.TopicMetadata]:
    is_allegro = args.allegro
    is_ft = args.ft
    is_ee_pose = args.ee_pose
    is_camera = args.camera
    is_franka = args.franka
    is_object_pose = args.object_pose
    is_compressed = args.compressed
    is_xela = args.xela
    is_d360 = args.d360

    topic_names_in_bag = [topic.name for topic in topics_in_bag]

    topics_to_extract = []
    os.makedirs(out_dir, exist_ok=True)

    if is_allegro:
        if ALLEGRO_TOPIC not in topic_names_in_bag:
            raise ValueError(f"{ALLEGRO_TOPIC} not in bag: {bag_path}")
        os.makedirs(out_dir + "/allegro", exist_ok=True)
        topics_to_extract += [ALLEGRO_TOPIC]

    if is_ft:
        if FT_TOPIC not in topic_names_in_bag:
            raise ValueError(f"{FT_TOPIC} not in bag: {bag_path}")
        topics_to_extract += [FT_TOPIC]

    if is_ee_pose:
        if MECA_EE_POSE_TOPIC in topic_names_in_bag:
            raise ValueError(f"{MECA_EE_POSE_TOPIC} not in bag: {bag_path}")
        topics_to_extract += [MECA_EE_POSE_TOPIC]

    if is_camera:
        camera_topic_suffix = "_camera/color/image_raw/compressed" if is_compressed else "_camera/color/image_raw"
        camera_topics_to_extract = [f"/{camera_name}{camera_topic_suffix}" for camera_name in CAMERA_NAMES]
        for camera_topic_to_extract in camera_topics_to_extract:
            if camera_topic_to_extract not in topic_names_in_bag:
                raise ValueError(f"{camera_topic_to_extract} not in {bag_path}")
            else:
                topics_to_extract.append(camera_topic_to_extract)

        for camera_name in CAMERA_NAMES:
            os.makedirs(out_dir + f"/{camera_name}", exist_ok=True)

    if is_franka:
        if FRANKA_JOINT_TOPIC not in topic_names_in_bag:
            raise ValueError(f"{FRANKA_JOINT_TOPIC} not in {bag_path}")
        os.makedirs(out_dir + "/franka", exist_ok=True)
        topics_to_extract += [FRANKA_JOINT_TOPIC]

    if is_object_pose:
        if OBJECT_MARKER_TOPIC not in topic_names_in_bag:
            raise ValueError(f"{OBJECT_MARKER_TOPIC} not in {bag_path}")
        topics_to_extract += [OBJECT_MARKER_TOPIC]

    if is_xela:
        if XELA_TOPIC not in topic_names_in_bag:
            raise ValueError(f"{XELA_TOPIC} not in {bag_path}")
        os.makedirs(out_dir + "/xela", exist_ok=True)
        topics_to_extract += [XELA_TOPIC]

    if is_d360:
        num_devices = 3
        device_topic = [f"/d360_{i}" for i in range(num_devices)]
        d360_subtopics = [
            "image_raw/compressed",
            "mic_0",
            "mic_1",
            "imu_raw_topic",
            "imu_quat_topic",
            "pressure_topic",
        ]
        d360_topics = [device_topic + "/" + subtopic for device_topic in device_topic for subtopic in d360_subtopics]
        for d360_topic in d360_topics:
            if d360_topic not in topic_names_in_bag:
                raise ValueError(f"{d360_topic} not in {bag_path}")
            # We are reloading the file a second time to load d360 images, so we can ignore from here
            topics_to_extract.append(d360_topic)
        os.makedirs(out_dir + "/d360", exist_ok=True)
    return topics_to_extract


def bag2dataset(args, bag_path, out_dir):
    bag_reader = create_bag_reader(bag_path)
    topics_in_bag = bag_reader.get_all_topics_and_types()

    topic_names_to_extract = sanity_check_requested_topics(args, bag_path, topics_in_bag, out_dir)
    topics_to_extract = []
    for topic in topics_in_bag:
        if topic.name in topic_names_to_extract:
            topics_to_extract.append(topic)

    msg_data = {}
    for topic, msg, timestamp in read_messages(bag_reader, topics_to_extract):
        # For some weird reason it seems franka topics have a very different timestamp (maybe bad communication)
        if topic == FRANKA_JOINT_TOPIC:
            stamp = timestamp / 1e9
        elif hasattr(msg, "header"):
            if msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9  # Convert ns to s
            else:
                if len(msg_data[topic]) <= 0:
                    print("WARN: {} has no time stamps in header, using bag stamps".format(topic))
                stamp = timestamp / 1e9  # Convert ns to s
        else:
            stamp = timestamp / 1e9
        if topic not in msg_data.keys():
            msg_data[topic] = []
        msg_data[topic].append([stamp, msg])

    del bag_reader

    print("Selected following topics from ROS bag:")
    for topic in msg_data.keys():
        print(f"{topic}")

    t0 = -np.inf
    for topic in msg_data.keys():
        assert len(msg_data[topic]) > 0, f"Missing topic: {topic}"
        if msg_data[topic][0][0] > t0:
            t0 = msg_data[topic][0][0]
        for i in range(len(msg_data[topic]) - 1):
            assert msg_data[topic][i][0] <= msg_data[topic][i + 1][0], (
                f"Out of order: {topic} at {i}: {msg_data[topic][i][0]} > {msg_data[topic][i + 1][0]}"
            )

    # Convert timestamps to be w.r.t t0
    for topic in msg_data.keys():
        for i in range(len(msg_data[topic])):
            msg_data[topic][i][0] = msg_data[topic][i][0] - t0
        print(f"Start time for {topic}: {msg_data[topic][0][0]}")

    is_xela = XELA_TOPIC in topic_names_to_extract
    is_force = FT_TOPIC in topic_names_to_extract
    is_ee_pose = MECA_EE_POSE_TOPIC in topic_names_to_extract
    is_franka = FRANKA_JOINT_TOPIC in topic_names_to_extract
    is_allegro = ALLEGRO_TOPIC in topic_names_to_extract
    is_object_pose = OBJECT_MARKER_TOPIC in topic_names_to_extract
    is_d360 = any(["d360" in topic for topic in topic_names_to_extract])
    is_camera = any(["_camera/color" in topic for topic in topic_names_to_extract])

    if is_d360:
        d360_topics = [topic for topic in topic_names_to_extract if "d360" in topic]
        devices = sorted(list(set([topic.split("/", 2)[1] for topic in d360_topics])))
        devices, topics, raw_msgs = load_ros_data(
            bag_path,
            image=True,
            audio=True,
            imu=True,
            pressure=True,
            filter_devices=[3],
        )
        data = process_d360_rosmsgs(devices, topics, raw_msgs, reference_timestamp=t0)

        # Save d360 images as well
        for device in devices:
            print(f"Saving d360 images to {out_dir}/d360/{device}/img/ ... ")
            path_save_imgs = f"{out_dir}/d360/{device}/img/"
            os.makedirs(path_save_imgs, exist_ok=True)
            image_data = data[device]["image_raw/compressed"]["data"]
            for i, img_bytes in enumerate(image_data):
                file_path = f"{path_save_imgs}/{i}"
                with open(file_path, "wb") as file:
                    file.write(img_bytes)
            print("Saved d360 images")

        pd.to_pickle(data, f"{out_dir}/d360/data.pickle")

    if is_xela:
        xela_data, xela_forces = extract_xela_topics(msg_data)
        with open(f"{out_dir}/xela/data.pkl", "wb") as file:
            pickle.dump(xela_data, file)
        with open(f"{out_dir}/xela/forces.pkl", "wb") as file:
            pickle.dump(xela_forces, file)

    if is_franka:
        print("Extracting Franka topics")
        joint_states_np = extract_joint_states(msg_data, "/franka/joint_states")
        joint_states = {"joint_states": joint_states_np}
        with open(f"{out_dir}/franka/data.pkl", "wb") as file:
            pickle.dump(joint_states, file)

    if is_force or is_ee_pose:
        data = {}
        if is_force:
            force_data = extract_force_topic(msg_data, "/netft_data2")
            data["force"] = force_data
        if is_ee_pose:
            robot_pose = extract_ee_pose_topic(msg_data, "/robot1/MecademicRobot_pose_fb")
            data["ee_pose"] = robot_pose
        with open(f"{out_dir}/data.pkl", "wb") as file:
            pickle.dump(data, file)

    if is_camera:
        print("Extracting camera topics")
        camera_topics = [topic for topic in topic_names_to_extract if "_camera/color" in topic]
        for camera_topic in camera_topics:
            extract_color_images(msg_data, camera_topic, out_dir)

    if is_allegro:
        print("Extracting Allegro topics")
        joint_states_np = extract_joint_states(msg_data, ALLEGRO_TOPIC)
        joint_states = {"joint_states": joint_states_np}
        with open(f"{out_dir}/allegro/data.pkl", "wb") as file:
            pickle.dump(joint_states, file)

    if is_object_pose:
        print("Extracting object pose topics")
        object_pose = extract_object_pose(msg_data, OBJECT_MARKER_TOPIC)
        with open(f"{out_dir}/object_pose.pkl", "wb") as file:
            pickle.dump(object_pose, file)


def main():
    """ """
    parser = argparse.ArgumentParser("Convert bag file to dataset")
    parser.add_argument("--bag_path", type=str, help="Path to bag folder", required=True)
    parser.add_argument("--out_dir", type=str, help="Path to output folder", required=True)
    parser.add_argument("--ft", action="store_true", help="Extract force data")
    parser.add_argument("--ee_pose", action="store_true", help="Extract end effector pose data")
    parser.add_argument("--camera", action="store_true", help="Extract camera data")
    parser.add_argument("--compressed", action="store_true", help="Extract compressed data")
    parser.add_argument("--allegro", action="store_true", help="Extract allegro data")
    parser.add_argument("--franka", action="store_true", help="Extract franka data")
    parser.add_argument("--object-pose", action="store_true", help="Extract object pose data")
    parser.add_argument("--d360", action="store_true", help="Extract d360 data")
    parser.add_argument("--xela", action="store_true", help="Extract xela force data")

    args = parser.parse_args()

    bag_path = args.bag_path
    out_dir = args.out_dir
    print(f"Bag path: {bag_path}")
    print(f"Output directory: {out_dir}")
    bags = []
    for folder in Path(bag_path).iterdir():
        print(f"Checking folder: {folder}")
        if folder.is_dir():
            for bag in glob.glob(str(folder) + "/*.mcap"):
                bags.append(bag)
                print(f"Found bag: {bag}")
    bags = sorted(bags)
    for bag in bags:
        print(f"Processing bag: {bag}")
        out_name = Path(bag).parent.name
        bag2dataset(args, bag, out_dir + "/" + out_name)
        print("*********")


if __name__ == "__main__":
    main()
