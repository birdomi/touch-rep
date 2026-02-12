import argparse
import operator
import time
from typing import Any, Tuple

import numpy as np
import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.qos
import rclpy.qos_event
import rclpy.time
import torch
from cv_bridge import CvBridge
from d360_msgs.msg import AudioDataD360, ImuRawD360, PressureD360, SparshD360
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, UInt8MultiArray
from torchaudio.compliance import kaldi

from scripts.d360.sparsh_ros2_node.utils import (
    CircularBuffer,
    find_intersections,
    modalities_config,
)

with_sim_allegro = True
sim_d360_obs = False


class SparshInputNode(Node):
    def __init__(self, device: str, buffer_time_s: float = 1.0) -> None:
        super().__init__(f"Sparsh_obs_{device}")

        self.fps = 30

        topics = [
            "image_raw/compressed",
            "mic_0",
            "mic_1",
            "imu_raw_topic",
            "pressure_topic",
        ]
        self.topic_fps = {
            "image_raw/compressed": 30,
            "mic_0": 94,
            "mic_1": 94,
            "imu_raw_topic": 400,
            "pressure_topic": 200,
        }
        self.topic_fields = {
            "image_raw/compressed": ["timestamps", "data"],
            "mic_0": ["timestamps", "data"],
            "mic_1": ["timestamps", "data"],
            "imu_raw_topic": ["stamp", "sensor_ts", "data"],
            "pressure_topic": ["stamp", "sensor_ts", "data"],
        }

        self.device = device
        self.topics = topics
        self.reentrant_callback_group = ReentrantCallbackGroup()
        msg_cls = {topic: self.get_msg_cls_by_topic(topic) for topic in topics}

        report_lost_msg = rclpy.qos_event.SubscriptionEventCallbacks(
            message_lost=lambda x: self.get_logger().warn(f"lost {x.total_count_change}")
        )

        self.cv_bridge = CvBridge()
        self.subs = {}
        msg_types = [CompressedImage, AudioDataD360, AudioDataD360, ImuRawD360, PressureD360]
        callbacks = [
            self.image_callback,
            self.mic_0_callback,
            self.mic_1_callback,
            self.imu_callback,
            self.pressure_callback,
        ]
        for topic, msg_type, callback in zip(topics, msg_types, callbacks):
            self.subs[topic] = self.create_subscription(
                msg_type,
                f"{device}/{topic}",
                callback,
                10,
                event_callbacks=report_lost_msg,
                callback_group=self.reentrant_callback_group,
            )
            self.get_logger().info(f"Subscribing to {device}/{topic} of type {msg_cls[topic]}")

        # create buffers
        self.buffer_time_s = buffer_time_s
        self.buffer_size = {topic: int(self.topic_fps[topic] * self.buffer_time_s) for topic in topics}

        for k, buffer_size in self.buffer_size.items():
            self.get_logger().info(f"{k} Buffer size: {buffer_size}")

        self.d360_buffers = {}
        self.d360_buffers["image_raw/compressed"] = CircularBuffer(
            (self.buffer_size["image_raw/compressed"], 224, 224, 3), dtype=np.uint8
        )
        self.d360_buffers["mic_0"] = CircularBuffer((self.buffer_size["mic_0"], 512), dtype=np.float32)
        self.d360_buffers["mic_1"] = CircularBuffer((self.buffer_size["mic_1"], 512), dtype=np.float32)
        self.d360_buffers["imu_raw_topic"] = CircularBuffer((self.buffer_size["imu_raw_topic"], 3), dtype=np.float32)
        self.d360_buffers["pressure_topic"] = CircularBuffer(self.buffer_size["pressure_topic"], dtype=np.float32)

        self.d360_timestamp_buffers = {}
        self.reference_image_timestamp = None
        self.d360_timestamp_buffers["mic_0"] = CircularBuffer((self.buffer_size["mic_0"]), dtype=np.float64)
        self.d360_timestamp_buffers["mic_1"] = CircularBuffer((self.buffer_size["mic_1"]), dtype=np.float64)
        self.d360_timestamp_buffers["imu_raw_topic"] = CircularBuffer(
            (self.buffer_size["imu_raw_topic"]), dtype=np.float64
        )
        self.d360_timestamp_buffers["pressure_topic"] = CircularBuffer(
            (self.buffer_size["pressure_topic"]), dtype=np.float64
        )

        self.start_time_buffers = None
        self.end_time_buffers = None

        self.start_time_obs = None
        self.end_time_obs = None

        # create d360_obs publisher
        self.timer = self.create_timer(1 / self.fps, self.get_d360_sparsh_inputs)
        self.obs_pub = self.create_publisher(SparshD360, f"{self.device}/sparsh_obs", 1)

    def get_msg_cls_by_topic(self, topic: str) -> Any:
        resolved_topic_name = self.resolve_topic_name(f"{self.device}/{topic}")

        time_start = time.time()
        while time.time() < time_start + 5:
            topic_type_dict = dict(self.get_topic_names_and_types())
            if resolved_topic_name in topic_type_dict:
                data_source_msg = topic_type_dict[resolved_topic_name][0]
                module, clspath = data_source_msg.split("/", 1)
                return operator.attrgetter(clspath.replace("/", "."))(__import__(module))
            else:
                time.sleep(0.5)  # allow some initialization time
        raise KeyError(f"{resolved_topic_name} not found in available keys {topic_type_dict.keys()}")

    def image_callback(self, msg: CompressedImage) -> None:
        topic = "image_raw/compressed"
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        self.d360_buffers[topic].push(img)
        self.reference_image_timestamp = timestamp

    def mic_callback(self, topic: str, msg: AudioDataD360) -> None:
        header_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        data = np.array(msg.data)
        self.d360_timestamp_buffers[topic].push(header_timestamp)
        self.d360_buffers[topic].push(data)

    def mic_0_callback(self, msg: AudioDataD360) -> None:
        self.mic_callback("mic_0", msg)

    def mic_1_callback(self, msg: AudioDataD360) -> None:
        self.mic_callback("mic_1", msg)

    def imu_callback(self, msg: ImuRawD360) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if msg.type != 6:
            return
        data = np.array([msg.x, msg.y, msg.z])
        self.d360_buffers["imu_raw_topic"].push(data)
        self.d360_timestamp_buffers["imu_raw_topic"].push(stamp)

    def pressure_callback(self, msg: PressureD360) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        data = np.array([msg.pressure])
        self.d360_buffers["pressure_topic"].push(data)
        self.d360_timestamp_buffers["pressure_topic"].push(stamp)

    def check_buffers_filled(self):

        if any([self.d360_buffers[topic].is_full() for topic in self.topics]):
            return True
        return False

    def pop_reference_timestamp(self):
        self.d360_timestamp_buffers["image_raw/compressed"].pop()

    def get_d360_sparsh_inputs(self):
        if self.start_time_buffers is None:
            self.start_time_buffers = time.time()

        if self.reference_image_timestamp is None:
            return None

        if not self.check_buffers_filled():
            return None

        if self.end_time_buffers is None:
            self.end_time_buffers = time.time()
            self.get_logger().info(f"Buffers filled up in {self.end_time_buffers - self.start_time_buffers} seconds")

        self.start_time_obs = time.time()
        obs = {}
        n_devices = 1
        if sim_d360_obs:
            obs["img"] = np.zeros((n_devices, 2, 3, 224, 224), dtype=np.float32)
            obs["mic_fbank"] = np.zeros((n_devices, 224, 128 * 2), dtype=np.float32)
            obs["imu_acc"] = np.zeros((n_devices, 224, 3), dtype=np.float32)
            obs["pressure"] = np.zeros((n_devices, 224, 1), dtype=np.float32)
            self.current_obs = obs
            try:
                self.publish_observation()
                self.get_logger().warn(f"SIMULATION d360 observation published OK")
            except Exception as e:
                self.get_logger().error(f"Error publishing observation: {e}")
            return
        # reference_timestamp = self.get_reference_timestamp()
        reference_timestamp = self.reference_image_timestamp
        got_img, obs["img"] = self.get_obs_image(reference_timestamp)
        got_mic, obs["mic_fbank"] = self.get_obs_audio(reference_timestamp)
        got_imu, obs["imu_acc"] = self.get_obs_imu(reference_timestamp)
        got_pressure, obs["pressure"] = self.get_obs_pressure(reference_timestamp)

        if not all([got_img, got_mic, got_imu, got_pressure]):
            return
        self.current_obs = obs
        try:
            self.publish_observation()
            self.end_time_obs = time.time()
            self.get_logger().info(f"Observation published in {self.end_time_obs - self.start_time_obs} seconds")
        except Exception as e:
            self.get_logger().error(f"Error publishing observation: {e}")
            return

    def publish_observation(self) -> None:
        msg = SparshD360()
        msg.header.stamp = self.get_clock().now().to_msg()

        s = self.current_obs["img"].shape
        data = np.ascontiguousarray(self.current_obs["img"]).flatten().tolist()
        msg.img = UInt8MultiArray()
        msg.img.data = data
        msg.img.layout.dim = [
            MultiArrayDimension(label="dim1", size=s[0], stride=s[0] * s[1] * s[2] * s[3] * s[4]),
            MultiArrayDimension(label="dim2", size=s[1], stride=s[1] * s[2] * s[3] * s[4]),
            MultiArrayDimension(label="dim3", size=s[2], stride=s[2] * s[3] * s[4]),
            MultiArrayDimension(label="dim4", size=s[3], stride=s[3] * s[4]),
            MultiArrayDimension(label="dim5", size=s[4], stride=s[4]),
        ]

        s = self.current_obs["mic_fbank"].shape
        data = np.array(self.current_obs["mic_fbank"]).astype(np.float32).flatten().tolist()
        msg.mic_fbank = Float32MultiArray()
        msg.mic_fbank.data = data
        msg.mic_fbank.layout.dim = [
            MultiArrayDimension(label="dim1", size=s[0], stride=s[0] * s[1] * s[2]),
            MultiArrayDimension(label="dim2", size=s[1], stride=s[1] * s[2]),
            MultiArrayDimension(label="dim3", size=s[2], stride=s[2]),
        ]

        s = self.current_obs["imu_acc"].shape
        data = np.array(self.current_obs["imu_acc"]).astype(np.float32).flatten().tolist()
        msg.imu_acc = Float32MultiArray()
        msg.imu_acc.data = data
        msg.imu_acc.layout.dim = [
            MultiArrayDimension(label="dim1", size=s[0], stride=s[0] * s[1] * s[2]),
            MultiArrayDimension(label="dim2", size=s[1], stride=s[1] * s[2]),
            MultiArrayDimension(label="dim3", size=s[2], stride=s[2]),
        ]

        s = self.current_obs["pressure"].shape
        data = np.array(self.current_obs["pressure"]).astype(np.float32).flatten().tolist()
        msg.pressure = Float32MultiArray()
        msg.pressure.data = data
        msg.pressure.layout.dim = [
            MultiArrayDimension(label="dim1", size=s[0], stride=s[0] * s[1] * s[2]),
            MultiArrayDimension(label="dim2", size=s[1], stride=s[1] * s[2]),
            MultiArrayDimension(label="dim3", size=s[2], stride=s[2]),
        ]
        self.obs_pub.publish(msg)

    def get_obs_image(self, ref_timestamp) -> Tuple[bool, np.ndarray]:
        length = modalities_config["img"]["length"]
        stride = modalities_config["img"]["stride"]

        try:
            diff = ref_timestamp - self.d360_timestamp_buffers["image_raw/compressed"]()
            idx = np.argmin(np.abs(diff))
            img_idxs = np.arange(idx - length * stride, idx, stride)
            imgs = []
            for img_idx in img_idxs:
                img = self.d360_buffers["image_raw/compressed"][img_idx]
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)
        except:
            imgs = np.zeros((length, 224, 224, 3), dtype=np.uint8)

        imgs = imgs.transpose(0, 3, 1, 2)
        imgs = np.expand_dims(imgs, axis=0)
        return True, imgs

    def get_obs_pressure(self, ref_timestamp) -> torch.Tensor:
        topic = "pressure_topic"
        length = modalities_config["pressure"]["length"]
        stride = modalities_config["pressure"]["stride"]

        raw_times = self.d360_timestamp_buffers[topic]()
        got_pressure = True
        if len(raw_times) < 1:
            pressure = np.zeros((length,), dtype=np.float32)
            self.get_logger().warn(f"{topic} len(raw_times) == 0, filling zeros")
            pressure = np.expand_dims(pressure, axis=0)
            pressure = np.expand_dims(pressure, axis=-1)
            got_pressure = False
            return got_pressure, pressure

        raw_data = self.d360_buffers[topic]()

        diff = ref_timestamp - raw_times
        idx = np.argmin(np.abs(diff))

        if idx == 0:
            # pressure and image are not synced
            self.get_logger().warn("pressure not synced")
            got_pressure = False
            idx = len(raw_times) - 1

        if idx < (length * stride):
            pressure = np.zeros((length,), dtype=np.float32)
            got_pressure = False
            self.get_logger().warn(f"{topic} not enough data in buffer, filling zeros")
        else:
            # self.get_logger().info(f"pressure Chosen timestamp: {raw_times[idx]}")
            pressure = raw_data[idx - length * stride : idx : stride]

        pressure = np.expand_dims(pressure, axis=0)
        pressure = np.expand_dims(pressure, axis=-1)
        return got_pressure, pressure

    def get_obs_imu(self, ref_timestamp) -> Tuple[bool, np.ndarray]:
        scale = 1 / 4096.0
        topic = "imu_raw_topic"
        length = modalities_config["imu_acc"]["length"]
        stride = modalities_config["imu_acc"]["stride"]

        raw_times = self.d360_timestamp_buffers[topic]()
        got_imu = True
        if len(raw_times) < 1:
            imu = np.zeros((length, 3), dtype=np.float32)
            self.get_logger().warn(f"{topic} len(raw_times) == 0, filling zeros")
            imu = np.expand_dims(imu, axis=0)
            got_imu = False
            return got_imu, imu

        raw_data = self.d360_buffers[topic]() * scale

        diff = ref_timestamp - raw_times
        idx = np.argmin(np.abs(diff))
        if idx == 0:
            # imu and image are not synced
            idx = len(raw_times) - 1
            got_imu = False
            self.get_logger().warn("imu not synced")

        if idx < length * stride:
            imu = np.zeros((length, 3), dtype=np.float32)
            got_imu = False
            self.get_logger().warn(f"{topic} not enough data in buffer, filling zeros")
        else:
            imu = raw_data[idx - length * stride : idx : stride]

        imu = np.expand_dims(imu, axis=0)
        return got_imu, imu

    def get_obs_audio(self, ref_timestamp) -> Tuple[bool, np.ndarray]:
        topic = "mic_fbank"
        length = modalities_config[topic]["length"]

        frame_shift = 2.5
        frame_length = 5
        audio_hz = 48000
        min_req_audio_samples = int(round((length * frame_shift + frame_length) * audio_hz / 1000))
        min_req_frames = (min_req_audio_samples // 512) + 1
        mic_data = []
        got_mic = True
        for mic in ["mic_0", "mic_1"]:
            mic_timestamps = self.d360_timestamp_buffers[mic]()
            if len(mic_timestamps) < 1:
                self.get_logger().warn(f"{mic} len mic_timestamps == 0, filling zeros")
                mic_samples = np.zeros((min_req_frames, 512), dtype=np.float32)
                mic_data.append(mic_samples)
                got_mic = False
                continue
            diff = ref_timestamp - mic_timestamps
            idx = np.argmin(np.abs(diff))
            if idx == 0:
                self.get_logger().warn(f"{mic} not synced")
                got_mic = False
                idx = len(mic_timestamps) - 1
            if idx < min_req_frames:
                self.get_logger().warn(f"{mic} not enough data in buffer, filling zeros")
                mic_samples = np.zeros((min_req_frames, 512), dtype=np.float32)
                got_mic = False
            else:
                mic_buffer = self.d360_buffers[mic]()
                # self.get_logger().info(f"{mic} chosen timestamp: {mic_timestamps[idx]}")
                mic_samples = mic_buffer[idx - min_req_frames : idx]
            mic_data.append(mic_samples)
        mic_data = np.concatenate(mic_data, axis=-1)

        mic_data = np.expand_dims(mic_data, axis=0)
        return got_mic, mic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="D360 Observation Node")
    parser.add_argument("--device", type=str, default="d360_0", help="Device name")
    parser.add_argument("--sim_d360_obs", action="store_true", help="Simulate D360 observations")
    args = parser.parse_args()

    global sim_d360_obs
    sim_d360_obs = args.sim_d360_obs

    rclpy.init()
    executor = rclpy.executors.SingleThreadedExecutor()
    node = SparshInputNode(device=args.device, buffer_time_s=5.0)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
