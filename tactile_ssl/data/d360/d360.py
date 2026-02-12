# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
D360 dataset class for loading and processing data from the D360 dataset (in pickle format).
This class handles the loading of device and system data, synchronization of time,
and extraction of tactile images and time-series data. 

Labels are extracted from dataset metadata (if available, e.g object,material or other tactile properties). 
Labels are used for online probes only (if available) and are not required for representation learning.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import os
import yaml
from omegaconf import DictConfig

from tactile_ssl.data.d360.utils import intersect_intervals
from tactile_ssl.utils.logging import get_pylogger
from PIL import Image
import io

logger = get_pylogger(__name__)


class D360:
    def __init__(self, config: DictConfig, synced: bool):
        self.sequences = []
        self.devices = []
        self.device_topics = []
        self.device_cfg = config.device if config.device is not None else {}
        self.system_cfg = config.system if config.system is not None else {}
        self.label_cfg = config.label if config.label is not None else {}
        self.device_data = {key: {} for key in self.device_cfg}
        self.system_data = {key: {} for key in self.system_cfg}
        self.label = {key: [] for key in self.label_cfg}
        self.num_devices = []

        for (_, dev_spec), (_, dev_data) in zip(self.device_cfg.items(), self.device_data.items()):
            if dev_spec.topic is not None:
                for topic in dev_spec.topic:
                    self.device_topics.append(topic)
                if "load" in dev_spec:
                    dev_data["init_time"] = []
                    dev_data["init_data"] = []
                    dev_data["init_seg"] = []
                    dev_data["interp"] = []
            dev_data["time"] = []
            dev_data["data"] = []
            dev_data["interval"] = []

        self.system_topics = []
        for (_, sys_spec), (_, sys_data) in zip(self.system_cfg.items(), self.device_cfg.items()):
            if sys_spec.topic is not None:
                for topic in sys_spec.topic:
                    self.system_topics.append(topic)
                if "load" in sys_spec:
                    sys_data["init_time"] = []
                    sys_data["init_data"] = []
                    sys_data["init_seg"] = []
                    sys_data["interp"] = []
            sys_data["time"] = []
            sys_data["data"] = []
            sys_data["interval"] = []

        self.device_data["img"]["rect"] = []
        self.device_data["img"]["bg"] = [] if self.device_cfg.img.rm_bg else None

        self.synced = synced
        self.min_time = config.min_time

        for topic, data in self.device_data.items():
            for subtopic, subdata in data.items():
                setattr(self, f"{topic}_{subtopic}", subdata)

        for topic, data in self.system_data.items():
            for subtopic, subdata in data.items():
                setattr(self, f"{topic}_{subtopic}", subdata)

        for topic, label in self.label.items():
            setattr(self, f"{topic}_label", label)

    def add_data(
        self,
        sequence: str,
        devices: Optional[List[str]] = None,
    ):
        logger.info(f"Loading {sequence}")

        self.load_data(sequence, devices)
        self.load_label(sequence)
        self.sync_time()
        self.sync_data()
        self.clear_data()

        logger.info(f"Loaded {sequence}")

    def extract_image(self, seq, dev, seg, idx):
        assert self.img_rect is not None
        sequence = self.sequences[seq]
        device = self.devices[seq][dev]
        rect = self.img_rect[seq][device]
        img_data = self.img_data[seq][dev][seg][idx]
        if not isinstance(img_data, bytes):
            img_path = os.path.join(sequence, device, "img", f"{img_data}")
            with open(img_path, "rb") as file:
                img_data = file.read()
        img = np.array(Image.open(io.BytesIO(img_data)), dtype=np.int16)
        img = img[rect[0] : rect[1], rect[2] : rect[3]]
        if self.img_bg is not None:
            bg = self.img_bg[device][rect[0] : rect[1], rect[2] : rect[3]]
            img = (img - bg) / 255 + 0.5
        else:
            img = img / 255
        return img.astype(np.float32)

    def check_topics(self, data: Dict[str, Any], topics: List[str]):
        for topic in topics:
            if topic not in data.keys():
                return False

            if topic == "pressure_topic":
                pressure = np.array(data["pressure_topic"]["pressure"])
                if (pressure >= 1000).sum() / len(pressure) <= 0.1:
                    return False
        return True

    def load_label(self, sequence: str):
        with open(os.path.join(sequence, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)
        for topic, cfg in self.label_cfg.items():
            self.label[topic].append(metadata.get(cfg.key, "none"))

    def load_data(self, sequence: str, devices: Optional[List[str]] = None):
        ros_msgs = pd.read_pickle(os.path.join(sequence, "data.pickle"))
        if devices is None:
            devices = [dev for dev in ros_msgs.keys() if dev.startswith("d360_") and dev.rsplit("_", 1)[-1].isdigit()]

        init_devs = devices
        devices = []
        for init_dev in init_devs:
            if self.check_topics(ros_msgs[init_dev], self.device_topics):
                devices.append(init_dev)
            else:
                assert not self.synced, "All device data should be available when synced==True"
                logger.warning(f"Ignore {init_dev} due to missing data")

        self.sequences.append(sequence)
        self.devices.append(devices)
        self.num_devices.append(len(devices))

        for cfg_name in self.device_cfg:
            dev_cfg = self.device_cfg[cfg_name]
            dev_data = self.device_data[cfg_name]
            if "load" in dev_cfg:
                dev_cfg.load(msgs=ros_msgs, sequence=sequence, devices=devices, config=dev_cfg, data=dev_data)

        for cfg_name in self.system_cfg:
            sys_cfg = self.system_cfg[cfg_name]
            sys_data = self.system_data[cfg_name]
            if "load" in sys_cfg:
                sys_cfg.load(msgs=ros_msgs, sequence=sequence, config=sys_cfg, data=sys_data)

    def sync_time(self):
        if self.synced:
            init_interval = [
                init_dev_seg
                for (_, cfg), (_, data) in zip(self.device_cfg.items(), self.device_data.items())
                if "load" in cfg
                for init_dev_seg in data["init_seg"][-1]
            ] + [
                data["init_seg"][-1]
                for (_, cfg), (_, data) in zip(self.system_cfg.items(), self.system_data.items())
                if "load" in cfg
            ]

            synced_interval = init_interval[0] if init_interval else []
            for interval in init_interval[1:]:
                synced_interval = intersect_intervals(synced_interval, interval)

            synced_interval = [interval for interval in synced_interval if interval[1] - interval[0] >= self.min_time]

            for cfg_name in self.device_cfg:
                dev_cfg = self.device_cfg[cfg_name]
                dev_data = self.device_data[cfg_name]
                synced_time = [
                    np.clip(
                        interval[0] + np.arange(0, interval[1] - interval[0], 1 / dev_cfg.sample_rate),
                        interval[0],
                        interval[1],
                    )
                    for interval in synced_interval
                ]
                dev_data["interval"].append(synced_interval)
                dev_data["time"].append(synced_time)

            for cfg_name in self.system_cfg:
                sys_cfg = self.system_cfg[cfg_name]
                sys_data = self.system_data[cfg_name]
                synced_time = [
                    np.clip(
                        interval[0] + np.arange(0, interval[1] - interval[0], 1 / sys_cfg.sample_rate),
                        interval[0],
                        interval[1],
                    )
                    for interval in synced_interval
                ]
                sys_data["interval"].append(synced_interval)
                sys_data["time"].append(synced_time)
        else:
            assert len(self.system_cfg) == 0, "There should be no system topics when synced==False."
            devices = self.devices[-1]
            num_devs = len(devices)

            synced_interval = {cfg_name: [] for cfg_name in self.device_cfg}
            synced_time = {cfg_name: [] for cfg_name in self.device_cfg}

            for dev in range(num_devs):
                init_dev_interval = [
                    data["init_seg"][-1][dev]
                    for (_, cfg), (_, data) in zip(self.device_cfg.items(), self.device_data.items())
                    if "load" in cfg
                ]

                synced_dev_interval = init_dev_interval[0] if init_dev_interval else []
                for interval in init_dev_interval[1:]:
                    synced_dev_interval = intersect_intervals(synced_dev_interval, interval)

                if "load" in self.device_cfg.img:
                    img_sample_rate = self.device_cfg.img.sample_rate
                    img_start = self.device_data["img"]["init_seg"][-1][dev][0][0]
                    synced_dev_interval = [
                        img_start
                        + np.array(
                            [
                                np.ceil((interval[0] - img_start) * img_sample_rate),
                                np.floor((interval[1] - img_start) * img_sample_rate),
                            ]
                        )
                        / img_sample_rate
                        for interval in synced_dev_interval
                    ]

                synced_dev_interval = [
                    interval for interval in synced_dev_interval if interval[1] - interval[0] >= self.min_time
                ]

                for cfg_name, dev_cfg in self.device_cfg.items():
                    synced_interval[cfg_name].append(synced_dev_interval)
                    synced_time[cfg_name].append(
                        [
                            np.clip(
                                interval[0] + np.arange(0, interval[1] - interval[0], 1 / dev_cfg.sample_rate),
                                interval[0],
                                interval[1],
                            )
                            for interval in synced_dev_interval
                        ]
                    )

            for cfg_name, data in self.device_data.items():
                data["interval"].append(synced_interval[cfg_name])
                data["time"].append(synced_time[cfg_name])

    def clear_data(self):
        for _, dev_data in self.device_data.items():
            for key in dev_data:
                if "init" in key or key == "interp":
                    assert len(dev_data[key]) == 1
                    del dev_data[key][-1]

        for _, sys_data in self.system_data.items():
            for key in sys_data:
                if "init" in key or key == "interp":
                    assert len(sys_data[key]) == 1
                    del sys_data[key][-1]

    def sync_data(self):
        for _, cfg in self.device_cfg.items():
            if "sync" in cfg:
                cfg.sync(config=self.device_cfg, data=self.device_data, synced=self.synced)

        for _, cfg in self.system_cfg.items():
            if "sync" in cfg:
                cfg.sync(config=self.system_cfg, data=self.system_data, synced=self.synced)
