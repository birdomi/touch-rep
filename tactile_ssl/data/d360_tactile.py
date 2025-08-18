# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The dataset is designed for multisensoty touch representation learning using the D360 dataset.

This class loads touch data from the Digit 360 sensor for self-supervised representation learning. It:
1. Processes data paths containing training sequences in pickle format
2. Handles time alignment and synchronization across touch modalities
3. Treats each device in a multi-device sequence as a separate data source
4. Implements data augmentation and normalization for various touch inputs
5. Supports different touch sensing types available in the Digit 360 sensor, including pressure,
IMU, microphone, and image data

"""

from pathlib import Path
import torch.utils.data as data
from omegaconf import DictConfig
import numpy as np
from typing import List, Dict, Any
import os

from .d360.d360 import D360
import torch
import torchvision.transforms.functional as F
import random


class D360SSLDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        sequences: List[str],
    ):
        self.config = config
        self.data_cfg = config.data
        self.aug = config.aug
        self.synced = self.config.synced

        self.device_msgs = config.msg.device if config.msg.device is not None else []
        self.system_msgs = config.msg.system if config.msg.system is not None else []
        self.label_msgs = config.msg.label if config.msg.label is not None else []

        assert self._check_d360_data(), "Miss D360 data"

        if self.data_cfg.device is not None:
            for topic, cfg in self.data_cfg.device.items():
                setattr(self, topic, cfg)

        if self.data_cfg.system is not None:
            for topic, cfg in self.data_cfg.system.items():
                setattr(self, topic, cfg)

        if self.data_cfg.label is not None:
            for topic, cfg in self.data_cfg.label.items():
                setattr(self, topic, cfg)

        self.mic_fbank.sample_rate = 1000 / self.mic_fbank.frame_shift

        assert float(self.mic_fbank.length).is_integer()
        assert float(
            self.imu_acc.sample_rate * self.mic_fbank.stride / (self.mic_fbank.sample_rate * self.imu_acc.stride)
        ).is_integer()
        self.imu_acc.length = 1 + int(
            (self.mic_fbank.length - 1)
            * self.imu_acc.sample_rate  
            * self.mic_fbank.stride  
            / (self.mic_fbank.sample_rate * self.imu_acc.stride)  
        )

        self.aug.mic_fbank.scale = None

        self.data_path = config.data_path
        self.sequences = sequences
        assert Path(config.data_path).exists(), f"{self.data_path} does not exist"

        self.data_idxs = []
        self.sequence_sizes = []

        print(f"Loading data from {self.data_path}")

        self.d360 = D360(config.data, self.synced)

        for sequence in self.sequences:
            sequence = os.path.join(config.data_path, f"{sequence}")
            print(f"Loading {sequence}")
            self.d360.add_data(sequence=sequence, devices=None)

        for topic in self.d360.device_data:
            setattr(self, f"{topic}_data", getattr(self.d360, f"{topic}_data"))

        for topic in self.d360.system_data:
            setattr(self, f"{topic}_data", getattr(self.d360, f"{topic}_data"))

        for topic in self.d360.label:
            setattr(self, f"{topic}_label", getattr(self.d360, f"{topic}_label"))

        if self.data_cfg.device is not None:
            for topic, cfg in self.data_cfg.device.items():
                if "length" in cfg:
                    setattr(self, f"{topic}_length", cfg.length)

                if "stride" in cfg:
                    setattr(self, f"{topic}_stride", cfg.stride)

        if self.data_cfg.system is not None:
            for topic, cfg in self.data_cfg.system.items():
                if "length" in cfg:
                    setattr(self, f"{topic}_length", cfg.length)

                if "stride" in cfg:
                    setattr(self, f"{topic}_stride", cfg.stride)

        self.setup()

    def _check_d360_data(self):
        if self.data_cfg.device is None:
            return False

        for msg in ["img", "mic_wave", "mic_fbank", "imu_acc"]:
            if msg not in self.data_cfg.device:
                return False

        for msg in self.device_msgs:
            if msg not in self.data_cfg.device:
                return False

        if self.data_cfg.system is not None:
            for msg in self.system_msgs:
                if msg not in self.data_cfg.system:
                    return False

        if self.data_cfg.label is not None:
            for msg in self.label_msgs:
                if msg not in self.data_cfg.label:
                    return False

        return True

    def setup(self):
        sizes = [0]
        img_time = self.d360.img_time
        if self.synced:
            for seq, img_seq_time in enumerate(img_time):
                for seg, img_seg_time in enumerate(img_seq_time):
                    img_ids = np.arange(0, len(img_seg_time)).astype(int)
                    img_end = img_time[seq][seg][img_ids]
                    dev_msg_starts = []
                    sys_msg_starts = []
                    if len(img_end) > 0:
                        for msg in self.device_msgs:
                            assert msg in self.data_cfg.device
                            sample_rate = self.data_cfg.device[msg].sample_rate
                            length = self.data_cfg.device[msg].length
                            stride = self.data_cfg.device[msg].stride
                            msg_start = (
                                np.floor((img_end - self.d360.device_data[msg]["time"][seq][seg][0]) * sample_rate)
                                - (length - 1) * stride  
                            )
                            dev_msg_starts.append(msg_start.astype(int))

                        for msg in self.system_msgs:
                            assert msg in self.data_cfg.system
                            sample_rate = self.data_cfg.system[msg].sample_rate
                            length = self.data_cfg.system[msg].length
                            stride = self.data_cfg.system[msg].stride
                            msg_start = (
                                np.floor((img_end - self.d360.system_data[msg]["time"][seq][seg][0]) * sample_rate)
                                - (length - 1) * stride  
                            )
                            sys_msg_starts.append(msg_start.astype(int))

                    if dev_msg_starts or sys_msg_starts:
                        num_msgs = min(
                            [np.count_nonzero(msg_start >= 0) for msg_start in dev_msg_starts + sys_msg_starts]
                        )
                        if dev_msg_starts:
                            dev_msg_starts = np.array([msg_start[-num_msgs:] for msg_start in dev_msg_starts]).T
                        else:
                            dev_msg_starts = np.zeros([num_msgs, 0])

                        if sys_msg_starts:
                            sys_msg_starts = np.array([msg_start[-num_msgs:] for msg_start in sys_msg_starts]).T
                        else:
                            sys_msg_starts = np.zeros([num_msgs, 0])
                        self.data_idxs = self.data_idxs + [
                            ([seq, seg], dev_start.tolist(), sys_start.tolist())
                            for dev_start, sys_start in zip(dev_msg_starts, sys_msg_starts)
                        ]
                sizes.append(len(self.data_idxs))
        else:
            for seq, img_seq_time in enumerate(img_time):
                for dev, img_dev_time in enumerate(img_seq_time):
                    for seg, img_seg_time in enumerate(img_dev_time):
                        img_ids = np.arange(0, len(img_seg_time)).astype(int)
                        img_end = img_seg_time[img_ids]
                        dev_msg_starts = []
                        if len(img_end) > 0:
                            for msg in self.device_msgs:
                                assert msg in self.data_cfg.device
                                sample_rate = self.data_cfg.device[msg].sample_rate
                                length = self.data_cfg.device[msg].length
                                stride = self.data_cfg.device[msg].stride
                                msg_time = self.d360.device_data[msg]["time"][seq][dev][seg]
                                msg_end = (
                                    np.round((img_end - msg_time[0]) * sample_rate)
                                    .clip(0, len(msg_time) - 1)
                                    .astype(int)
                                )
                                msg_end = msg_end - (msg_time[msg_end] > img_seg_time)
                                msg_start = msg_end - (length - 1) * stride
                                dev_msg_starts.append(msg_start.astype(int))
                        if dev_msg_starts:
                            num_msgs = min([np.count_nonzero(msg_start >= 0) for msg_start in dev_msg_starts])
                            if num_msgs > 0:
                                dev_msg_starts = np.array([msg_start[-num_msgs:] for msg_start in dev_msg_starts]).T
                                self.data_idxs = self.data_idxs + [
                                    ([seq, dev, seg], dev_start.tolist(), []) for dev_start in dev_msg_starts
                                ]
                sizes.append(len(self.data_idxs))
        self.sequence_sizes = [sizes[n] - sizes[n - 1] for n in range(1, len(sizes))]

    def update_normalization(self, normalization: DictConfig):
        self.aug.mic_fbank.scale = np.array(normalization.mic_fbank.std, dtype=np.float32)
        self.aug.pressure.scale = np.array(normalization.pressure.std, dtype=np.float32)

    def remove(self, sequence: int, devices: List[int]):
        self.data_idxs = [index for index in self.data_idxs if index[0] != sequence or index[1] not in devices]

    def get_data(self, seg_idx, sub_idx, set_data):
        if self.synced:
            seq, seg = seg_idx
            num_devs = self.d360.num_devices[seq]
            return np.stack([set_data(seq, dev, seg, sub_idx) for dev in range(num_devs)])
        else:
            seq, dev, seg = seg_idx
            return set_data(seq, dev, seg, sub_idx)

    def get_img(self, sample, seg_idx, sub_idx):
        length = self.img_length
        stride = self.img_stride

        crop_ratio = (
            random.uniform(self.aug.img.crop_aug.range[0], self.aug.img.crop_aug.range[1])
            if random.random() < self.aug.img.crop_aug.p
            else None
        )
        flip_img = random.random() < self.aug.img.flip_aug.p
        rot_ang = (
            random.uniform(self.aug.img.rot_aug.range[0], self.aug.img.rot_aug.range[1])
            if random.random() < self.aug.img.rot_aug.p
            else None
        )

        def set_data(seq, dev, seg, sub_idx):
            img = np.stack(
                [
                    self.d360.extract_image(seq, dev, seg, idx)
                    for idx in range(sub_idx, sub_idx + length * stride, stride)
                ]
            )

            _, h, w, _ = img.shape
            assert h == w
            img_fill = img[:, int(0.105 * h), int(0.105 * h)].mean(axis=0).tolist()
            img = torch.tensor(img.transpose(0, 3, 1, 2))

            img_size = (
                (int(self.img.size[0] / crop_ratio), int(self.img.size[1] / crop_ratio))
                if crop_ratio is not None
                else self.img.size
            )

            img = F.resize(img, img_size)

            if flip_img:
                img = F.hflip(img)

            if rot_ang is not None:
                img = F.rotate(img, rot_ang, fill=img_fill)

            if crop_ratio is not None:
                img = F.center_crop(img, [self.img.size[0], self.img.size[1]])

            return img.numpy()

        sample["img"] = self.get_data(seg_idx, sub_idx, set_data)

    def get_mic_fbank(self, sample, seg_idx, sub_idx):
        length = self.mic_fbank_length
        stride = self.mic_fbank_stride

        def set_data(seq, dev, seg, sub_idx):
            mic_fbank = np.concatenate(
                [data[sub_idx : sub_idx + length * stride : stride] for data in self.mic_fbank_data[seq][dev][seg]],
                axis=-1,
            )
            if self.aug.mic_fbank.rand_aug.p > 0 and random.random() < self.aug.mic_fbank.rand_aug.p:
                noise = (
                    np.random.uniform(
                        self.aug.mic_fbank.rand_aug.range[0], self.aug.mic_fbank.rand_aug.range[1], size=mic_fbank.shape
                    ).astype(np.float32)
                    * np.random.rand()
                )
                mic_fbank = mic_fbank + noise * self.aug.mic_fbank.scale
            return mic_fbank

        sample["mic_fbank"] = self.get_data(seg_idx, sub_idx, set_data)

    def get_imu_acc(self, sample, seg_idx, sub_idx):
        length = self.imu_acc_length
        stride = self.imu_acc_stride

        def set_data(seq, dev, seg, sub_idx):
            imu_acc = self.imu_acc_data[seq][dev][seg][sub_idx : sub_idx + length * stride : stride]
            if self.aug.imu_acc.rand_aug.p > 0 and random.random() < self.aug.imu_acc.rand_aug.p:
                noise = (
                    np.random.uniform(
                        self.aug.imu_acc.rand_aug.range[0], self.aug.imu_acc.rand_aug.range[1], size=imu_acc.shape
                    ).astype(np.float32)
                    * np.random.rand()
                )
                imu_acc = imu_acc + noise
            return imu_acc

        sample["imu_acc"] = self.get_data(seg_idx, sub_idx, set_data)

    def get_pressure(self, sample, seg_idx, sub_idx):
        length = self.pressure_length
        stride = self.pressure_stride
        assert stride == 1

        def set_data(seq, dev, seg, sub_idx):
            pressure = self.pressure_data[seq][dev][seg][sub_idx : sub_idx + length * stride : stride]
            if self.aug.pressure.rand_aug.p > 0 and random.random() < self.aug.pressure.rand_aug.p:
                noise = (
                    np.random.uniform(
                        self.aug.pressure.rand_aug.range[0], self.aug.pressure.rand_aug.range[1], size=pressure.shape
                    ).astype(np.float32)
                    * np.random.rand()
                )
                pressure = pressure + noise * self.aug.pressure.scale
            filtered_pressure: np.array = self.d360.device_cfg.pressure.filter(pressure)[:, None]
            return filtered_pressure.astype(np.float32).copy()

        sample["pressure"] = self.get_data(seg_idx, sub_idx, set_data)

    def get_force(self, sample, seg_idx, sub_idx):
        length = self.force_length
        stride = self.force_stride
        max_force = np.array(self.d360.device_cfg.force.max_force)

        def set_data(seq, dev, seg, sub_idx):
            return self.force_data[seq][dev][seg][sub_idx : sub_idx + length * stride : stride] / max_force

        force = self.get_data(seg_idx, sub_idx, set_data)
        sample["force"] = force[..., -1, :]
        sample["delta_force"] = force[..., -1, :] - force[..., 0, :]
        sample["force_scale"] = max_force

    def get_angle(self, sample, seg_idx, sub_idx):
        assert self.synced is False
        length = self.angle_length
        stride = self.angle_stride

        def set_data(seq, dev, seg, sub_idx):
            return self.angle_data[seq][dev][seg][sub_idx : sub_idx + length * stride : stride]

        angle = self.get_data(seg_idx, sub_idx, set_data)
        sample["angle"] = angle

    def get_height(self, sample, seg_idx, sub_idx):
        assert self.synced is False
        length = self.height_length
        stride = self.height_stride

        def set_data(seq, dev, seg, sub_idx):
            return self.height_data[seq][dev][seg][sub_idx : sub_idx + length * stride : stride]

        height = self.get_data(seg_idx, sub_idx, set_data)
        sample["height"] = height

    def get_label(self, sample: Dict[str, Any], label: str, seq: int):
        label_cls = getattr(self, f"{label}_cls")
        sample[label] = label_cls.get(getattr(self, f"{label}_label")[seq], -100)

    def __getitem__(self, index):
        seg_idx, dev_idx, sys_idx = self.data_idxs[index]
        sample = {}
        for msg, sub_idx in zip(self.device_msgs, dev_idx):
            getattr(self, f"get_{msg}")(sample, seg_idx, sub_idx)

        for msg, sub_idx in zip(self.system_msgs, sys_idx):
            getattr(self, f"get_{msg}")(sample, seg_idx, sub_idx)

        for label in self.label_msgs:
            self.get_label(sample, label, seg_idx[0])

        return sample

    def get_dev_msg(self, msg: str, index: int):
        msg_id = self.device_msgs.index(msg)
        seg_idx, dev_idx, _ = self.data_idxs[index]
        sample = {}
        getattr(self, f"get_{msg}")(sample, seg_idx, dev_idx[msg_id])
        return sample[msg]

    def get_sys_msg(self, msg: str, index: int):
        msg_id = self.system_msgs.index(msg)
        seg_idx, _, sys_idx = self.data_idxs[index]
        sample = {}
        getattr(self, f"get_{msg}")(sample, seg_idx, sys_idx[msg_id])
        return sample[msg]

    def __len__(self):
        return len(self.data_idxs)

    def update_label_cls(self, label: str, label_cls: List[str], sequence_cls: List[str] = []):
        if label not in self.label_msgs:
            self.label_msgs.append(label)
            assert len(sequence_cls) == len(self.sequences), "Inconsistent sequence label size."
            setattr(self, f"{label}_label", sequence_cls)
        setattr(self, f"{label}_cls", {label: n for n, label in enumerate(label_cls)})
