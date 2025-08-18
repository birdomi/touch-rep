from pyvrs.reader import SyncVRSReader, VRSRecord
from enum import IntEnum
import msgpack
import numpy as np
from typing import Dict, List
import json


class VrsStreamIds(IntEnum):
    IMU = 241
    REALSENSE = 2001
    BOWIE = 2002
    PPS = 2003
    HAND_TRACKING = 7001
    MOCAP = 7003
    IMAGE = 8002
    IMAGE_10BIT = 8010


def extract_msgpack_data(record: VRSRecord, field_name: str = "data_msgpack"):
    return msgpack.unpackb(bytes(record.metadata_blocks[0][field_name]))


def extract_hand_mocap_data(reader: SyncVRSReader, right_handed: bool = True):
    stream_id = reader.find_streams(VrsStreamIds.HAND_TRACKING)[0]
    timestamps = []
    records = []

    index = 0
    while index < reader.n_records:
        record = reader.read_next_record(stream_id=stream_id, record_type="data", index=index)
        if record is not None:
            index = record.record_index + 1
            timestamps.append(record.timestamp)
            records.append(extract_msgpack_data(record)["Hands"][right_handed])
        else:
            break
    return np.array(timestamps), records


def extract_hand_json_data(file: str):
    with open(file) as f:
        hand_raw_data = json.load(f)
    return hand_raw_data


def extract_bowie_data(reader: SyncVRSReader, flavor="gum/bowie", num_fingers=5, num_links=4):
    stream_id = reader.get_stream_for_flavor(VrsStreamIds.BOWIE, flavor=flavor)
    timestamps = []
    records = []

    index = 0
    while index < reader.n_records:
        record = reader.read_next_record(stream_id=stream_id, record_type="data", index=index)
        if record is not None:
            index = record.record_index + 1
            timestamps.append(record.timestamp)
            records.append(record.metadata_blocks[0])
        else:
            break

    mag_data = [[[] for _ in range(num_links)] for _ in range(num_fingers)]
    mag_sys_time = [[[] for _ in range(num_links)] for _ in range(num_fingers)]
    mag_dev_time = [[[] for _ in range(num_links)] for _ in range(num_fingers)]
    imu_data = [[[] for _ in range(num_links)] for _ in range(num_fingers)]
    imu_sys_time = [[[] for _ in range(num_links)] for _ in range(num_fingers)]
    imu_dev_time = [[[] for _ in range(num_links)] for _ in range(num_fingers)]

    for raw_time, raw_data in zip(timestamps, records):
        if raw_data["mag_secs"] != -1:
            finger = raw_data["finger"] - 1
            link = raw_data["link"] - 1
            dev_time = get_bowie_mag_time(raw_data)

            mag_data[finger][link].append(raw_data)
            mag_sys_time[finger][link].append(raw_time - dev_time)
            mag_dev_time[finger][link].append(dev_time)

        if raw_data["quat_secs"] != -1:
            finger = raw_data["finger"] - 1
            link = raw_data["link"] - 1
            dev_time = get_bowie_imu_time(raw_data)

            imu_data[finger][link].append(raw_data)
            imu_sys_time[finger][link].append(raw_time - dev_time)
            imu_dev_time[finger][link].append(dev_time)

    mag_time = [
        [min(mag_sys_time[finger][link]) + np.array(mag_dev_time[finger][link]) for link in range(num_links)]
        for finger in range(num_fingers)
    ]

    imu_time = [
        [min(imu_sys_time[finger][link]) + np.array(imu_dev_time[finger][link]) for link in range(num_links)]
        for finger in range(num_fingers)
    ]

    return {
        "mag_time": mag_time,
        "mag_data": mag_data,
        "imu_time": imu_time,
        "imu_data": imu_data,
    }


def get_bowie_mag_time(bowie_data):
    return bowie_data["mag_secs"] + bowie_data["mag_nanosecs"] / 1e9


def get_bowie_imu_time(bowie_data):
    return bowie_data["quat_secs"] + bowie_data["quat_nanosecs"] / 1e9


def extract_pps_data(reader: SyncVRSReader, flavor="gum/pps"):
    stream_id = reader.get_stream_for_flavor(VrsStreamIds.PPS, flavor=flavor)
    timestamps = []
    records = []

    index = 0
    while index < reader.n_records:
        record = reader.read_next_record(stream_id=stream_id, record_type="data", index=index)
        if record is not None:
            index = record.record_index + 1
            timestamps.append(record.timestamp)
            records.append(record.metadata_blocks[0])
        else:
            break
    return np.array(timestamps), records


def get_field_data(data_dicts: List[Dict], field: str):
    return [data_dict[field] for data_dict in data_dicts]
