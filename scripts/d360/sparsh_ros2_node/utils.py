import bisect
from typing import Optional, Tuple, Union

import numpy as np
from numpy._typing import _ShapeLike
from numpy.typing import DTypeLike


class CircularBuffer:
    def __init__(self, capacity: _ShapeLike, dtype: DTypeLike = np.float64):
        if isinstance(capacity, int):
            self.capacity = capacity
        elif isinstance(capacity, Tuple):
            self.capacity = capacity[0]
        else:
            raise ValueError("Capacity must be an integer or a tuple")
        self.buffer = np.empty(capacity, dtype=dtype)
        self.head = 0
        self.tail = 0
        self.size = 0

    def __call__(self, idx: Optional[slice] = None) -> np.ndarray:
        if idx is None:
            return self.buffer[self.head : self.tail]
        wrapped_idx = slice((self.head + idx.start) % self.capacity, (self.head + idx.stop) % self.capacity, idx.step)
        return self.buffer[wrapped_idx]

    def __getitem__(self, idx: int) -> np.ndarray:
        wrapped_idx = (self.head + idx) % self.capacity
        return self.buffer[wrapped_idx]

    def is_full(self):
        return self.size >= self.capacity

    def is_empty(self):
        return self.size == 0

    def push(self, item):
        if not self.is_full():
            self.size += 1
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity

    def pop(self):
        if self.is_empty():
            return None
        item = self.buffer[self.head].copy()
        self.buffer[self.head] = np.nan  # Mark as empty
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item

    def pop_multiple(self, n: int):
        items = self.buffer[self.head : self.head + n].copy()
        self.buffer[self.head : self.head + n] = np.nan  # Mark as empty
        self.head = (self.head + n) % self.capacity
        self.size -= n
        return items

    def peek_newest(self):
        if self.is_empty():
            return None
        newest_index = (self.tail - 1) % self.capacity
        return self.buffer[newest_index]

    def peek(self):
        if self.is_empty():
            return None
        return self.buffer[self.head]


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
