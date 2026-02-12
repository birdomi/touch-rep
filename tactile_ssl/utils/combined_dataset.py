import torch.utils.data as data
import numpy as np
from typing import Optional


class CombinedDataset(data.Dataset):
    def __init__(
        self,
        main_dataset: data.Dataset,
        supp_dataset: Optional[data.Dataset],
    ):
        self._main_dataset = main_dataset
        self._supp_dataset = supp_dataset

    def __getitem__(self, index):
        samples = {"main": self.main_dataset[index]}
        if self.supp_dataset is not None and len(self.supp_dataset) > 0:
            ratio = len(self.supp_dataset) / len(self.main_dataset)
            supp_index = min(int((index + np.random.rand()) * ratio), len(self._supp_dataset) - 1)
            samples["supp"] = self.supp_dataset[supp_index]
            samples["supp_index"] = supp_index
        return samples

    def __len__(self):
        return self.main_dataset.__len__()

    @property
    def main_dataset(self):
        return self._main_dataset

    @property
    def supp_dataset(self):
        return self._supp_dataset
