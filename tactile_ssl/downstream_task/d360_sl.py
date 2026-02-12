# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Supervised Learning module for downstream tasks using pre-trained D360 touch embeddings.

This module provides a framework for applying pre-trained tactile embeddings from Sparsh-X
to various supervised learning tasks.
"""


from typing import Any, Dict, Optional, Literal, List, Tuple
from functools import partial
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.sl_module import SLModule
from tactile_ssl.model.d360_transformer import D360Transformer

import einops


log = get_pylogger(__name__)


class D360SLModule(SLModule):
    def __init__(
        self,
        model_encoder: D360Transformer,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        sensors: Optional[List[str]] = None,
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: Literal["dino", "e2e"] = "dino",
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
        )
        self.sensors = sensors if sensors is not None else model_encoder.sensors

        for sensor in self.sensors:
            assert sensor in model_encoder.sensors, f"{sensor} is not available in model encoder."

        if "dino" in encoder_type or "e2e" in encoder_type:
            self.prepare_data = self.prepare_data_multimodal
        else:
            raise NotImplementedError

    def prepare_data_multimodal(self, batch: Dict[str, Any]) -> Tuple[Any, Any]:
        info = {"mic_fbank": "mic", "imu_acc": "imu"}
        xs_orig = {info.get(key, key): val for key, val in batch.items()}
        xs = {}

        for sensor in self.model_encoder.sensors:
            if sensor == "img":
                xs_orig["img"] = einops.rearrange(xs_orig["img"], "b n c h w -> b c n h w")
                xs["img"] = einops.rearrange(xs_orig["img"], "b c n h w -> b (c n) h w")
            elif sensor == "mic" or sensor == "imu" or sensor == "pressure":
                xs[sensor] = xs_orig[sensor]
            else:
                raise NotImplementedError

        return xs, xs_orig

    def encode(self, x: torch.Tensor):
        embeddings = self.model_encoder(x)
        embeddings = {sensor: embeddings[sensor] for sensor in self.sensors}
        embeddings = [F.layer_norm(embedding, (embedding.size(-1),)) for _, embedding in embeddings.items()]
        return embeddings

    def forward(self, x: torch.Tensor):
        embeddings = self.encode(x)
        z = torch.cat(embeddings, dim=1)

        if self.train_encoder:
            y_pred = self.model_task(z)
        else:
            y_pred = self.model_task(z.detach())
        return y_pred

    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        raise NotImplementedError

    @abstractmethod
    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        raise NotImplementedError

    @abstractmethod
    def on_validation_epoch_end(self, trainer_instance=None):
        raise NotImplementedError
