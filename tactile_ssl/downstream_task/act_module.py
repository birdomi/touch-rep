"""Training module for the ACT policy.

Loss follows the reference implementation: masked L1 on the predicted action
chunk plus a KL term pulling the CVAE posterior towards the prior.

Validation reports two numbers:

``val/l1``
    teacher-forced L1 — the CVAE encoder sees the ground-truth chunk, same as
    training, so it is directly comparable to ``train/l1``.
``val/l1_infer``
    L1 with the latent clamped to the prior mean, i.e. exactly what the policy
    does at deployment time. This is the number to watch.
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.algorithm.module import Module
from tactile_ssl.model.act import ACTPolicy, kl_divergence
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class ACTModule(Module, nn.Module):
    """Fabric-Trainer wrapper around :class:`~tactile_ssl.model.act.ACTPolicy`."""

    def __init__(
        self,
        model: ACTPolicy,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial] = None,
        kl_weight: float = 10.0,
        lr_backbone: Optional[float] = 1e-5,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.kl_weight = float(kl_weight)
        self.lr_backbone = lr_backbone
        self.optim_partial = optim_cfg
        self.scheduler_partial = scheduler_cfg

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        self.val_outputs = []
        self.last_val_metrics: Dict[str, float] = {}
        self.best_val_metrics: Dict[str, float] = {}
        self._best_state_dict = None

        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.parameters())
        log.info(f"ACTModule: {num_trainable:,} / {num_total:,} trainable params")

    def load_checkpoint(self, checkpoint: str):
        log.info(f"Loading ACT weights from {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        state = {key.replace("model.", "", 1) if key.startswith("model.") else key: value
                 for key, value in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        log.info(f"  missing={len(missing)} unexpected={len(unexpected)}")

    # ── loss ────────────────────────────────────────────────────────────────

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        actions = batch["actions"]
        is_pad = batch["is_pad"]
        tactile = batch.get("tactile") if self.model.use_tactile else None

        out = self.model(
            qpos=batch["qpos"],
            images=batch["images"],
            tactile=tactile,
            actions=actions,
            is_pad=is_pad,
        )

        l1 = self._masked_l1(out["a_hat"], actions, is_pad)
        kl = kl_divergence(out["mu"], out["logvar"])

        return {
            "loss": l1 + self.kl_weight * kl,
            "l1": l1.detach(),
            "kl": kl.detach(),
        }

    @staticmethod
    def _masked_l1(prediction: torch.Tensor, actions: torch.Tensor, is_pad: torch.Tensor):
        valid = (~is_pad).unsqueeze(-1)
        l1_all = F.l1_loss(prediction, actions, reduction="none")
        return (l1_all * valid).sum() / valid.expand_as(l1_all).sum().clamp(min=1)

    @torch.no_grad()
    def inference_l1(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Masked L1 with the latent at the prior mean (deployment behaviour)."""
        tactile = batch.get("tactile") if self.model.use_tactile else None
        out = self.model(qpos=batch["qpos"], images=batch["images"], tactile=tactile)
        return self._masked_l1(out["a_hat"], batch["actions"], batch["is_pad"])

    @torch.no_grad()
    def hold_state_l1(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Masked L1 of the 'repeat the current state for the whole chunk' policy.

        BrainCo actions sit very close to the measured state, so this trivial
        policy is a strong baseline: a model that does not beat it has learned
        nothing useful. Only defined when the state and action spaces match.
        """
        if self.model.state_dim != self.model.action_dim:
            return None
        qpos_raw = batch["qpos"] * self.model.qpos_std + self.model.qpos_mean
        hold = ((qpos_raw - self.model.action_mean) / self.model.action_std).unsqueeze(1)
        return self._masked_l1(hold.expand_as(batch["actions"]), batch["actions"], batch["is_pad"])

    # ── steps ───────────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, Any]):
        tactile = batch.get("tactile") if self.model.use_tactile else None
        return self.model(qpos=batch["qpos"], images=batch["images"], tactile=tactile)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.compute_loss(batch)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        outputs = self.compute_loss(batch)
        outputs["l1_infer"] = self.inference_l1(batch)
        hold = self.hold_state_l1(batch)
        if hold is not None:
            outputs["l1_hold"] = hold
        return outputs

    # ── optimization ────────────────────────────────────────────────────────

    def configure_optimizers(
        self, num_iterations_per_epoch: int, num_epochs: int, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        backbone_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if "backbones." in name else other_params).append(param)

        param_groups = [{"params": other_params}]
        if backbone_params:
            if self.lr_backbone is None:
                param_groups.append({"params": backbone_params})
            else:
                # ACT fine-tunes the ImageNet trunks an order of magnitude
                # slower than the freshly initialized transformer.
                param_groups.append({"params": backbone_params, "lr": float(self.lr_backbone)})
        log.info(
            f"Optimizer groups: transformer={sum(p.numel() for p in other_params):,} params, "
            f"backbone={sum(p.numel() for p in backbone_params):,} params "
            f"(lr_backbone={self.lr_backbone})"
        )

        optimizer = self.optim_partial(param_groups)
        if self.scheduler_partial is None:
            return optimizer, None, None

        scheduler = self.scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
        )
        return optimizer, {"scheduler": scheduler, "interval": "step", "monitor": None}, None

    # ── logging ─────────────────────────────────────────────────────────────

    @staticmethod
    def _scalar(value) -> float:
        return float(value.item() if torch.is_tensor(value) else value)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is None or not trainer_instance.should_log:
            return
        payload = {f"global_{label}_step": step}
        for key in ("loss", "l1", "kl", "l1_infer", "l1_hold"):
            if key in outputs:
                payload[f"{label}/{key}"] = self._scalar(outputs[key])
        trainer_instance.wandb.log(payload)

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_outputs.append({
            key: self._scalar(outputs[key])
            for key in ("loss", "l1", "kl", "l1_infer", "l1_hold")
            if key in outputs
        })
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_validation_epoch_end(self, trainer_instance=None):
        if not self.val_outputs:
            return

        metrics = {
            key: float(np.mean([out[key] for out in self.val_outputs]))
            for key in self.val_outputs[0]
        }
        self.val_outputs = []
        self.last_val_metrics = metrics

        log.info("=" * 46)
        log.info("Validation Results:")
        log.info(f"  loss     : {metrics.get('loss', float('nan')):.5f}")
        log.info(f"  l1       : {metrics.get('l1', float('nan')):.5f}")
        log.info(f"  kl       : {metrics.get('kl', float('nan')):.5f}")
        log.info(f"  l1_infer : {metrics.get('l1_infer', float('nan')):.5f}")
        if "l1_hold" in metrics:
            hold = metrics["l1_hold"]
            infer = metrics.get("l1_infer", float("nan"))
            verdict = "BEATS baseline" if infer < hold else "worse than baseline"
            log.info(f"  l1_hold  : {hold:.5f}  (hold-state baseline)")
            log.info(f"  --> ACT / baseline = {infer / hold:.3f}  [{verdict}]")
        log.info("=" * 46)

        # l1_infer is the deployment-time error, so model selection tracks it.
        selection_key = "l1_infer" if "l1_infer" in metrics else "loss"
        best = self.best_val_metrics.get(selection_key, float("inf"))
        if metrics[selection_key] < best:
            import copy
            import os

            self.best_val_metrics = dict(metrics)
            self._best_state_dict = copy.deepcopy(
                {key: value.cpu() for key, value in self.state_dict().items()}
            )
            log.info(f"New best val {selection_key}: {metrics[selection_key]:.5f}")
            # Persist it: the Trainer only writes last.ckpt plus periodic
            # snapshots, so without this the best epoch is lost on exit.
            if trainer_instance is not None and getattr(trainer_instance, "checkpoint_dir", None):
                os.makedirs(trainer_instance.checkpoint_dir, exist_ok=True)
                path = os.path.join(trainer_instance.checkpoint_dir, "best.ckpt")
                torch.save(
                    {
                        "model": self._best_state_dict,
                        "metrics": self.best_val_metrics,
                        "epoch": int(trainer_instance.current_epoch),
                    },
                    path,
                )
                log.info(f"  best weights written to {path}")

        if trainer_instance is not None:
            payload = {f"val/{key}_epoch": value for key, value in metrics.items()}
            payload["epoch"] = trainer_instance.current_epoch
            trainer_instance.wandb.log(payload)
