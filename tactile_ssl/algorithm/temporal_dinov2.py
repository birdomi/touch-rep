"""DINOv2 over temporal crops, for the frame-encoder + temporal-former stack.

The image-crop vocabulary maps onto the time axis:

===================  ==================================================
DINOv2 (images)      here
===================  ==================================================
2 global crops       2 long contiguous frame ranges
8 local crops        8 short contiguous frame ranges
iBOT patch masking   per-token contiguous frame ranges replaced by the
                     former's mask token
DINO / iBOT / KoLeo  reused unchanged from ``DINOv2Module``
===================  ==================================================

Both halves of :class:`FrameTemporalEncoder` train from scratch together:
the composite is the DINOv2 ``backbone``, so ``configure_optimizers`` collects
every parameter and ``update_moving_average`` tracks the whole thing.

Self-contained: subclasses ``DINOv2Module`` and overrides ``sample_masks`` /
``forward`` / ``training_step`` without touching any existing file.
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch

from tactile_ssl.loss.dino_loss import DINOLoss
from tactile_ssl.loss.ibot_patch_loss import iBOTPatchLoss

from .dinov2 import DINOv2Module


class TemporalDINOv2Module(DINOv2Module):
    """DINOv2 whose views are contiguous time ranges rather than sensor subsets."""

    def __init__(
        self,
        global_crop_frames: Tuple[int, int] = (9, 15),
        local_crop_frames: Tuple[int, int] = (4, 7),
        ibot_span_frames: Tuple[int, int] = (2, 5),
        sensor_input_key: str = "joint_contact",
        pos_input_key: str = "finger_xyz",
        frame_dino_loss_weight: float = 0.0,
        frame_ibot_loss_weight: float = 0.0,
        frame_dino_frames: int = 2,
        frame_dino_num_local_masks: int = 4,
        frame_dino_global_mask_scale: Tuple[float, float] = (0.6, 1.0),
        frame_dino_local_mask_scale: Tuple[float, float] = (0.3, 0.6),
        frame_dino_ibot_mask_ratio: Tuple[float, float] = (0.3, 0.7),
        **kwargs,
    ) -> None:
        # The parent consumes the dino_head partial; keep it for the
        # frame-level head built below.
        dino_head_partial = kwargs.get("dino_head")
        super().__init__(**kwargs)
        if self.num_global_masks != 2:
            raise ValueError(
                "TemporalDINOv2Module follows the DINOv2 two-global-view "
                f"objective; got num_global_masks={self.num_global_masks}"
            )
        self.global_crop_frames = tuple(int(v) for v in global_crop_frames)
        self.local_crop_frames = tuple(int(v) for v in local_crop_frames)
        self.ibot_span_frames = tuple(int(v) for v in ibot_span_frames)
        self.sensor_input_key = sensor_input_key
        self.pos_input_key = pos_input_key

        backbone = self.student_encoder_dict["backbone"]
        self.tokens_per_frame = int(backbone.tokens_per_frame)

        # ---- frame-level DINOv2 (applied to the frame encoder's outputs) ----
        # Views are joint subsets of single frames, exactly like the
        # frame-wise angle DINOv2, so the frame encoder's CLS and joint patch
        # tokens receive direct supervision even when the former only consumes
        # the CLS (frame_token_set="cls").
        self.frame_dino_loss_weight = float(frame_dino_loss_weight)
        self.frame_ibot_loss_weight = float(frame_ibot_loss_weight)
        self.frame_dino_enabled = (
            self.frame_dino_loss_weight > 0 or self.frame_ibot_loss_weight > 0
        )
        if self.frame_dino_enabled:
            self.frame_dino_frames = int(frame_dino_frames)
            self.frame_dino_num_local_masks = int(frame_dino_num_local_masks)
            self.frame_dino_global_mask_scale = tuple(frame_dino_global_mask_scale)
            self.frame_dino_local_mask_scale = tuple(frame_dino_local_mask_scale)
            self.frame_dino_ibot_mask_ratio = tuple(frame_dino_ibot_mask_ratio)

            head_factory = partial(dino_head_partial, in_dim=backbone.embed_dim)
            student_head, teacher_head = head_factory(), head_factory()
            teacher_head.requires_grad_(False)
            # Registering the pair under the same key in both ModuleDicts puts
            # the head on the optimizer (student) and the EMA update (teacher).
            self.student_encoder_dict["frame_dino_head"] = student_head
            self.student_encoder["frame_dino_head"] = student_head
            self.teacher_encoder_dict["frame_dino_head"] = teacher_head
            self.teacher_encoder["frame_dino_head"] = teacher_head

            out_dim = student_head.last_layer.out_features
            self.frame_dino_loss = DINOLoss(out_dim=out_dim)
            self.frame_ibot_loss = iBOTPatchLoss(patch_out_dim=out_dim)

    # ---------------------------------------------------------------- crops

    def _sample_crop_indices(
        self,
        num_crops: int,
        batch_size: int,
        window: int,
        length_range: Tuple[int, int],
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """``(num_crops, B, L)`` contiguous absolute frame indices.

        One length is drawn per crop (shared across the batch) so every crop
        in a group is a rectangular tensor; the start offset is per sample.
        """
        low, high = length_range
        high = min(high, window)
        low = max(1, min(low, high))
        length = int(
            torch.randint(
                low, high + 1, (1,), generator=generator, device=device
            ).item()
        )
        starts = torch.randint(
            0, window - length + 1, (num_crops, batch_size, 1),
            generator=generator, device=device,
        )
        offsets = torch.arange(length, device=device).view(1, 1, length)
        return starts + offsets

    def _sample_ibot_masks(
        self,
        crop_indices: torch.Tensor,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """``(num_crops, B, L, K)`` bool, contiguous in time per token.

        Each token position independently gets one masked span, so the task is
        "recover this sensor's behaviour over these frames from its neighbours
        and from the surrounding time", not "recover a whole missing frame".
        """
        num_crops, batch_size, length = crop_indices.shape
        tokens = self.tokens_per_frame

        low, high = self.ibot_span_frames
        high = min(high, length)
        low = max(0, min(low, high))
        ratio_low, ratio_high = self.ibot_mask_ratio
        # Fraction of token positions that get a masked span at all.
        keep = torch.rand(
            (num_crops, batch_size, tokens), generator=generator, device=device
        )
        threshold = ratio_low + torch.rand(
            (1,), generator=generator, device=device
        ) * (ratio_high - ratio_low)
        active = keep < threshold

        span = torch.randint(
            max(low, 1), high + 1, (num_crops, batch_size, tokens),
            generator=generator, device=device,
        )
        max_start = (length - span).clamp_min(0)
        start = (
            torch.rand(
                (num_crops, batch_size, tokens), generator=generator, device=device
            ) * (max_start + 1).float()
        ).floor().long()

        frames = torch.arange(length, device=device).view(1, 1, length, 1)
        start = start.unsqueeze(2)
        span = span.unsqueeze(2)
        masked = (frames >= start) & (frames < start + span)
        return masked & active.unsqueeze(2)

    def sample_masks(self, xs: torch.Tensor):
        """Temporal analogue of the sensor-mask sampler."""
        device = xs.device
        batch_size, window = xs.shape[0], xs.shape[1]
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.step))

        global_crops = self._sample_crop_indices(
            self.num_global_masks, batch_size, window,
            self.global_crop_frames, device, generator,
        )
        local_crops = self._sample_crop_indices(
            self.num_local_masks, batch_size, window,
            self.local_crop_frames, device, generator,
        )
        ibot_masks = self._sample_ibot_masks(global_crops, device, generator)
        return global_crops, local_crops, ibot_masks

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _gather_crops(tokens: torch.Tensor, crop_indices: torch.Tensor):
        """Select crop frames from pre-encoded tokens.

        Args:
            tokens: ``(B, W, K, D)`` encoded once for the whole window.
            crop_indices: ``(num_crops, B, L)``.

        Returns:
            ``(num_crops * B, L, K, D)`` tokens and ``(num_crops * B, L)``
            absolute frame indices, stacked the same way the DINOv2 code
            stacks views along the batch dimension.
        """
        num_crops, batch_size, length = crop_indices.shape
        _, _, tokens_per_frame, dim = tokens.shape
        index = crop_indices.reshape(num_crops * batch_size, length, 1, 1)
        index = index.expand(-1, -1, tokens_per_frame, dim)
        repeated = tokens.unsqueeze(0).expand(num_crops, -1, -1, -1, -1).flatten(0, 1)
        return torch.gather(repeated, 1, index), crop_indices.flatten(0, 1)

    # -------------------------------------------------------------- forward

    def forward(
        self,
        xs: torch.Tensor,
        pos: torch.Tensor,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        ibot_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        student = self.student_encoder_dict["backbone"]
        teacher = self.teacher_encoder_dict["backbone"]

        # The frame encoder is frame-independent, so encoding the window once
        # per network and gathering crops is identical to encoding each crop
        # separately, at a fraction of the cost.
        student_frames = student.encode_frames(xs, pos)
        student_global, global_index = self._gather_crops(student_frames, global_crops)
        student_local, local_index = self._gather_crops(student_frames, local_crops)

        masks_flat = ibot_masks.flatten(0, 1)
        student_global_dict = student.forward_features(
            student_global, global_index, masktoken_masks=masks_flat
        )
        student_local_dict = student.forward_features(student_local, local_index)

        student_global_cls = student_global_dict["x_norm_regtokens"][:, 0]
        student_local_cls = student_local_dict["x_norm_regtokens"][:, 0]
        student_patches = student_global_dict["x_norm_patchtokens"]

        # Flatten before nonzero: on a >1-D mask nonzero returns coordinate
        # tuples, which must not be used as flat indices.
        selected = masks_flat.flatten(1).flatten()
        mask_indices = torch.nonzero(selected).flatten()
        num_ibot_tokens = int(mask_indices.numel())

        student_masked = student_patches.flatten(0, 1)[mask_indices]

        with torch.no_grad():
            teacher_frames = teacher.encode_frames(xs, pos)
            teacher_global, _ = self._gather_crops(teacher_frames, global_crops)
            teacher_dict = teacher.forward_features(teacher_global, global_index)

            # Kept in view order, NOT reverse-cat'd as upstream DINOv2 does:
            # this repo's DINOLoss skips the i == j pair, so the cross-view
            # global terms are exactly the off-diagonal ones. Reversing here
            # too would flip them back into student view k vs teacher view k.
            teacher_cls = teacher_dict["x_norm_regtokens"][:, 0]

            teacher_masked = teacher_dict["x_norm_patchtokens"].flatten(0, 1)[mask_indices]

            teacher_cls_after_head = self.teacher_encoder_dict["dino_head"](teacher_cls)
            teacher_patch_after_head = self.teacher_encoder_dict["dino_head"](teacher_masked)

            if self.centering == "centering":
                teacher_dino_centered = self.dino_loss.softmax_center_teacher(
                    teacher_cls_after_head, teacher_temp=self.current_teacher_temp
                ).view(self.num_global_masks, -1, *teacher_cls_after_head.shape[1:])
                teacher_ibot_centered = self.ibot_patch_loss.softmax_center_teacher(
                    teacher_patch_after_head.unsqueeze(0),
                    teacher_temp=self.current_teacher_temp,
                ).squeeze(0)
                self.dino_loss.update_center(teacher_cls_after_head)
                self.ibot_patch_loss.update_center(teacher_patch_after_head)
            elif self.centering == "sinkhorn_knopp":
                teacher_dino_centered = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_after_head, teacher_temp=self.current_teacher_temp
                ).view(self.num_global_masks, -1, *teacher_cls_after_head.shape[1:])
                teacher_ibot_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    teacher_patch_after_head,
                    teacher_temp=self.current_teacher_temp,
                    n_masked_patches_tensor=torch.tensor(
                        num_ibot_tokens, dtype=int, device=xs.device
                    ),
                )
            else:
                raise NotImplementedError(f"Unknown centering: {self.centering}")

        head = self.student_encoder_dict["dino_head"]
        student_cls_after_head = torch.cat([
            head(student_global_cls), head(student_local_cls)
        ], dim=0)
        student_patch_after_head = head(student_masked)

        n_local_terms = max(self.num_local_masks * self.num_global_masks, 1)
        n_global_terms = (self.num_global_masks - 1) * self.num_global_masks

        dino_loss = self.dino_loss(
            student_cls_after_head.chunk(self.num_global_masks + self.num_local_masks),
            teacher_dino_centered,
        ) / (n_local_terms + n_global_terms)

        koleo_loss = self.koleo_weight * sum(
            self.koleo_loss(p) for p in student_global_cls.chunk(self.num_global_masks)
        )

        patch_loss = (1.0 / self.num_global_masks) * self.ibot_patch_loss(
            student_patch_after_head, teacher_ibot_centered
        )

        ssl_loss = dino_loss + patch_loss + koleo_loss
        return {
            "ssl_loss": ssl_loss,
            "loss": ssl_loss,
            "dino_loss": dino_loss.detach(),
            "patch_loss": patch_loss.detach(),
            "koleo_loss": koleo_loss.detach(),
            "ibot_tokens": torch.tensor(float(num_ibot_tokens)),
        }

    # ----------------------------------------------- frame-level DINOv2

    def _frame_level_losses(
        self, xs: torch.Tensor, pos: torch.Tensor, generator: torch.Generator
    ):
        """DINO + iBOT over subset views of sampled single frames.

        Both streams are cropped and both receive iBOT mask tokens, matching
        ``AngleDinov2Module.sample_masks``: keeping every pose token in a local
        view would let the CLS match views from the (uncropped) hand
        configuration alone, which is a shortcut when the sensor stream is
        proximity-only, and would leave the pose tokens without any iBOT
        target.

        Student global views additionally get mask tokens, so the same pass
        provides both the DINO CLS terms and the iBOT patch targets, as in
        image DINOv2.
        """
        backbone_s = self.student_encoder_dict["backbone"]
        backbone_t = self.teacher_encoder_dict["backbone"]
        fe_s, fe_t = backbone_s.frame_encoder, backbone_t.frame_encoder
        device = xs.device

        batch, window, num_joints, chans = xs.shape
        num_pos = pos.shape[2]
        num_frames = min(self.frame_dino_frames, window)
        frame_idx = torch.randperm(window, generator=generator, device=device)[:num_frames]
        xs_f = xs.index_select(1, frame_idx).reshape(batch * num_frames, 1, num_joints, chans)
        pos_f = pos.index_select(1, frame_idx).reshape(batch * num_frames, 1, num_pos, pos.shape[-1])
        eff = batch * num_frames

        def keep_size(scale, total):
            low, high = scale
            frac = low + (high - low) * float(
                torch.rand((), generator=generator, device=device)
            )
            return max(1, int(round(frac * total)))

        def sample_views(num_views, keep, total):
            scores = torch.rand(num_views, eff, total, generator=generator, device=device)
            return scores.topk(keep, dim=-1).indices.sort(dim=-1).values

        num_local = self.frame_dino_num_local_masks
        global_scale = self.frame_dino_global_mask_scale
        local_scale = self.frame_dino_local_mask_scale
        # A joint-only frame encoder has no sensor tokens to crop or mask, so
        # the pose stream carries the whole objective.
        pos_only = bool(getattr(fe_s, "pos_only", False))
        # Sensor and pose streams are cropped independently -- they have
        # different token counts (42 joints vs 10 fingers), so a shared keep
        # count would not even be expressible.
        if pos_only:
            sensor_global = sensor_local = None
        else:
            sensor_global = sample_views(2, keep_size(global_scale, num_joints), num_joints)
            sensor_local = sample_views(num_local, keep_size(local_scale, num_joints), num_joints)
        pos_global = sample_views(2, keep_size(global_scale, num_pos), num_pos)
        pos_local = sample_views(num_local, keep_size(local_scale, num_pos), num_pos)

        ratio_low, ratio_high = self.frame_dino_ibot_mask_ratio
        ratio = ratio_low + (ratio_high - ratio_low) * float(
            torch.rand((), generator=generator, device=device)
        )
        sensor_ibot = (
            None
            if pos_only
            else torch.rand(
                2, eff, sensor_global.shape[-1], generator=generator, device=device
            ) < ratio
        )
        pos_ibot = (
            torch.rand(2, eff, pos_global.shape[-1], generator=generator, device=device)
            < ratio
        )
        # patchtokens are laid out [pos tokens, sensor tokens], so the iBOT
        # selector over the full patch axis concatenates in that order.
        full_ibot = (
            pos_ibot if pos_only else torch.cat([pos_ibot, sensor_ibot], dim=-1)
        )

        student_global = fe_s.forward_features(
            xs_f, pos_f, masks=sensor_global, mask_type="tubelet",
            masktoken_masks=sensor_ibot, pos_masks=pos_global,
            pos_masktoken_masks=pos_ibot,
        )
        student_local = fe_s.forward_features(
            xs_f, pos_f, masks=sensor_local, mask_type="tubelet",
            pos_masks=pos_local,
        )
        with torch.no_grad():
            teacher_global = fe_t.forward_features(
                xs_f, pos_f, masks=sensor_global, mask_type="tubelet",
                pos_masks=pos_global,
            )

        head_s = self.student_encoder_dict["frame_dino_head"]
        head_t = self.teacher_encoder_dict["frame_dino_head"]

        student_cls_after_head = head_s(torch.cat([
            student_global["x_norm_regtokens"][:, 0],
            student_local["x_norm_regtokens"][:, 0],
        ], dim=0))

        with torch.no_grad():
            # View order preserved -- DINOLoss skips i == j, so the off-diagonal
            # pairs are already the cross-view ones (see the window-level note).
            teacher_cls = teacher_global["x_norm_regtokens"][:, 0]
            teacher_cls_after_head = head_t(teacher_cls)
            teacher_centered = self.frame_dino_loss.softmax_center_teacher(
                teacher_cls_after_head, teacher_temp=self.current_teacher_temp
            ).view(2, -1, teacher_cls_after_head.shape[-1])
            self.frame_dino_loss.update_center(teacher_cls_after_head)

        n_terms = 2 * 1 + num_local * 2
        dino = self.frame_dino_loss(
            student_cls_after_head.chunk(2 + num_local), teacher_centered
        ) / n_terms

        # Flatten before nonzero: coordinate tuples must not be flat indices.
        selected = torch.nonzero(full_ibot.flatten(0, 1).flatten()).flatten()
        frame_ibot = xs.new_zeros(())
        if selected.numel():
            # Both streams carry targets, so the whole patch axis is indexed.
            student_patches = student_global["x_norm_patchtokens"]
            student_masked = student_patches.flatten(0, 1)[selected]
            with torch.no_grad():
                teacher_patches = teacher_global["x_norm_patchtokens"]
                teacher_masked_after_head = head_t(teacher_patches.flatten(0, 1)[selected])
                teacher_ibot_centered = self.frame_ibot_loss.softmax_center_teacher(
                    teacher_masked_after_head.unsqueeze(0),
                    teacher_temp=self.current_teacher_temp,
                ).squeeze(0)
                self.frame_ibot_loss.update_center(teacher_masked_after_head)
            frame_ibot = 0.5 * self.frame_ibot_loss(
                head_s(student_masked), teacher_ibot_centered
            )

        return dino, frame_ibot, int(selected.numel())

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step += 1
        xs = batch[self.sensor_input_key]
        pos = batch[self.pos_input_key]

        global_crops, local_crops, ibot_masks = self.sample_masks(xs)
        fwd = self.forward(xs, pos, global_crops, local_crops, ibot_masks)

        loss = fwd["loss"]
        output = {
            "ssl_loss": fwd["ssl_loss"].item(),
            "dino_loss": fwd["dino_loss"].item(),
            "patch_loss": fwd["patch_loss"].item(),
            "koleo_loss": float(fwd["koleo_loss"]),
            "ibot_tokens": float(fwd["ibot_tokens"]),
            "online_probes_loss": 0.0,
        }

        if self.frame_dino_enabled:
            generator = torch.Generator(device=xs.device)
            generator.manual_seed(int(self.step) + 7919)
            frame_dino, frame_ibot, frame_ibot_tokens = self._frame_level_losses(
                xs, pos, generator
            )
            loss = (
                loss
                + self.frame_dino_loss_weight * frame_dino
                + self.frame_ibot_loss_weight * frame_ibot
            )
            output["frame_dino_loss"] = frame_dino.detach().item()
            output["frame_ibot_loss"] = frame_ibot.detach().item()
            output["frame_ibot_tokens"] = float(frame_ibot_tokens)

        output["loss"] = loss
        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_on_batch_end(self, outputs, stage: str = "train", trainer_instance=None):
        super().log_on_batch_end(outputs, stage=stage, trainer_instance=trainer_instance)
        if trainer_instance is None or (stage == "train" and not trainer_instance.should_log):
            return
        for key in ("dino_loss", "patch_loss", "koleo_loss", "ibot_tokens",
                    "frame_dino_loss", "frame_ibot_loss", "frame_ibot_tokens"):
            if key in outputs:
                trainer_instance.wandb.log({
                    f"{stage}/{key}": outputs[key],
                    f"global_{stage}_step": trainer_instance.step,
                })
