"""Shared tactile-force channel conversion."""

import numpy as np


RAW_FORCE_CHANNELS = (
    "normal_force",
    "tangential_force",
    "tangential_direction",
    "proximity",
)
FORCE_CHANNELS = (
    "normal_force",
    "tangential_force_x",
    "tangential_force_y",
    "proximity",
)


def force_direction_to_cartesian(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert [Fn, Ft, theta_deg, proximity] to Cartesian force channels."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 4:
        raise ValueError(f"Expected 4 force channels, got {values.shape}")

    raw_valid = values >= 0
    tangential_valid = raw_valid[..., 1] & raw_valid[..., 2]
    theta = np.deg2rad(np.where(tangential_valid, values[..., 2], 0.0))
    ft = np.where(tangential_valid, values[..., 1], 0.0)
    converted = np.stack(
        [
            np.where(raw_valid[..., 0], values[..., 0], 0.0),
            ft * np.cos(theta),
            ft * np.sin(theta),
            np.where(raw_valid[..., 3], values[..., 3], 0.0),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    valid = np.stack(
        [
            raw_valid[..., 0],
            tangential_valid,
            tangential_valid,
            raw_valid[..., 3],
        ],
        axis=-1,
    )
    return converted, valid
