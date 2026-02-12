import numpy as np
from scipy.spatial.transform import Rotation as Rot


class InterpRotation1D:
    def __init__(self, x, y, assume_sorted=False):
        x = np.array(x, copy=True)
        y = np.array(y, copy=True)

        if not assume_sorted:
            ind = np.argsort(x, kind="mergesort")
            x = x[ind]
            y = y[ind]

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim < 2 and y.shape[-2:] != (3, 3):
            raise ValueError("the y array must have at least two dimensions and the last two dimensions are (3,3).")

        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float64)

        self.x = x
        self.y = y
        del y, x

        dx = self.x[1:] - self.x[:-1]
        dy = Rot.from_matrix(self.y[:-1].transpose(0, 2, 1) @ self.y[1:]).as_rotvec()
        self.slope = dy / dx[:, None]

    def __call__(self, x_new) -> np.ndarray:
        if np.any(x_new < self.x[0]) or np.any(x_new > self.x[-1]):
            raise NotImplementedError

        x_new_indices = np.searchsorted(self.x, x_new)
        x_new_indices = x_new_indices.clip(1, len(self.x) - 1).astype(int)

        lo = x_new_indices - 1
        x_lo = self.x[lo]
        y_lo = self.y[lo]

        dx = (x_new - x_lo)[:, None]

        y_new = y_lo @ Rot.from_rotvec(self.slope[lo] * dx).as_matrix()

        return y_new
