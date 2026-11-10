"""Causal 6-hour and 24-hour summaries for the CNN.

For every vital and lab, each hour gets the mean, min, and max of the
z-scored measurements in the window, plus the fraction of those hours that
were actually measured. The window ends at the current hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numba
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import N_FEATURES, N_PHYSIO, PROCESSED

WINDOWS = (6, 24)
STATS_PER_WINDOW = 4
N_WINDOW_FEATURES = N_PHYSIO * STATS_PER_WINDOW * len(WINDOWS)
N_FEATURES_WINDOWS = N_FEATURES + N_WINDOW_FEATURES
ROOT = Path(__file__).resolve().parent / "data" / "processed_windows"


@numba.njit
def _window_min_max(values, observed, window):
    steps, channels = values.shape
    low = np.zeros((steps, channels), dtype=np.float32)
    high = np.zeros((steps, channels), dtype=np.float32)
    for column in range(channels):
        for hour in range(steps):
            start = 0 if hour + 1 < window else hour + 1 - window
            lo = np.float32(1.0e30)
            hi = np.float32(-1.0e30)
            seen = False
            for cursor in range(start, hour + 1):
                if observed[cursor, column]:
                    value = values[cursor, column]
                    if value < lo:
                        lo = value
                    if value > hi:
                        hi = value
                    seen = True
            if seen:
                low[hour, column] = lo
                high[hour, column] = hi
    return low, high


def window_block(phys: np.ndarray, mean: np.ndarray, std: np.ndarray, window: int) -> np.ndarray:
    observed = np.isfinite(phys)
    scaled = np.where(observed, (phys - mean) / std, 0.0).astype(np.float32)
    count_c = np.cumsum(observed, axis=0)
    sum_c = np.cumsum(scaled, axis=0)
    count = count_c.astype(np.float32, copy=True)
    total = sum_c.astype(np.float32, copy=True)
    if phys.shape[0] > window:
        count[window:] -= count_c[:-window]
        total[window:] -= sum_c[:-window]
    average = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    low, high = _window_min_max(scaled, observed, window)
    available = np.minimum(np.arange(1, phys.shape[0] + 1), window).astype(np.float32)[:, None]
    fraction = count / available
    return np.concatenate([average, low, high, fraction], axis=1).astype(np.float32)


def window_features(phys: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.concatenate([window_block(phys, mean, std, window) for window in WINDOWS], axis=1)


def self_check() -> None:
    _window_min_max(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.bool_), 1)
    phys = np.full((8, N_PHYSIO), np.nan, dtype=np.float32)
    phys[2, 0] = 80.0
    phys[6, 0] = 120.0
    mean = np.zeros(N_PHYSIO, dtype=np.float32)
    std = np.ones(N_PHYSIO, dtype=np.float32)
    block = window_block(phys, mean, std, 6)
    # mean, min, max, fraction for channel 0. Hour 1 is before the first measurement.
    if not np.allclose(block[1, [0, N_PHYSIO, 2 * N_PHYSIO, 3 * N_PHYSIO]], 0.0):
        raise AssertionError("window used a future measurement")
    hour2 = block[2, [0, N_PHYSIO, 2 * N_PHYSIO]]
    if not np.allclose(hour2, 80.0) or not np.isclose(block[2, 3 * N_PHYSIO], 1.0 / 3.0):
        raise AssertionError(f"hour 2 window was {block[2, [0, N_PHYSIO, 2 * N_PHYSIO, 3 * N_PHYSIO]]}")
    hour6 = block[6, [0, N_PHYSIO, 2 * N_PHYSIO, 3 * N_PHYSIO]]
    if not np.allclose(hour6, [100.0, 80.0, 120.0, 2.0 / 6.0]):
        raise AssertionError(f"hour 6 window was {hour6}")
    features = window_features(phys, mean, std)
    if features.shape != (8, N_WINDOW_FEATURES) or not np.isfinite(features).all():
        raise AssertionError("window features were the wrong shape or not finite")
    print("window feature self-check passed", flush=True)


def build(force: bool = False) -> None:
    marker = ROOT / "READY"
    if marker.exists() and not force:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("n_features") == N_FEATURES_WINDOWS:
            return
    self_check()
    stats = np.load(PROCESSED / "stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        values = np.load(PROCESSED / f"raw_{split}_values.npy", mmap_mode="r")
        offsets = np.load(PROCESSED / f"raw_{split}_offsets.npy")
        feature_offsets = np.load(PROCESSED / f"{split}_offsets.npy")
        if not np.array_equal(offsets, feature_offsets):
            raise RuntimeError(f"{split} raw and feature offsets differ")
        n_hours = int(offsets[-1])
        out = np.lib.format.open_memmap(
            ROOT / f"{split}_windows.npy", mode="w+", dtype=np.float32, shape=(n_hours, N_WINDOW_FEATURES)
        )
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            phys = np.asarray(values[int(start) : int(end), :N_PHYSIO], dtype=np.float32)
            block = window_features(phys, mean, std)
            if not np.isfinite(block).all():
                raise RuntimeError(f"non-finite window features in {split} patient {i}")
            out[int(start) : int(end)] = block
            if (i + 1) % 4000 == 0:
                print(f"window features {split} {i + 1}/{len(offsets) - 1}", flush=True)
        out.flush()
        del out
        print(f"window features {split} done, {n_hours} hours", flush=True)
    meta = {"n_features": N_FEATURES_WINDOWS, "windows": list(WINDOWS), "base_features": N_FEATURES}
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")


class WindowStayDataset(Dataset):
    def __init__(self, split: str):
        self.base = np.load(PROCESSED / f"{split}_features.npy", mmap_mode="r")
        self.extra = np.load(ROOT / f"{split}_windows.npy", mmap_mode="r")
        self.labels = np.load(PROCESSED / f"{split}_labels.npy", mmap_mode="r")
        self.offsets = np.load(PROCESSED / f"{split}_offsets.npy")

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def __getitem__(self, index: int):
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        features = np.concatenate([self.base[start:end], self.extra[start:end]], axis=1)
        target = np.asarray(self.labels[start:end], dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(features.T)), torch.from_numpy(target)
