"""Load the patient split and build causal per-hour features.

Statistics used for imputation and scaling are fit on training stays only.
Forward-fill never looks past the current hour. A value that has not been
measured yet is replaced with the training median, and the mask says so.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
FEATURE_VERSION = 1

PHYSIO = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
HEADER = PHYSIO + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]
N_PHYSIO = len(PHYSIO)
# 34 imputed values, 34 observation masks, 34 time-since-measurement,
# plus age, gender, MICU, SICU, unknown unit, hours before ICU, and current ICU hour.
N_FEATURES = N_PHYSIO * 3 + 7


def read_stay(path: Path) -> np.ndarray:
    lines = path.read_bytes().splitlines()
    if lines[0].decode("utf-8").strip().split("|") != HEADER:
        raise ValueError(f"unexpected header in {path}")
    rows = np.empty((len(lines) - 1, len(HEADER)), dtype=np.float32)
    for i, line in enumerate(lines[1:]):
        rows[i] = [float(part) for part in line.split(b"|")]
    return rows


def manifest_rows(split: str) -> list[dict]:
    path = DATA / "manifests" / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stay_path(split: str, row: dict) -> Path:
    return DATA / split / row["filename"]


def _unit_one_hot(unit1: float, unit2: float) -> tuple[float, float, float]:
    if unit1 == 1.0:
        return 1.0, 0.0, 0.0
    if unit2 == 1.0:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def transform_stay(raw: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (features[T, C], labels[T]) using only the current and past hours."""
    phys = raw[:, :N_PHYSIO]
    observed = np.isfinite(phys)
    t_index = np.arange(phys.shape[0])[:, None]
    last = np.where(observed, t_index, -1)
    np.maximum.accumulate(last, axis=0, out=last)
    gathered = np.take_along_axis(phys, np.clip(last, 0, None), axis=0)
    filled = np.where(last >= 0, gathered, stats["median"])
    values = (filled - stats["mean"]) / stats["std"]
    time_since = np.where(last >= 0, np.minimum(t_index - last, 24), 24) / 24.0
    age_raw = raw[0, N_PHYSIO]
    age = 0.0 if not np.isfinite(age_raw) else (age_raw - stats["age_mean"]) / stats["age_std"]
    gender = raw[0, N_PHYSIO + 1]
    unit = _unit_one_hot(raw[0, N_PHYSIO + 2], raw[0, N_PHYSIO + 3])
    hosp_raw = raw[0, N_PHYSIO + 4]
    # One training stay has no admission time. Use the training mean so its z-score is 0.
    hosp = 0.0 if not np.isfinite(hosp_raw) else (hosp_raw - stats["hosp_mean"]) / stats["hosp_std"]
    static = np.array([age, gender, unit[0], unit[1], unit[2], hosp], dtype=np.float32)
    static = np.repeat(static[None, :], phys.shape[0], axis=0)
    iculos = (raw[:, N_PHYSIO + 5] / 24.0)[:, None]
    features = np.concatenate(
        [values.astype(np.float32), observed.astype(np.float32), time_since.astype(np.float32), static, iculos.astype(np.float32)],
        axis=1,
    )
    labels = raw[:, -1].astype(np.uint8)
    return features, labels


def _fit_stats(rows: list[dict]) -> dict:
    chunks = [[] for _ in range(N_PHYSIO)]
    ages = []
    hosps = []
    n_pos = 0
    n_hours = 0
    for i, row in enumerate(rows, start=1):
        raw = read_stay(stay_path("train", row))
        phys = raw[:, :N_PHYSIO]
        observed = np.isfinite(phys)
        for j in range(N_PHYSIO):
            if observed[:, j].any():
                chunks[j].append(phys[observed[:, j], j])
        if np.isfinite(raw[0, N_PHYSIO]):
            ages.append(float(raw[0, N_PHYSIO]))
        if np.isfinite(raw[0, N_PHYSIO + 4]):
            hosps.append(float(raw[0, N_PHYSIO + 4]))
        labels = raw[:, -1]
        n_pos += int(np.sum(labels == 1))
        n_hours += int(labels.shape[0])
        if i % 5000 == 0:
            print(f"fit stats {i}/{len(rows)}", flush=True)

    mean = np.zeros(N_PHYSIO, dtype=np.float32)
    std = np.ones(N_PHYSIO, dtype=np.float32)
    median = np.zeros(N_PHYSIO, dtype=np.float32)
    for j in range(N_PHYSIO):
        values = np.concatenate(chunks[j]).astype(np.float64, copy=False)
        mean[j] = float(values.mean())
        deviation = float(values.std())
        std[j] = deviation if deviation > 1e-6 else 1.0
        median[j] = float(np.median(values))
    ages = np.asarray(ages, dtype=np.float64)
    hosps = np.asarray(hosps, dtype=np.float64)
    n_neg = n_hours - n_pos
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "age_mean": float(ages.mean()),
        "age_std": float(ages.std()) or 1.0,
        "hosp_mean": float(hosps.mean()),
        "hosp_std": float(hosps.std()) or 1.0,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_weight": n_neg / n_pos,
    }


def _write_split(split: str, rows: list[dict], stats: dict) -> None:
    n_hours = sum(int(row["n_hours"]) for row in rows)
    features = np.lib.format.open_memmap(
        PROCESSED / f"{split}_features.npy", mode="w+", dtype=np.float32, shape=(n_hours, N_FEATURES)
    )
    labels = np.lib.format.open_memmap(
        PROCESSED / f"{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(n_hours,)
    )
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    index_path = PROCESSED / f"{split}_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "hospital", "start", "n_hours"])
        writer.writeheader()
        cursor = 0
        for i, row in enumerate(rows):
            raw = read_stay(stay_path(split, row))
            if raw.shape[0] != int(row["n_hours"]):
                raise ValueError(f"{row['filename']} has {raw.shape[0]} hours, manifest says {row['n_hours']}")
            x, y = transform_stay(raw, stats)
            if x.shape != (raw.shape[0], N_FEATURES) or not np.isfinite(x).all():
                raise ValueError(f"bad features for {row['filename']}")
            if not np.isin(y, (0, 1)).all():
                raise ValueError(f"bad labels for {row['filename']}")
            end = cursor + raw.shape[0]
            features[cursor:end] = x
            labels[cursor:end] = y
            offsets[i + 1] = end
            writer.writerow(
                {"patient_id": row["patient_id"], "hospital": row["hospital"], "start": cursor, "n_hours": raw.shape[0]}
            )
            cursor = end
            if (i + 1) % 5000 == 0:
                print(f"wrote {split} {i + 1}/{len(rows)}", flush=True)
    if cursor != n_hours:
        raise ValueError(f"{split} wrote {cursor} hours, expected {n_hours}")
    features.flush()
    labels.flush()
    del features, labels
    np.save(PROCESSED / f"{split}_offsets.npy", offsets)


def prepare(force: bool = False) -> dict:
    marker = PROCESSED / "READY"
    if marker.exists() and not force:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("version") == FEATURE_VERSION:
            return meta
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("fitting imputation and scaling on training stays", flush=True)
    train_rows = manifest_rows("train")
    stats = _fit_stats(train_rows)
    arrays = {
        "mean": stats["mean"],
        "std": stats["std"],
        "median": stats["median"],
        "age_mean": np.float32(stats["age_mean"]),
        "age_std": np.float32(stats["age_std"]),
        "hosp_mean": np.float32(stats["hosp_mean"]),
        "hosp_std": np.float32(stats["hosp_std"]),
    }
    for split in ("train", "dev", "test"):
        print(f"building {split} features", flush=True)
        _write_split(split, manifest_rows(split), arrays)
    meta = {
        "version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "pos_weight": stats["pos_weight"],
        "n_pos": stats["n_pos"],
        "n_neg": stats["n_neg"],
        "age_mean": stats["age_mean"],
        "age_std": stats["age_std"],
        "hosp_mean": stats["hosp_mean"],
        "hosp_std": stats["hosp_std"],
    }
    np.savez(PROCESSED / "stats.npz", **arrays)
    (PROCESSED / "stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class StayDataset(Dataset):
    def __init__(self, split: str):
        self.features = np.load(PROCESSED / f"{split}_features.npy", mmap_mode="r")
        self.labels = np.load(PROCESSED / f"{split}_labels.npy", mmap_mode="r")
        self.offsets = np.load(PROCESSED / f"{split}_offsets.npy")

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def __getitem__(self, index: int):
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        x = np.array(self.features[start:end], dtype=np.float32).T
        y = np.array(self.labels[start:end], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def collate(batch):
    xs, ys = zip(*batch)
    lengths = [int(y.shape[0]) for y in ys]
    max_len = max(lengths)
    x = torch.zeros(len(batch), xs[0].shape[0], max_len)
    y = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (features, labels) in enumerate(zip(xs, ys)):
        n = labels.shape[0]
        x[i, :, :n] = features
        y[i, :n] = labels
        mask[i, :n] = True
    return x, y, mask


def load_label_sequences(split: str) -> list[np.ndarray]:
    labels = np.load(PROCESSED / f"{split}_labels.npy", mmap_mode="r")
    offsets = np.load(PROCESSED / f"{split}_offsets.npy")
    return [np.array(labels[int(a) : int(b)], dtype=np.int32) for a, b in zip(offsets[:-1], offsets[1:])]
