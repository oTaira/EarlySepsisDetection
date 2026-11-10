"""A few causal clinical summaries added to the hourly CNN inputs.

Shock index, partial SOFA, a two-item qSOFA, and how often vitals and labs
were measured over the last 24 hours. Nothing here reads a future hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset import N_PHYSIO, PROCESSED
from winner_data import load_raw

ROOT = Path(__file__).resolve().parent / "data" / "processed_clinical"
COLUMNS = ("shock_z", "shock_known", "sofa", "qsofa", "vital_rate_24", "lab_rate_24")
N_CLINICAL = len(COLUMNS)


def _causal_fill(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.where(np.isfinite(values), np.arange(values.shape[0]), -1)
    np.maximum.accumulate(index, out=index)
    filled = np.zeros(values.shape[0], dtype=np.float32)
    seen = index >= 0
    filled[seen] = values[index[seen]]
    return filled, seen


def _score_sofa(platelets, bilirubin, map_, creatinine, seen) -> np.ndarray:
    score = np.zeros(platelets.shape[0], dtype=np.float32)
    use = seen[:, 0]
    score[use] += np.where(platelets[use] < 20, 4, np.where(platelets[use] < 50, 3, np.where(platelets[use] < 100, 2, np.where(platelets[use] < 150, 1, 0))))
    use = seen[:, 1]
    score[use] += np.where(bilirubin[use] >= 12, 4, np.where(bilirubin[use] >= 6, 3, np.where(bilirubin[use] >= 2, 2, np.where(bilirubin[use] >= 1.2, 1, 0))))
    use = seen[:, 2]
    score[use] += (map_[use] < 70).astype(np.float32)
    use = seen[:, 3]
    score[use] += np.where(creatinine[use] >= 5, 4, np.where(creatinine[use] >= 3.5, 3, np.where(creatinine[use] >= 2, 2, np.where(creatinine[use] >= 1.2, 1, 0))))
    return score


def _rate(observed: np.ndarray, window: int = 24) -> np.ndarray:
    cumulative = np.cumsum(observed, axis=0)
    count = cumulative.copy()
    if observed.shape[0] > window:
        count[window:] -= cumulative[:-window]
    available = np.minimum(np.arange(1, observed.shape[0] + 1), window).astype(np.float32)
    return (count / available[:, None]).mean(axis=1).astype(np.float32)


def stay_clinical(phys: np.ndarray) -> np.ndarray:
    heart, heart_seen = _causal_fill(phys[:, 0])
    pressure, pressure_seen = _causal_fill(phys[:, 3])
    known = heart_seen & pressure_seen & (pressure >= 40)
    shock = np.zeros(phys.shape[0], dtype=np.float32)
    shock[known] = heart[known] / pressure[known]
    platelets, platelets_seen = _causal_fill(phys[:, 33])
    bilirubin, bilirubin_seen = _causal_fill(phys[:, 25])
    arterial, arterial_seen = _causal_fill(phys[:, 4])
    creatinine, creatinine_seen = _causal_fill(phys[:, 19])
    respiration, respiration_seen = _causal_fill(phys[:, 6])
    component_seen = np.stack((platelets_seen, bilirubin_seen, arterial_seen, creatinine_seen), axis=1)
    sofa = _score_sofa(platelets, bilirubin, arterial, creatinine, component_seen) / 13.0
    qsofa = np.zeros(phys.shape[0], dtype=np.float32)
    qsofa[respiration_seen] += (respiration[respiration_seen] >= 22).astype(np.float32)
    qsofa[pressure_seen] += (pressure[pressure_seen] <= 100).astype(np.float32)
    qsofa /= 2.0
    out = np.stack(
        (
            shock,
            known.astype(np.float32),
            sofa,
            qsofa,
            _rate(np.isfinite(phys[:, :8])),
            _rate(np.isfinite(phys[:, 8:N_PHYSIO])),
        ),
        axis=1,
    )
    return out.astype(np.float32)


def self_check() -> None:
    phys = np.full((6, N_PHYSIO), np.nan, dtype=np.float32)
    phys[2, 0] = 120
    phys[2, 3] = 80
    phys[4, 33] = 40
    block = stay_clinical(phys)
    if block[1, 1] != 0 or not np.isclose(block[2, 0], 1.5) or block[2, 1] != 1:
        raise AssertionError(f"shock used the wrong hours: {block[:3, :2]}")
    if not np.isclose(block[4, 2], 3 / 13):
        raise AssertionError(f"partial SOFA was {block[4, 2]}")
    if block[0, 4] != 0 or block[2, 4] <= 0:
        raise AssertionError("measurement rate leaked or missed an observation")
    print("clinical feature self-check passed", flush=True)


def build(force: bool = False) -> dict:
    marker = ROOT / "READY"
    if marker.exists() and not force:
        return json.loads(marker.read_text(encoding="utf-8"))
    self_check()
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = {}
    for split in ("train", "dev", "test"):
        values, _, offsets = load_raw(split)
        feature_offsets = np.load(PROCESSED / f"{split}_offsets.npy")
        if not np.array_equal(offsets, feature_offsets):
            raise RuntimeError(f"{split} offsets do not match")
        out = np.lib.format.open_memmap(
            ROOT / f"{split}.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), N_CLINICAL)
        )
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            phys = np.asarray(values[int(start) : int(end), :N_PHYSIO], dtype=np.float32)
            out[int(start) : int(end)] = stay_clinical(phys)
            if (index + 1) % 8000 == 0:
                print(f"clinical {split} {index + 1}/{len(offsets) - 1}", flush=True)
        out.flush()
        raw[split] = out
        print(f"clinical {split} done", flush=True)
    known = raw["train"][:, 1] > 0
    shock = raw["train"][known, 0].astype(np.float64)
    center = float(shock.mean())
    scale = float(shock.std()) or 1.0
    for split, table in raw.items():
        table[:, 0] = (table[:, 0] - center) / scale
        table.flush()
        del table
    meta = {"columns": list(COLUMNS), "shock_mean": center, "shock_std": scale, "n_features": 109 + N_CLINICAL}
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
