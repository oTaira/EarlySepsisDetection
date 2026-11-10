"""Retrain Separatrix with causal imputation.

Missing values hold the last past measurement. Hours before the first
measurement use the training mean. The window code is otherwise the same.
The original 407-column model and the lab-window model are both retrained.
Thresholds are chosen on dev. Test is scored once for each model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset import PROCESSED
from train_cnn_baseline import operating_metrics
from train_combined import LABS, PER_LAB, _lab_block
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_labs import fit, predict_split
from utility import choose_threshold
from winner_data import (
    _causal_fill,
    _index,
    labels_of,
    load_raw,
    separatrix_missingness,
    separatrix_type1,
    sequences,
    training_means,
)

RUN = Path(__file__).resolve().parent / "runs" / "separatrix_causal"


def _check_fill() -> None:
    filled = _causal_fill(np.array([np.nan, np.nan, 10.0, np.nan, 30.0]), 0.0)
    expected = np.array([0.0, 0.0, 10.0, 10.0, 30.0])
    if not np.allclose(filled, expected):
        raise RuntimeError(f"causal fill returned {filled}")


def causal_separatrix(split: str, means: np.ndarray) -> np.ndarray:
    path = PROCESSED / f"separatrix_causal_{split}.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        print(f"causal separatrix {split} {cached.shape}", flush=True)
        return cached
    from winner_data import SEP_COVARIATES

    values, _, offsets = load_raw(split)
    cov_idx = [_index(name) for name in SEP_COVARIATES]
    miss_idx = [j for j in range(40) if j not in (_index("Age"), _index("Gender"))]
    matrix = np.zeros((int(offsets[-1]), 407), dtype=np.float32)
    type1 = np.zeros((336, 245), dtype=np.float64)
    type2 = np.zeros((336, 162), dtype=np.float64)
    for patient, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw = np.asarray(values[int(start) : int(end)], dtype=np.float64)
        length = raw.shape[0]
        chosen = raw[:, cov_idx]
        imputed = np.column_stack([_causal_fill(chosen[:, j], float(means[cov_idx[j]])) for j in range(13)])
        present_cov = np.isfinite(chosen).astype(np.float64)
        separatrix_type1(imputed, present_cov, type1[:length])
        present_miss = np.isfinite(raw[:, miss_idx]).astype(np.float64)
        separatrix_missingness(present_miss, type2[:length])
        matrix[int(start) : int(end), :245] = type1[:length]
        matrix[int(start) : int(end), 245:] = type2[:length]
        if (patient + 1) % 4000 == 0:
            print(f"causal separatrix {split} {patient + 1}/{offsets.size - 1}", flush=True)
    np.save(path, matrix)
    print(f"cached {path} {matrix.shape}", flush=True)
    return np.load(path, mmap_mode="r")


def causal_labs(split: str, means: np.ndarray) -> np.ndarray:
    path = PROCESSED / f"lab_windows_causal_{split}.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        print(f"causal labs {split} {cached.shape}", flush=True)
        return cached
    values, _, offsets = load_raw(split)
    columns = [_index(name) for name in LABS]
    matrix = np.zeros((int(offsets[-1]), len(LABS) * PER_LAB), dtype=np.float32)
    block = np.zeros((336, PER_LAB), dtype=np.float64)
    for patient, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw = np.asarray(values[int(start) : int(end)], dtype=np.float64)
        length = raw.shape[0]
        for lab, column in enumerate(columns):
            series = raw[:, column]
            imputed = _causal_fill(series, float(means[column]))
            observed = np.isfinite(series).astype(np.float64)
            _lab_block(imputed, observed, block[:length])
            matrix[int(start) : int(end), lab * PER_LAB : (lab + 1) * PER_LAB] = block[:length]
        if (patient + 1) % 4000 == 0:
            print(f"causal labs {split} {patient + 1}/{offsets.size - 1}", flush=True)
    np.save(path, matrix)
    print(f"cached {path} {matrix.shape}", flush=True)
    return np.load(path, mmap_mode="r")


def empty_labs(rows: int) -> np.ndarray:
    return np.zeros((rows, 0), dtype=np.float32)


def standardization(separatrix, labs):
    from train_combined import standardization as standardize

    if labs.shape[1] == 0:
        return standardize(separatrix, np.zeros((separatrix.shape[0], 1), dtype=np.float32))[0][:407], None
    return standardize(separatrix, labs)


def moments(matrix) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    total_sq = np.zeros(matrix.shape[1], dtype=np.float64)
    count = matrix.shape[0]
    for start in range(0, count, 200_000):
        end = min(count, start + 200_000)
        values = np.asarray(matrix[start:end], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    mu = (total / count).astype(np.float32)
    variance = total_sq / count - np.square(total / count)
    sd = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    return mu, sd


def evaluate(name, models, keep, mu, sd, separatrix, labs) -> dict:
    scored = {}
    hospitals = {}
    threshold = None
    for split in ("dev", "test", "train"):
        _, _, offsets = load_raw(split)
        probabilities = sequences(predict_split(models, keep, mu, sd, separatrix[split], labs[split]), offsets)
        labels = labels_of(split)
        if split == "dev":
            threshold, _ = choose_threshold(labels, probabilities)
            print(f"{name} threshold {threshold:.3f}", flush=True)
        overall = operating_metrics(labels, probabilities, threshold)
        overall["patients"] = len(labels)
        sites = by_hospital(labels, probabilities, hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = sites
        print_scores(f"{name} {split}", overall, sites)
    return {"threshold": threshold, "kept_columns": len(keep), "train": scored["train"], "dev": scored["dev"], "test": scored["test"], "hospitals": hospitals}


def main() -> None:
    _check_fill()
    means = training_means()
    print("building causal features", flush=True)
    separatrix = {split: causal_separatrix(split, means) for split in ("train", "dev", "test")}
    labs = {split: causal_labs(split, means) for split in ("train", "dev", "test")}
    y_train_offsets = load_raw("train")[2]
    from train_winners import flat_labels

    y_train = flat_labels("train")
    RUN.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, lab_set in (
        ("separatrix", {split: empty_labs(separatrix[split].shape[0]) for split in separatrix}),
        ("separatrix_labs", labs),
    ):
        print(f"\ntraining {name}", flush=True)
        mu, sd = moments(separatrix["train"]) if lab_set["train"].shape[1] == 0 else moments_pair(separatrix["train"], lab_set["train"])
        if lab_set["train"].shape[1] == 0:
            mu_s, sd_s = moments(separatrix["train"])
            models, keep, _gain = fit(separatrix["train"], lab_set["train"], mu_s, sd_s, y_train, y_train_offsets)
            results[name] = evaluate(name, models, keep, mu_s, sd_s, separatrix, lab_set)
        else:
            mu_s, sd_s = moments_pair(separatrix["train"], lab_set["train"])
            models, keep, gain = fit(separatrix["train"], lab_set["train"], mu_s, sd_s, y_train, y_train_offsets)
            results[name] = evaluate(name, models, keep, mu_s, sd_s, separatrix, lab_set)
            results[name]["lab_gain"] = {lab: 0.0 for lab in LABS}
            for key, value in gain.items():
                column = int(key[1:])
                if column >= 407:
                    results[name]["lab_gain"][LABS[(column - 407) // PER_LAB]] += float(value)
        del mu, sd
    payload = {
        "imputation": "last past value; training mean before the first measurement",
        "leaky_separatrix_test_utility": 0.4545682369775407,
        "leaky_separatrix_labs_test_utility": 0.5169,
        "results": results,
    }
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, row in results.items():
        print(f"{name} test {row['test']['utility']:.4f}", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


def moments_pair(separatrix, labs):
    width = separatrix.shape[1] + labs.shape[1]
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    count = separatrix.shape[0]
    for start in range(0, count, 200_000):
        end = min(count, start + 200_000)
        block = np.concatenate([np.asarray(separatrix[start:end]), np.asarray(labs[start:end])], axis=1)
        values = block.astype(np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    mu = (total / count).astype(np.float32)
    variance = total_sq / count - np.square(total / count)
    sd = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    return mu, sd


if __name__ == "__main__":
    main()
