"""Retrain Separatrix on its own columns plus the eight lab window summaries.

The gain screen, the five undersampled XGBoost models, and the geometric
mean are unchanged. The alarm threshold is chosen on dev. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from ensemble_cnn_separatrix import separatrix_probabilities
from train_cnn_baseline import operating_metrics
from train_combined import LABS, PER_LAB, lab_matrix, standardization
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_mlp import cached_matrix
from train_winners import SEED, _hours, _undersample, flat_labels
from utility import choose_threshold
from winner_data import labels_of, load_raw, sequences, training_means

RUN = Path(__file__).resolve().parent / "runs" / "separatrix_labs"
BASE_COLUMNS = 407


def gather(separatrix, labs, mu, sd, index) -> np.ndarray:
    block = np.concatenate(
        [np.asarray(separatrix[index], dtype=np.float32), np.asarray(labs[index], dtype=np.float32)], axis=1
    )
    return (block - mu) / sd


def predict_split(models, keep, mu, sd, separatrix, labs) -> np.ndarray:
    count = separatrix.shape[0]
    probabilities = np.empty(count, dtype=np.float64)
    kept = np.asarray(keep)
    for start in range(0, count, 65_536):
        end = min(count, start + 65_536)
        block = gather(separatrix, labs, mu, sd, slice(start, end))
        probabilities[start:end] = separatrix_probabilities(models, block[:, kept])
    return probabilities


def fit(separatrix, labs, mu, sd, labels, offsets):
    rng = np.random.default_rng(SEED)
    patients = np.arange(offsets.size - 1)
    rng.shuffle(patients)
    n_screen = max(1, int(0.10 * patients.size))
    screen = _undersample(labels, _hours(patients[:n_screen], offsets), rng)
    screener = XGBClassifier(
        n_estimators=40, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=SEED
    )
    screener.fit(gather(separatrix, labs, mu, sd, screen), labels[screen], verbose=False)
    gain = screener.get_booster().get_score(importance_type="gain")
    keep = sorted(int(name[1:]) for name in gain if gain[name] > 0)
    if not keep:
        keep = list(range(separatrix.shape[1] + labs.shape[1]))
    lab_kept = sum(1 for column in keep if column >= BASE_COLUMNS)
    print(f"kept {len(keep)} of {BASE_COLUMNS + labs.shape[1]} features, {lab_kept} from the lab windows", flush=True)
    models = []
    for fold, ids in enumerate(np.array_split(patients[n_screen:], 5)):
        sample = _undersample(labels, _hours(ids, offsets), rng)
        model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(gather(separatrix, labs, mu, sd, sample)[:, keep], labels[sample], verbose=False)
        models.append(model)
        print(f"fold {fold + 1}/5", flush=True)
    return models, keep, gain


def lab_gain(gain: dict) -> dict[str, float]:
    totals = {name: 0.0 for name in LABS}
    for name, value in gain.items():
        column = int(name[1:])
        if column < BASE_COLUMNS:
            continue
        totals[LABS[(column - BASE_COLUMNS) // PER_LAB]] += float(value)
    return totals


def main() -> None:
    print("loading combined features", flush=True)
    means = training_means()
    separatrix = {split: cached_matrix(split, means) for split in ("train", "dev", "test")}
    labs = {split: lab_matrix(split, means) for split in ("train", "dev", "test")}
    mu, sd = standardization(separatrix["train"], labs["train"])
    y_train = flat_labels("train")
    _, _, offsets = load_raw("train")
    models, keep, gain = fit(separatrix["train"], labs["train"], mu, sd, y_train, offsets)
    gained = lab_gain(gain)
    print("lab gain " + " ".join(f"{name}={value:.1f}" for name, value in sorted(gained.items(), key=lambda item: -item[1])), flush=True)
    scored = {}
    hospitals = {}
    threshold = None
    for split in ("dev", "test", "train"):
        _, _, split_offsets = load_raw(split)
        probabilities = sequences(
            predict_split(models, keep, mu, sd, separatrix[split], labs[split]), split_offsets
        )
        labels = labels_of(split)
        if split == "dev":
            threshold, _ = choose_threshold(labels, probabilities)
            print(f"threshold {threshold:.3f}", flush=True)
        overall = operating_metrics(labels, probabilities, threshold)
        overall["patients"] = len(labels)
        sites = by_hospital(labels, probabilities, hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = sites
        print_scores(split, overall, sites)
    RUN.mkdir(parents=True, exist_ok=True)
    for index, model in enumerate(models):
        model.save_model(RUN / f"fold_{index}.json")
    payload = {
        "model": "Separatrix trees with eight lab window summaries added",
        "labs": LABS,
        "kept_columns": keep,
        "lab_columns_kept": sum(1 for column in keep if column >= BASE_COLUMNS),
        "lab_gain": gained,
        "threshold": threshold,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "separatrix_test_utility": 0.4545682369775407,
        "combined_lstm_test_utility": 0.3928,
    }
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"test {scored['test']['utility']:.4f}", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
