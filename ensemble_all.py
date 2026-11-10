"""Blend the CNN, the XGBoost models, the best LSTM, and the best attention model.

Weights are drawn on the simplex and the alarm threshold is chosen with them,
both on dev utility. Test is scored once for the selected blend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from dataset import StayDataset
from ensemble_cnn_separatrix import (
    cnn_probabilities,
    fit_separatrix,
    separatrix_probabilities,
    standardize,
)
from search_cnn_broad import predict_ordered
from search_sequence import build_model
from train_cnn_baseline import operating_metrics
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_mlp import cached_matrix
from train_winners import (
    SEED,
    build_ctl,
    build_sepsyd,
    fit_xgb,
    flat_labels,
    log_columns,
    positive_weight,
    sepsyd_means,
    shift_labels,
)
from utility import choose_threshold
from winner_data import labels_of, load_raw, sequences, training_means

RUN = Path(__file__).resolve().parent / "runs" / "ensemble_all"
SEQUENCE = Path(__file__).resolve().parent / "runs" / "sequence_search"
NAMES = ("cnn", "separatrix", "sepsyd", "ctl", "lstm", "attention")


def tree_scores(flat_for_split) -> dict[str, list[np.ndarray]]:
    output = {}
    for split in ("train", "dev", "test"):
        _, _, offsets = load_raw(split)
        output[split] = sequences(flat_for_split(split), offsets)
    return output


def fit_sepsyd():
    print("training Sepsyd", flush=True)
    skewed = log_columns()
    center = sepsyd_means(skewed)
    x_train = build_sepsyd("train", center, skewed)
    y_train = flat_labels("train")
    x_dev = build_sepsyd("dev", center, skewed)
    y_dev = flat_labels("dev")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=40,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=25,
        eval_metric="aucpr",
        random_state=SEED,
    )
    fit_xgb(model, x_train, y_train, x_dev, y_dev)

    def scores(split: str):
        matrix = {"train": x_train, "dev": x_dev}.get(split)
        if matrix is None:
            matrix = build_sepsyd(split, center, skewed)
        return model.predict_proba(matrix)[:, 1]

    return tree_scores(scores)


def fit_ctl():
    print("training CTL", flush=True)
    x_train = build_ctl("train")
    y_train = np.concatenate(shift_labels(labels_of("train"))).astype(np.int32)
    x_dev = build_ctl("dev")
    y_dev = np.concatenate(shift_labels(labels_of("dev"))).astype(np.int32)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=positive_weight(y_train),
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="aucpr",
        random_state=SEED,
        missing=np.nan,
    )
    fit_xgb(model, x_train, y_train, x_dev, y_dev)

    def scores(split: str):
        matrix = {"train": x_train, "dev": x_dev}.get(split)
        if matrix is None:
            matrix = build_ctl(split)
        return model.predict_proba(matrix)[:, 1]

    return tree_scores(scores)


def neural_scores(family: str, device) -> dict[str, list[np.ndarray]]:
    saved = torch.load(SEQUENCE / f"best_{family}.pt", map_location=device, weights_only=False)
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    output = {}
    for split in ("train", "dev", "test"):
        output[split] = predict_ordered(model, StayDataset(split), 128, device)
        print(f"{family} {split}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def blend(parts: dict[str, list[np.ndarray]], weights: np.ndarray, kind: str) -> list[np.ndarray]:
    mixed = []
    columns = [parts[name] for name in NAMES]
    for rows in zip(*columns):
        acc = None
        for weight, row in zip(weights, rows):
            if weight == 0:
                continue
            values = np.clip(np.asarray(row, dtype=np.float64), 1e-6, 1 - 1e-6)
            term = weight * values if kind == "probability" else weight * np.log(values / (1.0 - values))
            acc = term if acc is None else acc + term
        if kind == "logit":
            acc = 1.0 / (1.0 + np.exp(-acc))
        mixed.append(acc)
    return mixed


def candidates(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    weights = [np.eye(len(NAMES))[i] for i in range(len(NAMES))]
    pair = np.zeros(len(NAMES))
    pair[NAMES.index("cnn")] = 0.2
    pair[NAMES.index("separatrix")] = 0.8
    weights.append(pair)
    for _ in range(count):
        weights.append(rng.dirichlet(np.ones(len(NAMES))))
    return weights


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    means = training_means()
    models, keep, mu, sd = fit_separatrix(means)
    separatrix = {}
    for split in ("train", "dev", "test"):
        matrix = standardize(cached_matrix(split, means), mu, sd)[:, keep]
        _, _, offsets = load_raw(split)
        separatrix[split] = sequences(separatrix_probabilities(models, matrix), offsets)
        print(f"separatrix {split} features {len(keep)}", flush=True)
    parts = {
        "cnn": cnn_probabilities(device),
        "separatrix": separatrix,
        "sepsyd": fit_sepsyd(),
        "ctl": fit_ctl(),
        "lstm": neural_scores("lstm", device),
        "attention": neural_scores("attention", device),
    }
    labels = {split: labels_of(split) for split in ("train", "dev", "test")}
    for name in NAMES:
        for split in labels:
            if len(parts[name][split]) != len(labels[split]):
                raise RuntimeError(f"{name} {split} does not match the stays")
    dev_labels = labels["dev"]
    rng = np.random.default_rng(230)
    best = None
    checked = 0
    for kind in ("probability", "logit"):
        pool = candidates(rng, 1200)
        for weights in pool:
            mixed = blend({name: parts[name]["dev"] for name in NAMES}, weights, kind)
            threshold, utility = choose_threshold(dev_labels, mixed)
            checked += 1
            if best is None or utility > best["dev_utility"]:
                best = {
                    "kind": kind,
                    "weights": {name: float(weight) for name, weight in zip(NAMES, weights)},
                    "threshold": float(threshold),
                    "dev_utility": float(utility),
                }
                print(
                    f"new best {kind} dev {utility:.4f} "
                    + " ".join(f"{name}={weight:.2f}" for name, weight in best["weights"].items()),
                    flush=True,
                )
    print(f"checked {checked} blends. selected {json.dumps(best)}", flush=True)
    chosen = np.array([best["weights"][name] for name in NAMES])
    scored = {}
    hospitals = {}
    for split in ("train", "dev", "test"):
        mixed = blend({name: parts[name][split] for name in NAMES}, chosen, best["kind"])
        overall = operating_metrics(labels[split], mixed, best["threshold"])
        overall["patients"] = len(labels[split])
        by_site = by_hospital(labels[split], mixed, hospital_names(split), best["threshold"])
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"full ensemble {split}", overall, by_site)
    payload = {
        "models": list(NAMES),
        "selection": "simplex weights and alarm threshold chosen on dev utility",
        "blends_checked": checked,
        "selected": best,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "cnn_separatrix_test_utility": 0.4580,
        "separatrix_test_utility": 0.4545682369775407,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"full ensemble dev {scored['dev']['utility']:.4f} test {scored['test']['utility']:.4f}", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
