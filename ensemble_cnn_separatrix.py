"""Blend the regularized CNN with the Separatrix trees.

The mixing weight is chosen on dev utility, and so is the alarm threshold.
Test is scored once for that pair. Separatrix is retrained with the same
seed and the same cached feature matrix as the earlier run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from dataset import N_FEATURES, StayDataset, collate, load_label_sequences
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, predict
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_mlp import cached_matrix
from train_winners import SEED, _hours, _undersample, flat_labels
from utility import choose_threshold
from winner_data import labels_of, load_raw, sequences, training_means

CNN_RUN = Path(__file__).resolve().parent / "runs" / "cnn_regularized"
RUN = Path(__file__).resolve().parent / "runs" / "ensemble_cnn_separatrix"
WEIGHTS = np.linspace(0.0, 1.0, 21)


def standardize(matrix: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=np.float32) - mu) / sd


def fit_separatrix(means):
    print("retraining Separatrix trees", flush=True)
    raw = cached_matrix("train", means)
    y_train = flat_labels("train")
    mu = np.asarray(raw, dtype=np.float64).mean(axis=0).astype(np.float32)
    sd = np.asarray(raw, dtype=np.float64).std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    x_train = standardize(raw, mu, sd)
    _, _, offsets = load_raw("train")
    rng = np.random.default_rng(SEED)
    patients = np.arange(offsets.size - 1)
    rng.shuffle(patients)
    n_screen = max(1, int(0.10 * patients.size))
    screen = _undersample(y_train, _hours(patients[:n_screen], offsets), rng)
    screener = XGBClassifier(
        n_estimators=40, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=SEED
    )
    screener.fit(x_train[screen], y_train[screen], verbose=False)
    gain = screener.get_booster().get_score(importance_type="gain")
    keep = sorted(int(name[1:]) for name in gain if gain[name] > 0)
    print(f"kept {len(keep)} of {x_train.shape[1]} features", flush=True)
    models = []
    for fold, ids in enumerate(np.array_split(patients[n_screen:], 5)):
        sample = _undersample(y_train, _hours(ids, offsets), rng)
        model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(x_train[sample][:, keep], y_train[sample], verbose=False)
        models.append(model)
        print(f"fold {fold + 1}/5", flush=True)
    return models, keep, mu, sd


def separatrix_probabilities(models, matrix) -> np.ndarray:
    clipped = [np.clip(model.predict_proba(matrix)[:, 1], 1e-6, 1 - 1e-6) for model in models]
    return np.exp(np.mean(np.log(np.vstack(clipped)), axis=0))


def cnn_probabilities(device) -> dict[str, list[np.ndarray]]:
    saved = torch.load(CNN_RUN / "model.pt", map_location=device, weights_only=False)
    config = saved["config"]
    model = SimpleCausalCNN(
        N_FEATURES,
        hidden=config["hidden"],
        kernel_size=config["kernel_size"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        activation=SWInSiLU(config["swin_a"], config["swin_omega"]),
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    output = {}
    for split in ("train", "dev", "test"):
        loader = DataLoader(StayDataset(split), batch_size=128, shuffle=False, collate_fn=collate)
        output[split] = predict(model, loader, device)
        print(f"cnn {split}", flush=True)
    return output


def blend(left, right, weight: float, kind: str) -> list[np.ndarray]:
    mixed = []
    for sep, cnn in zip(left, right):
        sep = np.clip(np.asarray(sep, dtype=np.float64), 1e-6, 1 - 1e-6)
        cnn = np.clip(np.asarray(cnn, dtype=np.float64), 1e-6, 1 - 1e-6)
        if kind == "probability":
            mixed.append(weight * sep + (1.0 - weight) * cnn)
        else:
            combined = weight * np.log(sep / (1.0 - sep)) + (1.0 - weight) * np.log(cnn / (1.0 - cnn))
            mixed.append(1.0 / (1.0 + np.exp(-combined)))
    return mixed


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = training_means()
    models, keep, mu, sd = fit_separatrix(means)
    trees = {}
    labels = {}
    for split in ("train", "dev", "test"):
        matrix = standardize(cached_matrix(split, means), mu, sd)[:, keep]
        _, _, offsets = load_raw(split)
        trees[split] = sequences(separatrix_probabilities(models, matrix), offsets)
        labels[split] = labels_of(split)
        if any(len(a) != len(b) for a, b in zip(trees[split], labels[split])):
            raise RuntimeError(f"{split} tree scores do not match the stays")
        print(f"separatrix {split}", flush=True)
    cnn = cnn_probabilities(device)
    for split in ("train", "dev", "test"):
        if len(cnn[split]) != len(trees[split]):
            raise RuntimeError(f"{split} CNN and Separatrix patient counts differ")
    dev_labels = labels["dev"]
    best = None
    trials = []
    for kind in ("probability", "logit"):
        for weight in WEIGHTS:
            mixed = blend(trees["dev"], cnn["dev"], float(weight), kind)
            threshold, utility = choose_threshold(dev_labels, mixed)
            row = {"kind": kind, "separatrix_weight": float(weight), "threshold": float(threshold), "dev_utility": float(utility)}
            trials.append(row)
            print(f"{kind} weight {weight:.2f}  dev utility {utility:.4f}  threshold {threshold:.3f}", flush=True)
            if best is None or utility > best["dev_utility"]:
                best = row
    print(
        f"selected {best['kind']} separatrix weight {best['separatrix_weight']:.2f} "
        f"threshold {best['threshold']:.3f}",
        flush=True,
    )
    scored = {}
    hospitals = {}
    for split in ("train", "dev", "test"):
        mixed = blend(trees[split], cnn[split], best["separatrix_weight"], best["kind"])
        overall = operating_metrics(labels[split], mixed, best["threshold"])
        overall["patients"] = len(labels[split])
        by_site = by_hospital(labels[split], mixed, hospital_names(split), best["threshold"])
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"ensemble {split}", overall, by_site)
    payload = {
        "model": "blend of regularized causal CNN and Separatrix trees",
        "selection": "mixing weight and alarm threshold chosen on dev utility",
        "selected": best,
        "dev_grid": trials,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "separatrix_tree_test_utility": 0.4545682369775407,
        "regularized_cnn_test_utility": 0.3850805973453087,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ensemble dev {scored['dev']['utility']:.4f} test {scored['test']['utility']:.4f}", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
