"""Blend the best causal model of each architecture.

The five pieces are the regularized CNN, the searched LSTM, the searched
attention model, the clinical graph network, and the Separatrix trees with
lab windows. Missing values hold the last past measurement. Weights and the
alarm threshold are chosen on dev. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from compare_graphs import clinical_edges
from dataset import N_FEATURES, StayDataset, collate, prepare
from model import SWInSiLU, SimpleCausalCNN
from search_cnn_broad import predict_ordered
from search_gat import build_model as build_gat
from search_gat import predict_ordered as predict_gat
from search_sequence import build_model as build_sequence
from train_cnn_baseline import operating_metrics, predict
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_causal import causal_labs, causal_separatrix, moments_pair
from train_separatrix_labs import fit, predict_split
from train_winners import flat_labels
from utility import best_predictions, choose_threshold, hour_utility
from winner_data import labels_of, load_raw, sequences, training_means

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "ensemble_causal"
TREES = ROOT / "runs" / "separatrix_causal"
NAMES = ("cnn", "lstm", "attention", "gat", "trees")


SPLITS = ("dev", "test")


def model_cache(name: str) -> Path:
    return RUN / f"{name}_probabilities.npz"


def save_one(name: str, scores: dict[str, list[np.ndarray]]) -> None:
    payload = {}
    for split, sequences_ in scores.items():
        payload[split] = np.concatenate(sequences_).astype(np.float32)
        payload[f"{split}_len"] = np.asarray([len(row) for row in sequences_], dtype=np.int32)
    RUN.mkdir(parents=True, exist_ok=True)
    path = model_cache(name)
    np.savez_compressed(path, **payload)
    print(f"cached {path}", flush=True)


def load_one(name: str) -> dict[str, list[np.ndarray] | np.ndarray]:
    stored = np.load(model_cache(name))
    output = {}
    for split in SPLITS:
        flat = stored[split].astype(np.float64)
        lengths = stored[f"{split}_len"]
        rows = []
        start = 0
        for length in lengths:
            rows.append(flat[start : start + int(length)])
            start += int(length)
        output[split] = rows
        output[f"{split}_flat"] = flat
    print(f"loaded {name}", flush=True)
    return output


def utility_tables(labels):
    positive = []
    negative = []
    inaction = 0.0
    oracle = 0.0
    for row in labels:
        gain, quiet = hour_utility(row)
        positive.append(gain)
        negative.append(quiet)
        inaction += float(quiet.sum())
        oracle += float(np.where(best_predictions(row), gain, quiet).sum())
    denom = oracle - inaction
    return np.concatenate(positive), np.concatenate(negative), inaction, denom


def best_cutoff(flat: np.ndarray, positive: np.ndarray, negative: np.ndarray, inaction: float, denom: float):
    thresholds = np.linspace(0.0, 1.0, 201)
    alarm = flat[None, :] >= thresholds[:, None]
    totals = np.where(alarm, positive, negative).sum(axis=1)
    scores = (totals - inaction) / denom
    index = int(len(scores) - 1 - np.argmax(scores[::-1]))
    return float(thresholds[index]), float(scores[index])


def blend_flat(columns: list[np.ndarray], weights: np.ndarray, kind: str) -> np.ndarray:
    acc = None
    for weight, column in zip(weights, columns):
        if weight == 0:
            continue
        values = np.clip(column, 1e-6, 1.0 - 1e-6)
        term = weight * values if kind == "probability" else weight * np.log(values / (1.0 - values))
        acc = term if acc is None else acc + term
    if kind == "logit":
        acc = 1.0 / (1.0 + np.exp(-acc))
    return acc


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
    equal = np.full(len(NAMES), 1.0 / len(NAMES))
    weights.append(equal)
    for _ in range(count):
        weights.append(rng.dirichlet(np.ones(len(NAMES))))
    return weights


def load_xgb(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def tree_models(means):
    ready = TREES / "ready.json"
    if ready.exists():
        norm = np.load(TREES / "norm.npz")
        models = [load_xgb(TREES / f"fold_{index}.json") for index in range(5)]
        print(f"loaded causal trees, {int(norm['keep'].size)} features", flush=True)
        return models, norm["keep"].astype(int), norm["mu"], norm["sd"]
    separatrix = causal_separatrix("train", means)
    labs = causal_labs("train", means)
    mu, sd = moments_pair(separatrix, labs)
    models, keep, _gain = fit(separatrix, labs, mu, sd, flat_labels("train"), load_raw("train")[2])
    TREES.mkdir(parents=True, exist_ok=True)
    for index, model in enumerate(models):
        model.save_model(TREES / f"fold_{index}.json")
    np.savez(TREES / "norm.npz", mu=mu, sd=sd, keep=np.asarray(keep, dtype=np.int32))
    ready.write_text(json.dumps({"kept_columns": len(keep)}, indent=2), encoding="utf-8")
    print("saved causal trees", flush=True)
    return models, np.asarray(keep, dtype=int), mu, sd


def tree_scores(means) -> dict[str, list[np.ndarray]]:
    models, keep, mu, sd = tree_models(means)
    output = {}
    for split in SPLITS:
        separatrix = causal_separatrix(split, means)
        labs = causal_labs(split, means)
        _, _, offsets = load_raw(split)
        output[split] = sequences(predict_split(models, keep, mu, sd, separatrix, labs), offsets)
        print(f"trees {split}", flush=True)
    return output


def cnn_scores(device) -> dict[str, list[np.ndarray]]:
    saved = torch.load(ROOT / "runs" / "cnn_regularized" / "model.pt", map_location=device, weights_only=False)
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
    for split in SPLITS:
        loader = DataLoader(StayDataset(split), batch_size=128, shuffle=False, collate_fn=collate)
        output[split] = predict(model, loader, device)
        print(f"cnn {split}", flush=True)
    del model
    return output


def sequence_scores(family: str, device) -> dict[str, list[np.ndarray]]:
    saved = torch.load(ROOT / "runs" / "sequence_search" / f"best_{family}.pt", map_location=device, weights_only=False)
    model = build_sequence(saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    output = {}
    for split in SPLITS:
        output[split] = predict_ordered(model, StayDataset(split), 128, device)
        print(f"{family} {split}", flush=True)
    del model
    return output


def gat_scores(device) -> dict[str, list[np.ndarray]]:
    saved = torch.load(
        ROOT / "runs" / "graph_compare" / "all_hospitals_clinical.pt", map_location=device, weights_only=False
    )
    model = build_gat(saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    edges = clinical_edges()
    output = {}
    for split in SPLITS:
        print(f"gat {split} start", flush=True)
        output[split] = predict_gat(model, StayDataset(split), 128, edges, device)
        print(f"gat {split}", flush=True)
    del model
    return output


def main() -> None:
    torch.set_float32_matmul_precision("high")
    prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    means = training_means()
    scorers = {
        "trees": lambda: tree_scores(means),
        "cnn": lambda: cnn_scores(device),
        "lstm": lambda: sequence_scores("lstm", device),
        "attention": lambda: sequence_scores("attention", device),
        "gat": lambda: gat_scores(device),
    }
    parts = {}
    for name in NAMES:
        if model_cache(name).exists():
            parts[name] = load_one(name)
            continue
        parts[name] = scorers[name]()
        save_one(name, parts[name])
        if device.type == "cuda":
            torch.cuda.empty_cache()
        parts[name] = load_one(name)
    labels = {split: labels_of(split) for split in SPLITS}
    for name in NAMES:
        for split in labels:
            if len(parts[name][split]) != len(labels[split]):
                raise RuntimeError(f"{name} {split} does not match the stays")
    positive, negative, inaction, denom = utility_tables(labels["dev"])
    probe = np.full(len(NAMES), 1.0 / len(NAMES))
    probe_flat = blend_flat([parts[name]["dev_flat"] for name in NAMES], probe, "probability")
    fast_threshold, fast_utility = best_cutoff(probe_flat, positive, negative, inaction, denom)
    slow_threshold, slow_utility = choose_threshold(
        labels["dev"], blend({name: parts[name]["dev"] for name in NAMES}, probe, "probability")
    )
    if abs(fast_utility - slow_utility) > 1e-6 or fast_threshold != slow_threshold:
        raise RuntimeError(f"blend score {fast_utility} at {fast_threshold} disagrees with {slow_utility} at {slow_threshold}")
    rng = np.random.default_rng(230)
    best = None
    checked = 0
    for kind in ("probability", "logit"):
        for weights in candidates(rng, 1200):
            flat = blend_flat([parts[name]["dev_flat"] for name in NAMES], weights, kind)
            threshold, utility = best_cutoff(flat, positive, negative, inaction, denom)
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
            elif checked % 400 == 0:
                print(f"checked {checked}", flush=True)
    print(f"checked {checked} blends. selected {json.dumps(best)}", flush=True)
    chosen = np.array([best["weights"][name] for name in NAMES])
    scored = {}
    hospitals = {}
    for split in SPLITS:
        mixed = blend({name: parts[name][split] for name in NAMES}, chosen, best["kind"])
        overall = operating_metrics(labels[split], mixed, best["threshold"])
        overall["patients"] = len(labels[split])
        sites = by_hospital(labels[split], mixed, hospital_names(split), best["threshold"])
        scored[split] = overall
        hospitals[split] = sites
        print_scores(f"causal ensemble {split}", overall, sites)
    payload = {
        "models": list(NAMES),
        "imputation": "last past value; training mean or median before the first measurement",
        "selection": "simplex weights and alarm threshold chosen on dev utility",
        "blends_checked": checked,
        "selected": best,
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"causal ensemble dev {scored['dev']['utility']:.4f} test {scored['test']['utility']:.4f}", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
