"""Reload the packaged models and score them at the frozen dev thresholds.

The first run copies the neural checkpoints into packaged/ and saves the
XGBoost models used by the ensemble. Later runs only score.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from compare_graphs import clinical_edges
from dataset import N_FEATURES, StayDataset, collate, load_label_sequences, prepare
from ensemble_all import NAMES, blend
from ensemble_cnn_separatrix import fit_separatrix, separatrix_probabilities, standardize
from model import SWInSiLU, SimpleCausalCNN
from search_cnn_broad import predict_ordered
from search_gat import build_model as build_gat
from search_gat import predict_ordered as predict_gat
from search_sequence import build_model as build_sequence
from train_cnn_baseline import operating_metrics, predict
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
from winner_data import labels_of, load_raw, sequences, training_means

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "packaged"
RUNS = ROOT / "runs"
SOURCES = {
    "cnn": RUNS / "cnn_regularized" / "model.pt",
    "lstm": RUNS / "sequence_search" / "best_lstm.pt",
    "attention": RUNS / "sequence_search" / "best_attention.pt",
    "gat": RUNS / "graph_compare" / "all_hospitals_clinical.pt",
}
EXPECTED = {
    "cnn": 0.3850805973453087,
    "lstm": 0.3752571288529807,
    "attention": 0.3706236708750689,
    "gat": 0.30048777402702287,
    "ensemble": 0.46605503861873826,
}
ENSEMBLE = {
    "kind": "probability",
    "threshold": 0.445,
    "weights": {
        "cnn": 0.023370604878074876,
        "separatrix": 0.5599555963072756,
        "sepsyd": 0.24727133929320483,
        "ctl": 0.022974805256188155,
        "lstm": 0.1461278684840972,
        "attention": 0.00029978578115927345,
    },
}


def copy_checkpoints() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        destination = PACK / f"{name}.pt"
        if not destination.exists():
            shutil.copy2(source, destination)
            print(f"copied {name}", flush=True)
    gat_path = PACK / "gat.pt"
    saved = torch.load(gat_path, map_location="cpu", weights_only=False)
    if "threshold" not in saved:
        saved["threshold"] = 0.635
        saved["graph"] = "clinical"
        torch.save(saved, gat_path)


def save_trees() -> None:
    tree_dir = PACK / "trees"
    if (tree_dir / "ready.json").exists():
        return
    tree_dir.mkdir(parents=True, exist_ok=True)
    means = training_means()
    models, keep, mu, sd = fit_separatrix(means)
    for index, model in enumerate(models):
        model.save_model(tree_dir / f"separatrix_{index}.json")
    np.savez(tree_dir / "separatrix_norm.npz", mu=mu, sd=sd, keep=np.asarray(keep, dtype=np.int32))

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
    model.save_model(tree_dir / "sepsyd.json")
    np.savez(tree_dir / "sepsyd_prep.npz", center=center, skewed=np.asarray(sorted(skewed), dtype=np.int32))

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
    model.save_model(tree_dir / "ctl.json")
    (tree_dir / "ready.json").write_text(json.dumps({"separatrix_features": len(keep)}, indent=2), encoding="utf-8")
    print("saved tree models", flush=True)


def load_xgb(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def tree_probabilities(device_unused=None) -> dict[str, dict[str, list[np.ndarray]]]:
    del device_unused
    tree_dir = PACK / "trees"
    norm = np.load(tree_dir / "separatrix_norm.npz")
    mu, sd, keep = norm["mu"], norm["sd"], norm["keep"].astype(int)
    separatrix_models = [load_xgb(tree_dir / f"separatrix_{index}.json") for index in range(5)]
    prep = np.load(tree_dir / "sepsyd_prep.npz")
    sepsyd = load_xgb(tree_dir / "sepsyd.json")
    ctl = load_xgb(tree_dir / "ctl.json")
    means = training_means()
    output = {"separatrix": {}, "sepsyd": {}, "ctl": {}}
    for split in ("dev", "test"):
        _, _, offsets = load_raw(split)
        matrix = standardize(cached_matrix(split, means), mu, sd)[:, keep]
        output["separatrix"][split] = sequences(separatrix_probabilities(separatrix_models, matrix), offsets)
        sepsyd_matrix = build_sepsyd(split, prep["center"], prep["skewed"].tolist())
        output["sepsyd"][split] = sequences(sepsyd.predict_proba(sepsyd_matrix)[:, 1], offsets)
        output["ctl"][split] = sequences(ctl.predict_proba(build_ctl(split))[:, 1], offsets)
        print(f"trees {split}", flush=True)
    return output


def neural_probabilities(device) -> dict[str, dict[str, list[np.ndarray]]]:
    output = {}
    cnn_saved = torch.load(PACK / "cnn.pt", map_location=device, weights_only=False)
    config = cnn_saved["config"]
    cnn = SimpleCausalCNN(
        N_FEATURES,
        hidden=config["hidden"],
        kernel_size=config["kernel_size"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        activation=SWInSiLU(config["swin_a"], config["swin_omega"]),
    ).to(device)
    cnn.load_state_dict(cnn_saved["state_dict"])
    output["cnn"] = {}
    for split in ("dev", "test"):
        loader = DataLoader(StayDataset(split), batch_size=128, shuffle=False, collate_fn=collate)
        output["cnn"][split] = predict(cnn, loader, device)
    del cnn

    for name in ("lstm", "attention"):
        saved = torch.load(PACK / f"{name}.pt", map_location=device, weights_only=False)
        model = build_sequence(saved["config"]).to(device)
        model.load_state_dict(saved["state_dict"])
        output[name] = {
            split: predict_ordered(model, StayDataset(split), 128, device) for split in ("dev", "test")
        }
        del model

    saved = torch.load(PACK / "gat.pt", map_location=device, weights_only=False)
    gat = build_gat(saved["config"]).to(device)
    gat.load_state_dict(saved["state_dict"])
    edges = clinical_edges()
    output["gat"] = {
        split: predict_gat(gat, StayDataset(split), 128, edges, device) for split in ("dev", "test")
    }
    del gat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def thresholds() -> dict[str, float]:
    values = {}
    cnn = torch.load(PACK / "cnn.pt", map_location="cpu", weights_only=False)
    values["cnn"] = float(cnn["threshold"])
    for name in ("lstm", "attention"):
        saved = torch.load(PACK / f"{name}.pt", map_location="cpu", weights_only=False)
        values[name] = float(saved["dev"]["threshold"])
    gat = torch.load(PACK / "gat.pt", map_location="cpu", weights_only=False)
    values["gat"] = float(gat["threshold"])
    values["ensemble"] = float(ENSEMBLE["threshold"])
    return values


def score_sequences(name, probabilities, threshold) -> dict:
    report = {}
    for split in ("dev", "test"):
        labels = load_label_sequences(split)
        overall = operating_metrics(labels, probabilities[split], threshold)
        overall["patients"] = len(labels)
        sites = by_hospital(labels, probabilities[split], hospital_names(split), threshold)
        report[split] = overall
        report[f"{split}_hospitals"] = sites
        print_scores(f"{name} {split}", overall, sites)
    return report


def main() -> None:
    torch.set_float32_matmul_precision("high")
    prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    copy_checkpoints()
    save_trees()
    frozen = thresholds()
    neural = neural_probabilities(device)
    trees = tree_probabilities()
    parts = {**neural, **trees}
    report = {}
    for name in ("cnn", "lstm", "attention", "gat"):
        report[name] = score_sequences(name, neural[name], frozen[name])
    weights = np.array([ENSEMBLE["weights"][name] for name in NAMES])
    mixed = {
        split: blend({name: parts[name][split] for name in NAMES}, weights, ENSEMBLE["kind"])
        for split in ("dev", "test")
    }
    report["ensemble"] = score_sequences("ensemble", mixed, frozen["ensemble"])
    for name, expected in EXPECTED.items():
        actual = report[name]["test"]["utility"]
        if abs(actual - expected) > 1e-4:
            raise RuntimeError(f"{name} test utility {actual:.6f} does not match {expected:.6f}")
    payload = {"thresholds": frozen, "ensemble": ENSEMBLE, "scores": report}
    (PACK / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("packaged scores match the saved test utilities", flush=True)
    print(f"wrote {PACK / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
