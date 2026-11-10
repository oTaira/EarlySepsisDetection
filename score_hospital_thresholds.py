"""Choose a separate alarm threshold for each hospital on the saved CNN.

Hospital is known at prediction time. Thresholds are chosen on dev and frozen
for test. The cohort utility uses each stay's own hospital threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import N_FEATURES, StayDataset, collate, load_label_sequences
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, predict
from train_lstm import hospital_names, print_scores
from utility import choose_threshold, normalized_utility

RUN = Path(__file__).resolve().parent / "runs" / "cnn_regularized"


def predictions_for(probabilities, hospitals, thresholds) -> list[np.ndarray]:
    return [np.asarray(probs) >= thresholds[hospital] for probs, hospital in zip(probabilities, hospitals)]


def operating_from_predictions(labels, probabilities, predictions) -> dict:
    y_parts = []
    p_parts = []
    tp = 0
    positive = 0
    false_alarms = 0
    nonseptic_hours = 0
    for row_labels, probs, pred in zip(labels, probabilities, predictions):
        row_labels = np.asarray(row_labels).astype(np.int32)
        probs = np.asarray(probs, dtype=np.float64)
        pred = np.asarray(pred).astype(bool)
        y_parts.append(row_labels)
        p_parts.append(probs)
        tp += int(np.sum((row_labels == 1) & pred))
        positive += int(np.sum(row_labels == 1))
        if not np.any(row_labels):
            false_alarms += int(np.sum(pred))
            nonseptic_hours += int(row_labels.shape[0])
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "utility": float(normalized_utility(labels, predictions)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "hour_sensitivity": float(tp / positive),
        "fp_per_100_nonseptic_hours": float(100.0 * false_alarms / nonseptic_hours),
        "patients": len(labels),
    }


def subset(labels, probabilities, hospitals, name):
    chosen = [hospital == name for hospital in hospitals]
    return (
        [row for row, keep in zip(labels, chosen) if keep],
        [row for row, keep in zip(probabilities, chosen) if keep],
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(RUN / "model.pt", map_location=device, weights_only=False)
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
    shared = float(saved["threshold"])
    scored = {}
    hospitals = {}
    probabilities = {}
    for split in ("dev", "test"):
        loader = DataLoader(StayDataset(split), batch_size=128, shuffle=False, collate_fn=collate)
        labels = load_label_sequences(split)
        probabilities[split] = predict(model, loader, device)
        hospitals[split] = hospital_names(split)
        scored[split] = {"labels": labels}
        print(f"scored {split}", flush=True)

    thresholds = {}
    dev_utility = {}
    for name in ("A", "B"):
        labels, probs = subset(scored["dev"]["labels"], probabilities["dev"], hospitals["dev"], name)
        threshold, utility = choose_threshold(labels, probs)
        thresholds[name] = threshold
        dev_utility[name] = utility
        print(f"hospital {name} dev threshold {threshold:.3f} utility {utility:.4f}", flush=True)

    payload = {"shared_threshold": shared, "hospital_thresholds": thresholds, "dev_utility_at_own_threshold": dev_utility}
    for split in ("dev", "test"):
        labels = scored[split]["labels"]
        probs = probabilities[split]
        sites = hospitals[split]
        shared_metrics = operating_metrics(labels, probs, shared)
        shared_metrics["patients"] = len(labels)
        own_predictions = predictions_for(probs, sites, thresholds)
        own = operating_from_predictions(labels, probs, own_predictions)
        by_site = {}
        for name in ("A", "B"):
            site_labels, site_probs = subset(labels, probs, sites, name)
            site_predictions = predictions_for(site_probs, [name] * len(site_labels), thresholds)
            metrics = operating_from_predictions(site_labels, site_probs, site_predictions)
            by_site[name] = metrics
        payload[split] = {"shared_threshold": shared_metrics, "per_hospital_thresholds": own, "hospitals": by_site}
        print_scores(f"{split} shared threshold {shared:.3f}", shared_metrics, {
            name: operating_metrics(*subset(labels, probs, sites, name), shared) | {"patients": sum(h == name for h in sites)}
            for name in ("A", "B")
        })
        print_scores(f"{split} per-hospital thresholds", own, by_site)
    (RUN / "hospital_thresholds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {RUN / 'hospital_thresholds.json'}", flush=True)


if __name__ == "__main__":
    main()
