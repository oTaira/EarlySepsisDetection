"""Train the causal CNN baseline and score it with challenge utility.

The alarm threshold is chosen on the dev set. Test is scored once, with that
frozen threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import N_FEATURES, StayDataset, collate, load_label_sequences, prepare
from model import SimpleCausalCNN
from utility import best_predictions, choose_threshold, normalized_utility, self_check

SEED = 230
BATCH_SIZE = 128
MAX_EPOCHS = 10
PATIENCE = 3
LR = 1e-3
RUN = Path(__file__).resolve().parent / "runs" / "cnn_baseline"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def operating_metrics(label_seqs, probability_seqs, threshold: float) -> dict:
    y_parts = []
    p_parts = []
    tp = 0
    positive = 0
    false_alarms = 0
    nonseptic_hours = 0
    predictions = []
    for labels, probs in zip(label_seqs, probability_seqs):
        labels = np.asarray(labels).astype(np.int32)
        probs = np.asarray(probs, dtype=np.float64)
        pred = probs >= threshold
        predictions.append(pred)
        y_parts.append(labels)
        p_parts.append(probs)
        tp += int(np.sum((labels == 1) & pred))
        positive += int(np.sum(labels == 1))
        if not np.any(labels):
            false_alarms += int(np.sum(pred))
            nonseptic_hours += int(labels.shape[0])
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    return {
        "utility": float(normalized_utility(label_seqs, predictions)),
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "hour_sensitivity": float(tp / positive),
        "fp_per_100_nonseptic_hours": float(100.0 * false_alarms / nonseptic_hours),
    }


@torch.no_grad()
def predict(model, loader, device) -> list[np.ndarray]:
    model.eval()
    sequences = []
    for features, _labels, mask in loader:
        logits = model(features.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        valid = mask.numpy()
        for i in range(probs.shape[0]):
            n = int(valid[i].sum())
            sequences.append(probs[i, :n].astype(np.float64))
    return sequences


def train_epoch(model, loader, optimizer, pos_weight, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = int(mask.sum().item())
        total += float(loss.detach().item()) * n
        hours += n
    return total / hours


def main() -> None:
    set_seed(SEED)
    self_check()
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    dev_labels = load_label_sequences("dev")
    oracle = normalized_utility(dev_labels, [best_predictions(y) for y in dev_labels])
    inaction = normalized_utility(dev_labels, [np.zeros(len(y), dtype=bool) for y in dev_labels])
    print(f"dev scorer check: oracle {oracle:.6f}, no alarms {inaction:.6f}", flush=True)
    if abs(oracle - 1.0) > 1e-6 or abs(inaction) > 1e-6:
        raise RuntimeError("utility normalization failed on the dev set")

    train_set = StayDataset("train")
    dev_set = StayDataset("dev")
    test_set = StayDataset("test")
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, generator=generator)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = SimpleCausalCNN(N_FEATURES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([meta["pos_weight"]], device=device)
    print(
        f"CNN parameters {n_params}, receptive field {model.receptive_field}h, "
        f"positive-class weight {meta['pos_weight']:.2f}",
        flush=True,
    )

    RUN.mkdir(parents=True, exist_ok=True)
    best_utility = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        dev_probs = predict(model, dev_loader, device)
        threshold, utility = choose_threshold(dev_labels, dev_probs)
        history.append({"epoch": epoch, "train_loss": loss, "dev_utility": utility, "threshold": threshold})
        print(
            f"epoch {epoch:02d}  train loss {loss:.4f}  dev utility {utility:.4f}  threshold {threshold:.3f}",
            flush=True,
        )
        if utility > best_utility:
            best_utility = utility
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"stopping after epoch {epoch}; best dev utility was epoch {best_epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    dev_probs = predict(model, dev_loader, device)
    dev_threshold, _ = choose_threshold(dev_labels, dev_probs)
    dev_metrics = operating_metrics(dev_labels, dev_probs, dev_threshold)
    test_labels = load_label_sequences("test")
    test_probs = predict(model, test_loader, device)
    test_metrics = operating_metrics(test_labels, test_probs, dev_threshold)

    payload = {
        "model": "3-layer causal CNN, kernel 8, 32 channels",
        "seed": SEED,
        "parameters": n_params,
        "receptive_field_hours": model.receptive_field,
        "pos_weight": meta["pos_weight"],
        "best_epoch": best_epoch,
        "threshold_selected_on": "dev",
        "history": history,
        "dev": dev_metrics,
        "test": test_metrics,
    }
    torch.save({"state_dict": best_state, "threshold": dev_threshold, "meta": payload}, RUN / "model.pt")
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("", flush=True)
    print(f"Best epoch {best_epoch}. Threshold {dev_threshold:.3f} frozen on dev.", flush=True)
    print(
        f"Dev  utility {dev_metrics['utility']:.4f}  AUPRC {dev_metrics['auprc']:.4f}  "
        f"AUROC {dev_metrics['auroc']:.4f}  sensitivity {dev_metrics['hour_sensitivity']:.3f}  "
        f"FP/100 nonseptic hours {dev_metrics['fp_per_100_nonseptic_hours']:.2f}",
        flush=True,
    )
    print(
        f"Test utility {test_metrics['utility']:.4f}  AUPRC {test_metrics['auprc']:.4f}  "
        f"AUROC {test_metrics['auroc']:.4f}  sensitivity {test_metrics['hour_sensitivity']:.3f}  "
        f"FP/100 nonseptic hours {test_metrics['fp_per_100_nonseptic_hours']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
