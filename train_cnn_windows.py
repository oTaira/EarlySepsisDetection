"""Retrain the winning SWIn CNN on the causal window features.

The architecture and optimizer match search trial 16. Only the inputs change.
The threshold is chosen on dev. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import collate, load_label_sequences, prepare
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import MAX_EPOCHS, PATIENCE, operating_metrics, predict, set_seed, train_epoch
from utility import choose_threshold
from window_features import N_FEATURES_WINDOWS, WindowStayDataset, build

SEED = 246  # same seed as search trial 16
CONFIG = {
    "hidden": 96,
    "kernel_size": 9,
    "n_layers": 5,
    "dropout": 0.149,
    "lr": 0.00022868030122031387,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "swin_a": 0.05,
    "swin_omega": 0.5,
}
RUN = Path(__file__).resolve().parent / "runs" / "cnn_windows"


@torch.no_grad()
def split_loss(model, loader, pos_weight, device) -> float:
    model.eval()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pos_weight)
        n = int(mask.sum().item())
        total += float(loss.item()) * n
        hours += n
    return total / hours


def main() -> None:
    meta = prepare()
    build()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}, features {N_FEATURES_WINDOWS}", flush=True)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    batch = CONFIG["batch_size"]
    train_loader = DataLoader(
        WindowStayDataset("train"), batch_size=batch, shuffle=True, collate_fn=collate, generator=generator
    )
    dev_loader = DataLoader(WindowStayDataset("dev"), batch_size=batch, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(WindowStayDataset("test"), batch_size=batch, shuffle=False, collate_fn=collate)
    model = SimpleCausalCNN(
        N_FEATURES_WINDOWS,
        hidden=CONFIG["hidden"],
        kernel_size=CONFIG["kernel_size"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
        activation=SWInSiLU(CONFIG["swin_a"], CONFIG["swin_omega"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    pos_weight = torch.tensor([meta["pos_weight"]], device=device)
    dev_labels = load_label_sequences("dev")
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        dev_loss = split_loss(model, dev_loader, pos_weight, device)
        test_loss = split_loss(model, test_loader, pos_weight, device)
        dev_probs = predict(model, dev_loader, device)
        threshold, utility = choose_threshold(dev_labels, dev_probs)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "dev_loss": dev_loss,
                "test_loss": test_loss,
                "dev_utility": utility,
                "threshold": threshold,
            }
        )
        print(
            f"epoch {epoch:02d}  train loss {loss:.4f}  dev loss {dev_loss:.4f}  "
            f"test loss {test_loss:.4f}  dev utility {utility:.4f}",
            flush=True,
        )
        if utility > best_utility:
            best_utility = utility
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_dev_probs = dev_probs
            best_threshold = threshold
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"stopping after epoch {epoch}; best dev utility was epoch {best_epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    dev_metrics = operating_metrics(dev_labels, best_dev_probs, best_threshold)
    train_labels = load_label_sequences("train")
    ordered_train = DataLoader(WindowStayDataset("train"), batch_size=batch, shuffle=False, collate_fn=collate)
    train_probs = predict(model, ordered_train, device)
    train_metrics = operating_metrics(train_labels, train_probs, best_threshold)
    test_labels = load_label_sequences("test")
    test_metrics = operating_metrics(test_labels, predict(model, test_loader, device), best_threshold)
    payload = {
        "model": "SWIn-SiLU causal CNN with 6h and 24h window summaries",
        "config": CONFIG,
        "seed": SEED,
        "parameters": sum(p.numel() for p in model.parameters()),
        "n_features": N_FEATURES_WINDOWS,
        "receptive_field_hours": model.receptive_field,
        "best_epoch": best_epoch,
        "history": history,
        "train": train_metrics,
        "dev": dev_metrics,
        "test": test_metrics,
        "previous_swin_test_utility": 0.3863257255342203,
        "relu_baseline_test_utility": 0.3692955341402302,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": CONFIG, "threshold": best_threshold}, RUN / "model.pt")
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("", flush=True)
    print(f"Best epoch {best_epoch}. Threshold {best_threshold:.3f} frozen on dev.", flush=True)
    for name, row in ("train", train_metrics), ("dev", dev_metrics), ("test", test_metrics):
        print(
            f"{name:<5} utility {row['utility']:.4f}  AUPRC {row['auprc']:.4f}  "
            f"AUROC {row['auroc']:.4f}  sensitivity {row['hour_sensitivity']:.3f}  "
            f"FP/100 {row['fp_per_100_nonseptic_hours']:.2f}",
            flush=True,
        )
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
