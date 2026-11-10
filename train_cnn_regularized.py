"""Regularize the SWIn CNN on the original hourly inputs.

Smaller network, stronger dropout and weight decay, and Gaussian noise on
measured values during training only. The epoch count is fixed in advance.
The threshold is chosen on dev after training. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import N_FEATURES, N_PHYSIO, StayDataset, collate, load_label_sequences, prepare
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, predict, set_seed
from utility import choose_threshold

SEED = 246
EPOCHS = 10
NOISE_STD = 0.1
CONFIG = {
    "hidden": 64,
    "kernel_size": 9,
    "n_layers": 3,
    "dropout": 0.4,
    "lr": 0.00022868030122031387,
    "weight_decay": 1e-3,
    "batch_size": 128,
    "swin_a": 0.05,
    "swin_omega": 0.5,
    "noise_std": NOISE_STD,
}
RUN = Path(__file__).resolve().parent / "runs" / "cnn_regularized"


def augment(features: torch.Tensor) -> torch.Tensor:
    """Add noise to z-scored values that were measured at that hour."""
    observed = features[:, N_PHYSIO : 2 * N_PHYSIO]
    noise = torch.randn_like(features[:, :N_PHYSIO]) * NOISE_STD
    out = features.clone()
    out[:, :N_PHYSIO] = features[:, :N_PHYSIO] + noise * observed
    return out


def train_epoch(model, loader, optimizer, pos_weight, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = augment(features.to(device))
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
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}. Fixed {EPOCHS} epochs, no dev-based checkpointing.", flush=True)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    batch = CONFIG["batch_size"]
    train_loader = DataLoader(
        StayDataset("train"), batch_size=batch, shuffle=True, collate_fn=collate, generator=generator
    )
    dev_loader = DataLoader(StayDataset("dev"), batch_size=batch, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(StayDataset("test"), batch_size=batch, shuffle=False, collate_fn=collate)
    ordered_train = DataLoader(StayDataset("train"), batch_size=batch, shuffle=False, collate_fn=collate)
    model = SimpleCausalCNN(
        N_FEATURES,
        hidden=CONFIG["hidden"],
        kernel_size=CONFIG["kernel_size"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
        activation=SWInSiLU(CONFIG["swin_a"], CONFIG["swin_omega"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    pos_weight = torch.tensor([meta["pos_weight"]], device=device)
    dev_labels = load_label_sequences("dev")
    history = []
    for epoch in range(1, EPOCHS + 1):
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

    final = history[-1]
    threshold = final["threshold"]
    dev_probs = predict(model, dev_loader, device)
    dev_metrics = operating_metrics(dev_labels, dev_probs, threshold)
    train_metrics = operating_metrics(load_label_sequences("train"), predict(model, ordered_train, device), threshold)
    test_metrics = operating_metrics(load_label_sequences("test"), predict(model, test_loader, device), threshold)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {
        "model": "Regularized SWIn-SiLU causal CNN on hourly inputs",
        "config": CONFIG,
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint": "last epoch",
        "parameters": sum(p.numel() for p in model.parameters()),
        "n_features": N_FEATURES,
        "receptive_field_hours": model.receptive_field,
        "history": history,
        "train": train_metrics,
        "dev": dev_metrics,
        "test": test_metrics,
        "previous_swin_test_utility": 0.3863257255342203,
        "window_cnn_test_utility": 0.3684946993114249,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": CONFIG, "threshold": threshold}, RUN / "model.pt")
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("", flush=True)
    print(f"Kept epoch {EPOCHS}. Threshold {threshold:.3f} frozen on dev.", flush=True)
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
