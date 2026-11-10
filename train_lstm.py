"""Train a causal LSTM with the same regularization as the SWIn CNN.

Dropout, weight decay, and noise on measured values. The epoch count is fixed.
The threshold is chosen on dev after training. Test is scored once, overall
and inside each hospital.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    N_FEATURES,
    N_PHYSIO,
    StayDataset,
    collate,
    load_label_sequences,
    manifest_rows,
    prepare,
)
from model import SWInSiLU, CausalLSTM, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, predict, set_seed
from utility import choose_threshold

SEED = 230
EPOCHS = 10
NOISE_STD = 0.1
CONFIG = {
    "hidden": 64,
    "n_layers": 2,
    "dropout": 0.4,
    "lr": 0.00022868030122031387,
    "weight_decay": 1e-3,
    "batch_size": 128,
    "noise_std": NOISE_STD,
}
RUN = Path(__file__).resolve().parent / "runs" / "lstm"
CNN_RUN = Path(__file__).resolve().parent / "runs" / "cnn_regularized" / "model.pt"


def augment(features: torch.Tensor) -> torch.Tensor:
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


def hospital_names(split: str) -> list[str]:
    names = [row["hospital"] for row in manifest_rows(split)]
    return names


def by_hospital(label_seqs, probability_seqs, hospitals, threshold: float) -> dict:
    if len(label_seqs) != len(hospitals):
        raise RuntimeError("hospital list does not match the patient order")
    report = {}
    for name in sorted(set(hospitals)):
        chosen = [hospital == name for hospital in hospitals]
        labels = [seq for seq, keep in zip(label_seqs, chosen) if keep]
        probs = [seq for seq, keep in zip(probability_seqs, chosen) if keep]
        metrics = operating_metrics(labels, probs, threshold)
        metrics["patients"] = len(labels)
        report[name] = metrics
    return report


def score_split(model, split: str, threshold: float, device) -> tuple[dict, dict]:
    loader = DataLoader(StayDataset(split), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate)
    labels = load_label_sequences(split)
    probs = predict(model, loader, device)
    overall = operating_metrics(labels, probs, threshold)
    overall["patients"] = len(labels)
    hospitals = by_hospital(labels, probs, hospital_names(split), threshold)
    return overall, hospitals


def print_scores(title: str, overall: dict, hospitals: dict) -> None:
    print(f"\n{title}", flush=True)
    print(f"{'group':<8}{'patients':>10}{'utility':>10}{'AUPRC':>10}{'AUROC':>10}{'sens':>8}{'FP/100':>10}", flush=True)
    rows = [("all", overall)] + [(name, hospitals[name]) for name in hospitals]
    for name, row in rows:
        patients = row.get("patients", "")
        patient_text = f"{patients:10d}" if isinstance(patients, int) else f"{'':>10}"
        print(
            f"{name:<8}{patient_text}{row['utility']:10.4f}{row['auprc']:10.4f}{row['auroc']:10.4f}"
            f"{row['hour_sensitivity']:8.3f}{row['fp_per_100_nonseptic_hours']:10.2f}",
            flush=True,
        )


def regularized_cnn_test(device) -> dict | None:
    if not CNN_RUN.exists():
        return None
    saved = torch.load(CNN_RUN, map_location=device, weights_only=False)
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
    overall, hospitals = score_split(model, "test", float(saved["threshold"]), device)
    print_scores("Regularized CNN test, same threshold as its own run", overall, hospitals)
    return {"test": overall, "hospitals": hospitals}


def main() -> None:
    meta = prepare()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}. Fixed {EPOCHS} epochs.", flush=True)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    batch = CONFIG["batch_size"]
    train_loader = DataLoader(
        StayDataset("train"), batch_size=batch, shuffle=True, collate_fn=collate, generator=generator
    )
    dev_loader = DataLoader(StayDataset("dev"), batch_size=batch, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(StayDataset("test"), batch_size=batch, shuffle=False, collate_fn=collate)
    model = CausalLSTM(N_FEATURES, CONFIG["hidden"], CONFIG["n_layers"], CONFIG["dropout"]).to(device)
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

    threshold = history[-1]["threshold"]
    results = {}
    hospital_results = {}
    for split in ("train", "dev", "test"):
        overall, hospitals = score_split(model, split, threshold, device)
        results[split] = overall
        hospital_results[split] = hospitals
        print_scores(f"LSTM {split}", overall, hospitals)

    cnn_hospitals = regularized_cnn_test(device)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {
        "model": "2-layer causal LSTM",
        "config": CONFIG,
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint": "last epoch",
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "train": results["train"],
        "dev": results["dev"],
        "test": results["test"],
        "hospitals": hospital_results,
        "regularized_cnn_test_by_hospital": cnn_hospitals,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": CONFIG, "threshold": threshold}, RUN / "model.pt")
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nKept epoch {EPOCHS}. Threshold {threshold:.3f} frozen on dev.", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
