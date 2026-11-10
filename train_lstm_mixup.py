"""Train the searched LSTM with a small septic mixup at several strengths.

A normal stay is blended with a septic stay. The septic fraction is the
mix weight, and the septic stay's labels are kept. The original batch is
always trained on as well. Weighted BCE is the loss. The epoch with the
highest dev utility is kept for each mix weight. Test is scored after the
dev comparison, once per trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import N_FEATURES, StayDataset, collate, load_label_sequences, prepare
from model import CausalLSTM
from search_cnn_broad import LengthBatchSampler, predict_ordered
from search_sequence import augment
from train_cnn_baseline import operating_metrics, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold

RUN = Path(__file__).resolve().parent / "runs" / "lstm_mixup"
EPOCHS = 10
SEED = 248
RATIOS = (0.0, 0.05, 0.10, 0.20, 0.30)
CONFIG = {
    "hidden": 64,
    "n_layers": 3,
    "dropout": 0.282,
    "lr": 0.0006441257389397937,
    "weight_decay": 0.0001,
    "batch_size": 64,
    "noise_std": 0.1,
    "schedule": "cosine",
}


def mixed_examples(features, labels, mask, alpha: float, copies: int):
    """Extra stays: mostly the septic stay, with a normal stay blended in.

    Labels are copied from the septic stay. `copies` draws a new normal
    partner for each septic stay, so the batch gains that many examples.
    """
    if alpha == 0 or copies == 0:
        return None
    septic = ((labels > 0.5) & mask).any(dim=1)
    origins = septic.nonzero(as_tuple=False).flatten()
    partners = (~septic).nonzero(as_tuple=False).flatten()
    if origins.numel() == 0 or partners.numel() == 0:
        return None
    origins = origins.repeat(copies)
    chosen = partners[torch.randint(0, partners.numel(), (origins.numel(),), device=features.device)]
    pieces = []
    targets = []
    lengths = []
    for origin, partner in zip(origins.tolist(), chosen.tolist()):
        span = min(int(mask[origin].sum()), int(mask[partner].sum()))
        if span == 0:
            continue
        pieces.append((1.0 - alpha) * features[partner, :, :span] + alpha * features[origin, :, :span])
        targets.append(labels[origin, :span])
        lengths.append(span)
    if not pieces:
        return None
    longest = max(lengths)
    mixed = features.new_zeros(len(pieces), features.shape[1], longest)
    mixed_labels = labels.new_zeros(len(pieces), longest)
    mixed_mask = torch.zeros(len(pieces), longest, dtype=torch.bool, device=features.device)
    for index, (piece, target, span) in enumerate(zip(pieces, targets, lengths)):
        mixed[index, :, :span] = piece
        mixed_labels[index, :span] = target
        mixed_mask[index, :span] = True
    return mixed, mixed_labels, mixed_mask


def train_epoch(model, loader, optimizer, pos_weight, alpha: float, copies: int, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = augment(features.to(device, non_blocking=True), CONFIG["noise_std"])
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(features)[mask], labels[mask], pos_weight=pos_weight, reduction="sum")
        count = int(mask.sum().item())
        mixed = mixed_examples(features, labels, mask, alpha, copies)
        if mixed is not None:
            mixed_features, mixed_labels, mixed_mask = mixed
            extra = F.binary_cross_entropy_with_logits(
                model(mixed_features)[mixed_mask], mixed_labels[mixed_mask], pos_weight=pos_weight, reduction="sum"
            )
            loss = loss + extra
            count += int(mixed_mask.sum().item())
        (loss / count).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().item())
        hours += count
    return total / max(hours, 1)


@torch.no_grad()
def dev_loss(model, dataset, pos_weight, device) -> float:
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate, pin_memory=device.type == "cuda")
    model.eval()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        loss = F.binary_cross_entropy_with_logits(model(features)[mask], labels[mask], pos_weight=pos_weight)
        count = int(mask.sum().item())
        total += float(loss.item()) * count
        hours += count
    return total / max(hours, 1)


def score(model, splits, threshold, device):
    scored = {}
    hospitals = {}
    for split, dataset in splits.items():
        probabilities = predict_ordered(model, dataset, 128, device)
        labels = load_label_sequences(split)
        overall = operating_metrics(labels, probabilities, threshold)
        overall["patients"] = len(labels)
        sites = by_hospital(labels, probabilities, hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = sites
        print_scores(f"{split}", overall, sites)
    return scored, hospitals


def train_one(alpha: float, copies: int, train_set, dev_set, dev_labels, pos_weight_value, device) -> dict:
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    sampler = LengthBatchSampler(np.diff(train_set.offsets), CONFIG["batch_size"], generator)
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate, pin_memory=device.type == "cuda")
    model = CausalLSTM(N_FEATURES, CONFIG["hidden"], CONFIG["n_layers"], CONFIG["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=CONFIG["lr"] / 20)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_threshold = None
    history = []
    print(f"\nmix {alpha:.2f}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, pos_weight, alpha, copies, device)
        schedule.step()
        held_out = dev_loss(model, dev_set, pos_weight, device)
        probabilities = predict_ordered(model, dev_set, 128, device)
        threshold, utility = choose_threshold(dev_labels, probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "dev_loss": float(held_out),
                "dev_utility": float(utility),
                "threshold": float(threshold),
            }
        )
        print(
            f"  epoch {epoch:02d}  train loss {loss:.4f}  dev loss {held_out:.4f}  dev utility {utility:.4f}",
            flush=True,
        )
        if utility > best_utility:
            best_utility = float(utility)
            best_epoch = epoch
            best_threshold = float(threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    print(f"kept epoch {best_epoch}  threshold {best_threshold:.3f}", flush=True)
    return {
        "model": model,
        "septic_fraction": alpha,
        "copies": copies,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "dev_utility": best_utility,
        "history": history,
    }


def main(ratios=RATIOS, copies: int = 1, metrics_name: str = "metrics.json") -> None:
    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}  copies {copies}", flush=True)
    splits = {split: StayDataset(split) for split in ("train", "dev", "test")}
    dev_labels = load_label_sequences("dev")
    RUN.mkdir(parents=True, exist_ok=True)
    trained = []
    for alpha in ratios:
        result = train_one(alpha, copies, splits["train"], splits["dev"], dev_labels, float(meta["pos_weight"]), device)
        model = result.pop("model")
        torch.save(
            {"state_dict": {key: value.cpu() for key, value in model.state_dict().items()}, "config": CONFIG, "alpha": alpha},
            RUN / f"mix_{alpha:.2f}.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        trained.append(result)
        (RUN / "partial.json").write_text(json.dumps(trained, indent=2), encoding="utf-8")
    winner = max(trained, key=lambda row: row["dev_utility"])
    print(
        f"\nselected mix {winner['septic_fraction']:.2f} by dev utility {winner['dev_utility']:.4f}",
        flush=True,
    )
    report = []
    for row in trained:
        saved = torch.load(RUN / f"mix_{row['septic_fraction']:.2f}.pt", map_location=device, weights_only=False)
        model = CausalLSTM(N_FEATURES, CONFIG["hidden"], CONFIG["n_layers"], CONFIG["dropout"]).to(device)
        model.load_state_dict(saved["state_dict"])
        print(f"\nmix {row['septic_fraction']:.2f}", flush=True)
        scored, hospitals = score(model, splits, row["threshold"], device)
        record = dict(row)
        record["dev"] = scored["dev"]
        record["test"] = scored["test"]
        record["train"] = scored["train"]
        record["hospitals"] = hospitals
        record["selected"] = row["septic_fraction"] == winner["septic_fraction"]
        report.append(record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "model": "searched causal LSTM with a small septic mix into a normal stay",
        "config": CONFIG,
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint": "epoch with the highest dev utility, separately for each mix weight",
        "mix": "mixed = (1 - septic_fraction) * normal stay + septic_fraction * septic stay; labels copied from the septic stay",
        "copies_per_septic_stay": copies,
        "loss": "weighted BCE on the real batch and the mixed stays together",
        "selection": "highest dev utility across mix weights",
        "unmixed_test_utility": 0.3752571288529807,
        "runs": report,
    }
    (RUN / metrics_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\ncomparison", flush=True)
    for row in report:
        mark = " selected" if row["selected"] else ""
        print(
            f"mix {row['septic_fraction']:.2f}  epoch {row['best_epoch']:2d}  "
            f"dev {row['dev']['utility']:.4f}  test {row['test']['utility']:.4f}{mark}",
            flush=True,
        )
    print(f"wrote {RUN / metrics_name}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=float, nargs="*", default=list(RATIOS))
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--metrics", default="metrics.json")
    args = parser.parse_args()
    main(tuple(args.ratios), args.copies, args.metrics)
