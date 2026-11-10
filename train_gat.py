"""Train the correlation GAT and choose the edge threshold on dev utility.

Each hour is a 34-node graph. An undirected edge is kept when the training-set
Pearson correlation of jointly observed values is at least the threshold and
the pair was measured together at least 500 times. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import N_PHYSIO, StayDataset, collate, load_label_sequences, prepare
from model import PhysioGAT
from train_cnn_baseline import operating_metrics, predict, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold
from winner_data import load_raw

SEED = 230
EPOCHS = 10
NOISE_STD = 0.1
MIN_PAIRS = 500
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
RUN = Path(__file__).resolve().parent / "runs" / "gat"


def augment(features: torch.Tensor) -> torch.Tensor:
    observed = features[:, N_PHYSIO : 2 * N_PHYSIO]
    noise = torch.randn_like(features[:, :N_PHYSIO]) * NOISE_STD
    out = features.clone()
    out[:, :N_PHYSIO] = features[:, :N_PHYSIO] + noise * observed
    return out


def correlation_matrix() -> tuple[np.ndarray, np.ndarray]:
    cached = RUN / "correlation.npy"
    support_path = RUN / "pair_counts.npy"
    if cached.exists() and support_path.exists():
        return np.load(cached), np.load(support_path)
    values, _, _ = load_raw("train")
    phys = np.asarray(values[:, :N_PHYSIO], dtype=np.float64)
    corr = np.eye(N_PHYSIO, dtype=np.float64)
    support = np.zeros((N_PHYSIO, N_PHYSIO), dtype=np.int64)
    for left in range(N_PHYSIO):
        column = phys[:, left]
        finite = np.isfinite(column)
        for right in range(left + 1, N_PHYSIO):
            both = finite & np.isfinite(phys[:, right])
            count = int(both.sum())
            support[left, right] = support[right, left] = count
            relation = 0.0
            if count >= MIN_PAIRS:
                a = column[both]
                b = phys[both, right]
                a = a - a.mean()
                b = b - b.mean()
                denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
                if denom > 0:
                    relation = float((a * b).sum() / denom)
            corr[left, right] = corr[right, left] = relation
        print(f"correlation column {left + 1}/{N_PHYSIO}", flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    np.save(cached, corr)
    np.save(support_path, support)
    return corr, support


def edges_for(corr: np.ndarray, threshold: float) -> torch.Tensor:
    src, dst = np.where(np.triu(np.abs(corr) >= threshold, k=1))
    if src.size == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    pairs = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(pairs, dtype=torch.long)


def train_epoch(model, loader, optimizer, pos_weight, edge_index, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = augment(features.to(device))
        labels = labels.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = int(mask.sum().item())
        total += float(loss.detach().item()) * n
        hours += n
    return total / hours


@torch.no_grad()
def split_loss(model, loader, pos_weight, edge_index, device) -> float:
    model.eval()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        logits = model(features, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pos_weight)
        n = int(mask.sum().item())
        total += float(loss.item()) * n
        hours += n
    return total / hours


@torch.no_grad()
def predict_gat(model, loader, edge_index, device) -> list[np.ndarray]:
    model.eval()
    sequences = []
    for features, _labels, mask in loader:
        logits = model(features.to(device), edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()
        valid = mask.numpy()
        for i in range(probs.shape[0]):
            n = int(valid[i].sum())
            sequences.append(probs[i, :n].astype(np.float64))
    return sequences


def score_split(model, split: str, threshold: float, edge_index, device) -> tuple[dict, dict]:
    loader = DataLoader(StayDataset(split), batch_size=64, shuffle=False, collate_fn=collate)
    labels = load_label_sequences(split)
    probs = predict_gat(model, loader, edge_index, device)
    overall = operating_metrics(labels, probs, threshold)
    overall["patients"] = len(labels)
    hospitals = by_hospital(labels, probs, hospital_names(split), threshold)
    return overall, hospitals


def train_one(threshold: float, edge_index, loaders, pos_weight, dev_labels, device) -> dict:
    set_seed(SEED)
    model = PhysioGAT(hidden=64, dropout=0.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00022868030122031387, weight_decay=1e-3)
    train_loader, dev_loader, test_loader = loaders
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, edge_index, device)
        dev_loss = split_loss(model, dev_loader, pos_weight, edge_index, device)
        test_loss = split_loss(model, test_loader, pos_weight, edge_index, device)
        dev_probs = predict_gat(model, dev_loader, edge_index, device)
        alarm, utility = choose_threshold(dev_labels, dev_probs)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "dev_loss": dev_loss,
                "test_loss": test_loss,
                "dev_utility": utility,
                "threshold": alarm,
            }
        )
        print(
            f"  epoch {epoch:02d}  train loss {loss:.4f}  dev loss {dev_loss:.4f}  "
            f"test loss {test_loss:.4f}  dev utility {utility:.4f}",
            flush=True,
        )
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "correlation_threshold": threshold,
        "undirected_edges": int(edge_index.shape[1] // 2),
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "state": state,
        "dev_utility": history[-1]["dev_utility"],
        "alarm_threshold": history[-1]["threshold"],
    }


def main() -> None:
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    corr, support = correlation_matrix()
    off = np.abs(corr[np.triu_indices(N_PHYSIO, k=1)])
    reliable = support[np.triu_indices(N_PHYSIO, k=1)] >= MIN_PAIRS
    print(
        f"pairwise |r| median {np.median(off[reliable]):.3f}, "
        f"max {off[reliable].max():.3f}, pairs {int(reliable.sum())}",
        flush=True,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    batch = 64
    train_loader = DataLoader(
        StayDataset("train"), batch_size=batch, shuffle=True, collate_fn=collate, generator=generator
    )
    dev_loader = DataLoader(StayDataset("dev"), batch_size=batch, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(StayDataset("test"), batch_size=batch, shuffle=False, collate_fn=collate)
    pos_weight = torch.tensor([meta["pos_weight"]], device=device)
    dev_labels = load_label_sequences("dev")
    seen = set()
    trials = []
    best = None
    for threshold in THRESHOLDS:
        edge_index = edges_for(corr, threshold)
        signature = edge_index.shape[1]
        if signature in seen:
            print(f"threshold {threshold:.2f} repeats a previous graph, skipping", flush=True)
            continue
        seen.add(signature)
        print(
            f"\nthreshold {threshold:.2f}  undirected edges {edge_index.shape[1] // 2}",
            flush=True,
        )
        # Reshuffle from the same seed for every graph.
        generator.manual_seed(SEED)
        result = train_one(
            threshold,
            edge_index.to(device),
            (train_loader, dev_loader, test_loader),
            pos_weight,
            dev_labels,
            device,
        )
        state = result.pop("state")
        trials.append(result)
        if best is None or result["dev_utility"] > best["dev_utility"]:
            best = result
            best_state = state
            best_edges = edge_index
    model = PhysioGAT(hidden=64, dropout=0.4).to(device)
    model.load_state_dict(best_state)
    alarm = best["alarm_threshold"]
    scored = {}
    hospitals = {}
    for split in ("train", "dev", "test"):
        overall, by_site = score_split(model, split, alarm, best_edges.to(device), device)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"GAT {split}", overall, by_site)
    payload = {
        "model": "2-layer GAT on 34 physiological nodes",
        "edge_rule": f"|Pearson r| >= threshold on training pairs with at least {MIN_PAIRS} joint observations",
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint": "last epoch of the dev-selected correlation threshold",
        "thresholds_tried": trials,
        "selected": {key: value for key, value in best.items() if key != "history"} | {"history": best["history"]},
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
    }
    # histories are already inside trials; drop duplicate bulk from selected if needed
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": best_state,
            "correlation_threshold": best["correlation_threshold"],
            "edge_index": best_edges,
            "alarm_threshold": alarm,
        },
        RUN / "model.pt",
    )
    print(
        f"\nSelected |r| >= {best['correlation_threshold']:.2f}. "
        f"Dev {scored['dev']['utility']:.4f}. Test {scored['test']['utility']:.4f}.",
        flush=True,
    )
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
