"""Overnight random search for the correlation GAT.

Each trial trains up to 10 epochs. Weights from the best dev-utility epoch are
kept, and a trial stops after three epochs without a new best. The alarm
threshold is chosen on dev. Test is scored once, after the search, for the
single best trial. Stays are batched with others of similar length so padding
does not stretch every batch to the longest stay in the sample.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Sampler

from dataset import N_PHYSIO, StayDataset, collate, load_label_sequences, prepare
from model import SWInSiLU, PhysioGAT
from train_cnn_baseline import operating_metrics, set_seed
from train_gat import correlation_matrix, edges_for
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold

SEED = 230
MAX_EPOCHS = 10
PATIENCE = 3
MAX_TRIALS = 200
BUDGET_HOURS = 10
EDGE_THRESHOLDS = (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
RUN = Path(__file__).resolve().parent / "runs" / "gat_search"


class LengthBatchSampler(Sampler[list[int]]):
    """Batches of similar stay length, with the batch order reshuffled each epoch."""

    def __init__(self, lengths: np.ndarray, batch_size: int, generator: torch.Generator):
        order = np.argsort(lengths, kind="mergesort")
        self.batches = [order[start : start + batch_size].tolist() for start in range(0, len(order), batch_size)]
        self.generator = generator

    def __iter__(self):
        permutation = torch.randperm(len(self.batches), generator=self.generator).tolist()
        for index in permutation:
            yield self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


def sample_config(rng: np.random.Generator, thresholds: list[float]) -> dict:
    hidden = 64
    heads = 4
    for _ in range(30):
        hidden = int(rng.choice([16, 32, 48, 64, 96]))
        heads = int(rng.choice([1, 2, 4, 8]))
        if hidden * heads <= 256:
            break
    activation = str(rng.choice(["elu", "relu", "silu", "gelu", "mish", "swin"]))
    config = {
        "hidden": hidden,
        "heads": heads,
        "n_layers": int(rng.choice([2, 3])),
        "dropout": round(float(rng.uniform(0.0, 0.5)), 3),
        "lr": float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3)))),
        "weight_decay": float(rng.choice([0.0, 1e-6, 1e-5, 1e-4, 1e-3])),
        "batch_size": int(rng.choice([32, 64, 128])),
        "edge_threshold": float(rng.choice(thresholds)),
        "noise_std": float(rng.choice([0.0, 0.05, 0.1])),
        "pos_weight_scale": float(rng.choice([0.25, 0.5, 1.0, 2.0])),
        "negative_slope": float(rng.choice([0.1, 0.2, 0.3])),
        "pool": str(rng.choice(["mean", "max"])),
        "activation": activation,
    }
    if activation == "swin":
        config["swin_a"] = float(rng.choice([0.05, 0.1, 0.2]))
        config["swin_omega"] = float(rng.choice([0.5, 1.0, 2.0]))
    return config


def activation_module(config: dict) -> nn.Module:
    name = config["activation"]
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name == "swin":
        return SWInSiLU(config["swin_a"], config["swin_omega"])
    raise ValueError(name)


def build_model(config: dict) -> PhysioGAT:
    return PhysioGAT(
        hidden=config["hidden"],
        heads=config["heads"],
        dropout=config["dropout"],
        n_layers=config["n_layers"],
        activation=activation_module(config),
        negative_slope=config["negative_slope"],
        pool=config["pool"],
    )


def augment(features: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0:
        return features
    observed = features[:, N_PHYSIO : 2 * N_PHYSIO]
    noise = torch.randn_like(features[:, :N_PHYSIO]) * noise_std
    out = features.clone()
    out[:, :N_PHYSIO] = features[:, :N_PHYSIO] + noise * observed
    return out


def train_epoch(model, loader, optimizer, pos_weight, edge_index, noise_std, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = augment(features.to(device, non_blocking=True), noise_std)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(mask.sum().item())
        total += float(loss.detach().item()) * count
        hours += count
    return total / max(hours, 1)


@torch.no_grad()
def predict_ordered(model, dataset: StayDataset, batch_size: int, edge_index, device) -> list[np.ndarray]:
    """Score every stay once, grouped by length, and return manifest order."""
    model.eval()
    lengths = np.diff(dataset.offsets)
    order = np.argsort(lengths, kind="mergesort")
    sequences: list[np.ndarray | None] = [None] * len(dataset)
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        features, _labels, mask = collate([dataset[int(index)] for index in chosen])
        logits = model(features.to(device, non_blocking=True), edge_index)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        valid = mask.numpy()
        for row, index in enumerate(chosen):
            length = int(valid[row].sum())
            sequences[int(index)] = probabilities[row, :length].astype(np.float64)
    if any(sequence is None for sequence in sequences):
        raise RuntimeError("a stay was not scored")
    return sequences


def run_trial(trial: int, config: dict, train_set, dev_set, dev_labels, edge_index, pos_weight_value: float, device, max_epochs: int) -> dict:
    set_seed(SEED + trial)
    generator = torch.Generator()
    generator.manual_seed(SEED + trial)
    sampler = LengthBatchSampler(np.diff(train_set.offsets), config["batch_size"], generator)
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    pos_weight = torch.tensor([pos_weight_value * config["pos_weight_scale"]], device=device)
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_dev = None
    stale = 0
    history = []
    parameters = sum(parameter.numel() for parameter in model.parameters())
    try:
        for epoch in range(1, max_epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, pos_weight, edge_index, config["noise_std"], device)
            if not np.isfinite(loss):
                history.append({"epoch": epoch, "train_loss": None, "dev_utility": None, "threshold": None})
                print(f"  epoch {epoch:02d}  non-finite loss, stopping trial", flush=True)
                break
            dev_probabilities = predict_ordered(model, dev_set, max(config["batch_size"], 64), edge_index, device)
            threshold, utility = choose_threshold(dev_labels, dev_probabilities)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss),
                    "dev_utility": float(utility),
                    "threshold": float(threshold),
                }
            )
            print(f"  epoch {epoch:02d}  train loss {loss:.4f}  dev utility {utility:.4f}", flush=True)
            if utility > best_utility:
                best_utility = float(utility)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_dev = operating_metrics(dev_labels, dev_probabilities, threshold)
                best_dev["patients"] = len(dev_labels)
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    finally:
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "trial": trial,
        "config": config,
        "parameters": parameters,
        "undirected_edges": int(edge_index.shape[1] // 2),
        "best_epoch": best_epoch,
        "dev": best_dev,
        "history": history,
        "state": best_state,
    }


def score_split(model, dataset, split: str, threshold: float, edge_index, device):
    labels = load_label_sequences(split)
    probabilities = predict_ordered(model, dataset, 128, edge_index, device)
    overall = operating_metrics(labels, probabilities, threshold)
    overall["patients"] = len(labels)
    hospitals = by_hospital(labels, probabilities, hospital_names(split), threshold)
    return overall, hospitals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Train one epoch of one trial and stop.")
    args = parser.parse_args()
    max_trials = 1 if args.smoke else MAX_TRIALS
    max_epochs = 1 if args.smoke else MAX_EPOCHS
    budget_seconds = 20 * 60 if args.smoke else BUDGET_HOURS * 3600
    run_dir = RUN.parent / "gat_search_smoke" if args.smoke else RUN

    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corr, _support = correlation_matrix()
    graphs = []
    seen_counts = set()
    for threshold in EDGE_THRESHOLDS:
        edge_index = edges_for(corr, threshold)
        count = int(edge_index.shape[1] // 2)
        if count in seen_counts:
            print(f"threshold {threshold:.2f} repeats {count} edges, skipping", flush=True)
            continue
        seen_counts.add(count)
        graphs.append((float(threshold), count, edge_index))
        print(f"threshold {threshold:.2f}  undirected edges {count}", flush=True)
    graph_for = {threshold: edge_index.to(device) for threshold, _count, edge_index in graphs}
    thresholds = [threshold for threshold, _count, _edges in graphs]

    print(
        f"device {device}. Up to {max_trials} trials or {budget_seconds / 3600:.1f} hours. "
        f"{max_epochs} epochs, patience {PATIENCE}. Test is scored once.",
        flush=True,
    )
    train_set = StayDataset("train")
    dev_set = StayDataset("dev")
    dev_labels = load_label_sequences("dev")
    rng = np.random.default_rng(SEED)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    best = None
    best_state = None

    for trial in range(1, max_trials + 1):
        elapsed = time.perf_counter() - started
        if trial > 1 and elapsed >= budget_seconds:
            print(f"budget reached after {trial - 1} trials", flush=True)
            break
        config = sample_config(rng, thresholds)
        print(f"\ntrial {trial}/{max_trials}  {json.dumps(config)}", flush=True)
        try:
            result = run_trial(
                trial,
                config,
                train_set,
                dev_set,
                dev_labels,
                graph_for[config["edge_threshold"]],
                float(meta["pos_weight"]),
                device,
                max_epochs,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"trial {trial} skipped: out of memory", flush=True)
            records.append({"trial": trial, "config": config, "error": "out of memory"})
            (run_dir / "search.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            continue
        state = result.pop("state")
        records.append(result)
        (run_dir / "search.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if result["dev"] is None:
            print(f"trial {trial} produced no finite dev score", flush=True)
            continue
        utility = result["dev"]["utility"]
        print(
            f"trial {trial} best epoch {result['best_epoch']}  dev utility {utility:.4f}  "
            f"params {result['parameters']}  edges {result['undirected_edges']}",
            flush=True,
        )
        if best is None or utility > best["dev"]["utility"]:
            best = result
            best_state = state
            torch.save(
                {
                    "state_dict": state,
                    "config": config,
                    "dev": result["dev"],
                    "best_epoch": result["best_epoch"],
                    "edge_threshold": config["edge_threshold"],
                },
                run_dir / "best.pt",
            )
            print(f"new best dev utility {utility:.4f}", flush=True)

    if args.smoke:
        print("smoke trial finished", flush=True)
        return
    if best is None or best_state is None:
        print("no trial finished with a dev score", flush=True)
        return

    model = build_model(best["config"]).to(device)
    model.load_state_dict(best_state)
    edge_index = graph_for[best["config"]["edge_threshold"]]
    threshold = best["dev"]["threshold"]
    scored = {}
    hospitals = {}
    sets = {"train": train_set, "dev": dev_set, "test": StayDataset("test")}
    for split, dataset in sets.items():
        overall, by_site = score_split(model, dataset, split, threshold, edge_index, device)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"GAT {split}", overall, by_site)
    summary = {
        "model": "correlation GAT, random search",
        "trials_completed": sum(1 for record in records if "dev" in record and record["dev"] is not None),
        "trials_attempted": len(records),
        "budget_hours": BUDGET_HOURS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection": "highest dev utility, weights from that trial's best epoch",
        "elapsed_hours": (time.perf_counter() - started) / 3600,
        "best": best,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "regularized_cnn_test_utility": 0.3851,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": best_state,
            "config": best["config"],
            "threshold": threshold,
            "edge_threshold": best["config"]["edge_threshold"],
        },
        run_dir / "model.pt",
    )
    print(
        f"\nbest trial {best['trial']}  dev {scored['dev']['utility']:.4f}  test {scored['test']['utility']:.4f}",
        flush=True,
    )
    print(f"config {json.dumps(best['config'])}", flush=True)
    print(f"wrote {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
