"""Broader CNN search: dilated causal 1D, variable-by-time 2D, and temporal attention.

Training loss is weighted cross-entropy. Dev utility chooses the alarm
threshold and the winning trial. Test is scored once. Each trial keeps the
epoch with the best dev utility and stops after three epochs without a new best.
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

from dataset import N_FEATURES, N_PHYSIO, StayDataset, collate, load_label_sequences, prepare
from model import CausalAttentionNet, FeatureTimeCNN, SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold

SEED = 231
MAX_EPOCHS = 8
PATIENCE = 3
MAX_TRIALS = 80
BUDGET_HOURS = 6
RUN = Path(__file__).resolve().parent / "runs" / "cnn_broad"


class LengthBatchSampler(Sampler[list[int]]):
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


def sample_config(rng: np.random.Generator) -> dict:
    family = str(rng.choice(["causal_1d", "feature_time_2d", "temporal_attention"], p=[0.5, 0.25, 0.25]))
    activation = str(rng.choice(["relu", "silu", "gelu", "swin"]))
    config = {
        "family": family,
        "dropout": round(float(rng.uniform(0.0, 0.4)), 3),
        "lr": float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3)))),
        "weight_decay": float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        "batch_size": int(rng.choice([64, 128])),
        "noise_std": float(rng.choice([0.0, 0.05, 0.1])),
        "activation": activation,
    }
    if activation == "swin":
        config["swin_a"] = float(rng.choice([0.05, 0.1, 0.2]))
        config["swin_omega"] = float(rng.choice([0.5, 1.0, 2.0]))
    if family == "causal_1d":
        config.update(
            hidden=int(rng.choice([64, 96, 128, 192])),
            kernel_size=int(rng.choice([5, 7, 9, 11])),
            n_layers=int(rng.choice([3, 4, 5, 6])),
            dilation=int(rng.choice([1, 2, 4])),
        )
    elif family == "feature_time_2d":
        config.update(
            hidden=int(rng.choice([16, 32, 48])),
            kernel_h=int(rng.choice([3, 7, 15, 31])),
            kernel_w=int(rng.choice([3, 5, 9])),
            n_layers=int(rng.choice([3, 4, 5])),
        )
    else:
        config.update(
            hidden=int(rng.choice([32, 64, 96])),
            heads=int(rng.choice([2, 4])),
            n_layers=int(rng.choice([1, 2])),
            batch_size=int(rng.choice([32, 64])),
        )
    return config


def activation_module(config: dict) -> nn.Module:
    name = config["activation"]
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swin":
        return SWInSiLU(config["swin_a"], config["swin_omega"])
    raise ValueError(name)


def build_model(config: dict) -> nn.Module:
    activation = activation_module(config)
    family = config["family"]
    if family == "causal_1d":
        model = SimpleCausalCNN(
            N_FEATURES,
            hidden=config["hidden"],
            kernel_size=config["kernel_size"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            activation=activation,
            dilation=config["dilation"],
        )
        model.family_note = {
            "time_receptive_field": model.receptive_field,
            "feature_receptive_field": N_FEATURES,
        }
        return model
    if family == "feature_time_2d":
        model = FeatureTimeCNN(
            hidden=config["hidden"],
            kernel_h=config["kernel_h"],
            kernel_w=config["kernel_w"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            activation=activation,
        )
        model.family_note = {
            "time_receptive_field": model.time_receptive_field,
            "feature_receptive_field": model.feature_receptive_field,
        }
        return model
    model = CausalAttentionNet(
        N_FEATURES,
        hidden=config["hidden"],
        heads=config["heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        activation=activation,
    )
    model.family_note = {"time_receptive_field": "all past hours", "feature_receptive_field": N_FEATURES}
    return model


def augment(features: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0:
        return features
    observed = features[:, N_PHYSIO : 2 * N_PHYSIO]
    noise = torch.randn_like(features[:, :N_PHYSIO]) * noise_std
    out = features.clone()
    out[:, :N_PHYSIO] = features[:, :N_PHYSIO] + noise * observed
    return out


def train_epoch(model, loader, optimizer, pos_weight, noise_std, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = augment(features.to(device, non_blocking=True), noise_std)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(features)[mask], labels[mask], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(mask.sum().item())
        total += float(loss.detach().item()) * count
        hours += count
    return total / max(hours, 1)


@torch.no_grad()
def predict_ordered(model, dataset: StayDataset, batch_size: int, device) -> list[np.ndarray]:
    model.eval()
    lengths = np.diff(dataset.offsets)
    order = np.argsort(lengths, kind="mergesort")
    sequences: list[np.ndarray | None] = [None] * len(dataset)
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        features, _labels, mask = collate([dataset[int(index)] for index in chosen])
        probabilities = torch.sigmoid(model(features.to(device, non_blocking=True))).cpu().numpy()
        valid = mask.numpy()
        for row, index in enumerate(chosen):
            length = int(valid[row].sum())
            sequences[int(index)] = probabilities[row, :length].astype(np.float64)
    if any(sequence is None for sequence in sequences):
        raise RuntimeError("a stay was not scored")
    return sequences


def run_trial(trial: int, config: dict, train_set, dev_set, dev_labels, pos_weight_value: float, device, max_epochs: int) -> dict:
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
    note = dict(model.family_note)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    pos_weight = torch.tensor([pos_weight_value], device=device)
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_dev = None
    stale = 0
    history = []
    parameters = sum(parameter.numel() for parameter in model.parameters())
    try:
        for epoch in range(1, max_epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, pos_weight, config["noise_std"], device)
            if not np.isfinite(loss):
                print(f"  epoch {epoch:02d}  non-finite loss, stopping trial", flush=True)
                break
            dev_probabilities = predict_ordered(model, dev_set, max(config["batch_size"], 64), device)
            threshold, utility = choose_threshold(dev_labels, dev_probabilities)
            history.append(
                {"epoch": epoch, "train_loss": float(loss), "dev_utility": float(utility), "threshold": float(threshold)}
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
        "receptive_field": note,
        "best_epoch": best_epoch,
        "dev": best_dev,
        "history": history,
        "state": best_state,
    }


def score_split(model, dataset, split: str, threshold: float, device):
    labels = load_label_sequences(split)
    probabilities = predict_ordered(model, dataset, 128, device)
    overall = operating_metrics(labels, probabilities, threshold)
    overall["patients"] = len(labels)
    hospitals = by_hospital(labels, probabilities, hospital_names(split), threshold)
    return overall, hospitals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    max_trials = 3 if args.smoke else MAX_TRIALS
    max_epochs = 1 if args.smoke else MAX_EPOCHS
    budget_seconds = 30 * 60 if args.smoke else BUDGET_HOURS * 3600
    run_dir = RUN.parent / "cnn_broad_smoke" if args.smoke else RUN

    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device {device}. Up to {max_trials} trials or {budget_seconds / 3600:.1f} hours. "
        "Families: dilated causal 1D, variable-by-time 2D, temporal attention. "
        "Loss is weighted cross-entropy. Test is scored once.",
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
    if args.smoke:
        configs = []
        seen = set()
        while len(configs) < 3:
            drawn = sample_config(rng)
            if drawn["family"] not in seen:
                seen.add(drawn["family"])
                configs.append(drawn)
    else:
        configs = None

    for trial in range(1, max_trials + 1):
        if trial > 1 and time.perf_counter() - started >= budget_seconds:
            print(f"budget reached after {trial - 1} trials", flush=True)
            break
        config = configs[trial - 1] if configs is not None else sample_config(rng)
        print(f"\ntrial {trial}/{max_trials}  {json.dumps(config)}", flush=True)
        try:
            result = run_trial(trial, config, train_set, dev_set, dev_labels, float(meta["pos_weight"]), device, max_epochs)
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
            f"trial {trial} {config['family']} epoch {result['best_epoch']}  "
            f"dev utility {utility:.4f}  params {result['parameters']}",
            flush=True,
        )
        if best is None or utility > best["dev"]["utility"]:
            best = result
            best_state = state
            torch.save({"state_dict": state, "config": config, "dev": result["dev"]}, run_dir / "best.pt")
            print(f"new best dev utility {utility:.4f}", flush=True)

    if args.smoke:
        print("smoke trials finished", flush=True)
        return
    if best is None or best_state is None:
        print("no trial finished with a dev score", flush=True)
        return

    model = build_model(best["config"]).to(device)
    model.load_state_dict(best_state)
    threshold = best["dev"]["threshold"]
    scored = {}
    hospitals = {}
    for split, dataset in {"train": train_set, "dev": dev_set, "test": StayDataset("test")}.items():
        overall, by_site = score_split(model, dataset, split, threshold, device)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"{best['config']['family']} {split}", overall, by_site)
    summary = {
        "loss": "weighted binary cross-entropy; utility is the selection score",
        "families": ["causal_1d", "feature_time_2d", "temporal_attention"],
        "trials_completed": sum(1 for record in records if record.get("dev")),
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
        {"state_dict": best_state, "config": best["config"], "threshold": threshold},
        run_dir / "model.pt",
    )
    print(
        f"\nbest trial {best['trial']} {best['config']['family']}  "
        f"dev {scored['dev']['utility']:.4f}  test {scored['test']['utility']:.4f}",
        flush=True,
    )
    print(f"wrote {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
