"""Train the searched LSTM on the raw hours plus the Separatrix columns.

The architecture, learning rate, dropout, noise, and cosine schedule are the
ones from the best sequence-search LSTM. The extra columns are the 204
features the Separatrix trees kept, standardized on the training hours.
The epoch with the highest dev utility is kept. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

from dataset import N_FEATURES, PROCESSED, StayDataset, collate, load_label_sequences, prepare
from ensemble_cnn_separatrix import standardize
from model import CausalLSTM
from search_cnn_broad import LengthBatchSampler, predict_ordered
from search_sequence import augment
from train_cnn_baseline import operating_metrics, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_mlp import cached_matrix
from train_winners import SEED as WINNER_SEED
from train_winners import _hours, _undersample, flat_labels
from utility import choose_threshold
from winner_data import load_raw, training_means

RUN = Path(__file__).resolve().parent / "runs" / "lstm_separatrix"
EPOCHS = 10
SEED = 248
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


class SeparatrixStayDataset(Dataset):
    """Stay tensors with the Separatrix columns appended on the channel axis."""

    def __init__(self, split: str, extra: np.ndarray):
        self.base = StayDataset(split)
        self.offsets = self.base.offsets
        self.extra = np.asarray(extra, dtype=np.float32)
        if self.extra.shape[0] != int(self.offsets[-1]):
            raise RuntimeError(f"{split} Separatrix rows do not match the stay hours")
        if not np.isfinite(self.extra).all():
            raise RuntimeError(f"{split} Separatrix features are not finite")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        features, labels = self.base[index]
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        added = torch.from_numpy(self.extra[start:end].T.copy())
        return torch.cat((features, added), dim=0), labels


def kept_columns(means):
    raw = cached_matrix("train", means)
    y_train = flat_labels("train")
    mu = np.asarray(raw, dtype=np.float64).mean(axis=0).astype(np.float32)
    sd = np.asarray(raw, dtype=np.float64).std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    x_train = standardize(raw, mu, sd)
    _, _, offsets = load_raw("train")
    rng = np.random.default_rng(WINNER_SEED)
    patients = np.arange(offsets.size - 1)
    rng.shuffle(patients)
    n_screen = max(1, int(0.10 * patients.size))
    screen = _undersample(y_train, _hours(patients[:n_screen], offsets), rng)
    screener = XGBClassifier(
        n_estimators=40, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=WINNER_SEED
    )
    screener.fit(x_train[screen], y_train[screen], verbose=False)
    gain = screener.get_booster().get_score(importance_type="gain")
    keep = sorted(int(name[1:]) for name in gain if gain[name] > 0)
    print(f"kept {len(keep)} of {x_train.shape[1]} Separatrix features", flush=True)
    return keep, mu, sd


def extra_matrix(split: str, means, keep, mu, sd) -> np.ndarray:
    raw_offsets = load_raw(split)[2]
    stay_offsets = np.load(PROCESSED / f"{split}_offsets.npy")
    if not np.array_equal(raw_offsets, stay_offsets):
        raise RuntimeError(f"{split} stay order does not match the Separatrix hours")
    matrix = standardize(cached_matrix(split, means), mu, sd)[:, keep]
    print(f"{split} extra features {matrix.shape}", flush=True)
    return matrix


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


def train_epoch(model, loader, optimizer, pos_weight, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = augment(features.to(device, non_blocking=True), CONFIG["noise_std"])
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


def score(model, splits, threshold, device):
    scored = {}
    hospitals = {}
    for split, dataset in splits.items():
        probabilities = predict_ordered(model, dataset, 128, device)
        labels = load_label_sequences(split)
        overall = operating_metrics(labels, probabilities, threshold)
        overall["patients"] = len(labels)
        by_site = by_hospital(labels, probabilities, hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(split, overall, by_site)
    return scored, hospitals


def main() -> None:
    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    means = training_means()
    keep, mu, sd = kept_columns(means)
    splits = {
        split: SeparatrixStayDataset(split, extra_matrix(split, means, keep, mu, sd))
        for split in ("train", "dev", "test")
    }
    dev_labels = load_label_sequences("dev")
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    sampler = LengthBatchSampler(np.diff(splits["train"].offsets), CONFIG["batch_size"], generator)
    loader = DataLoader(
        splits["train"], batch_sampler=sampler, collate_fn=collate, pin_memory=device.type == "cuda"
    )
    model = CausalLSTM(N_FEATURES + len(keep), CONFIG["hidden"], CONFIG["n_layers"], CONFIG["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=CONFIG["lr"] / 20)
    pos_weight = torch.tensor([float(meta["pos_weight"])], device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters {parameters}  input channels {N_FEATURES + len(keep)}", flush=True)
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_threshold = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, pos_weight, device)
        schedule.step()
        held_out = dev_loss(model, splits["dev"], pos_weight, device)
        probabilities = predict_ordered(model, splits["dev"], 128, device)
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
            f"epoch {epoch:02d}  train loss {loss:.4f}  dev loss {held_out:.4f}  dev utility {utility:.4f}",
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
    scored, hospitals = score(model, splits, best_threshold, device)
    payload = {
        "model": "causal LSTM on raw hours plus Separatrix columns",
        "config": CONFIG,
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint": "epoch with the highest dev utility",
        "best_epoch": best_epoch,
        "parameters": parameters,
        "raw_channels": N_FEATURES,
        "separatrix_channels": len(keep),
        "kept_columns": keep,
        "history": history,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "searched_lstm_test_utility": 0.3753,
        "separatrix_test_utility": 0.4545682369775407,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {"state_dict": best_state, "config": CONFIG, "threshold": best_threshold, "kept_columns": keep},
        RUN / "model.pt",
    )
    print(f"dev {scored['dev']['utility']:.4f}  test {scored['test']['utility']:.4f}", flush=True)
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
