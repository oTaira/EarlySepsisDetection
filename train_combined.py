"""CNN and LSTM on Separatrix features plus the lab window summaries.

The labs are the ones with gain in Sepsyd and CTL: FiO2, lactate, pH,
creatinine, BUN, WBC, AST, and PaCO2. Each lab gets the same causal window
summaries Separatrix builds for a vital. The epoch with the highest dev
utility is kept. Test is scored once per model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from torch.utils.data import DataLoader, Dataset

from dataset import PROCESSED, StayDataset, collate, load_label_sequences, prepare
from model import SWInSiLU, CausalLSTM, SimpleCausalCNN
from search_cnn_broad import LengthBatchSampler, predict_ordered
from train_cnn_baseline import operating_metrics, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_mlp import cached_matrix
from utility import choose_threshold
from winner_data import _causal_fill, _prefix_entropy, _quantile, load_raw, training_means
from winner_data import _index

RUN = Path(__file__).resolve().parent / "runs" / "combined_features"
EPOCHS = 10
LABS = ["FiO2", "Lactate", "pH", "Creatinine", "BUN", "WBC", "AST", "PaCO2"]
PER_LAB = 23
CNN = {
    "hidden": 64,
    "kernel_size": 9,
    "n_layers": 3,
    "dropout": 0.4,
    "lr": 0.00022868030122031387,
    "weight_decay": 0.001,
    "batch_size": 128,
    "noise_std": 0.1,
    "seed": 246,
}
LSTM = {
    "hidden": 64,
    "n_layers": 3,
    "dropout": 0.282,
    "lr": 0.0006441257389397937,
    "weight_decay": 0.0001,
    "batch_size": 64,
    "noise_std": 0.1,
    "seed": 248,
}


@njit
def _lab_block(imputed, observed, out):
    t_len = imputed.shape[0]
    for t in range(t_len):
        pos = 0
        for w in (5, 11):
            start = t + 1 - w
            buf = np.empty(w)
            for k in range(w):
                src = start + k
                buf[k] = imputed[0] if src < 0 else imputed[src]
            buf.sort()
            mean = 0.0
            for k in range(w):
                mean += buf[k]
            mean /= w
            var = 0.0
            for k in range(w):
                var += (buf[k] - mean) * (buf[k] - mean)
            var /= w
            out[t, pos] = mean
            out[t, pos + 1] = buf[0]
            out[t, pos + 2] = buf[w - 1]
            out[t, pos + 3] = _quantile(buf, 0.5)
            out[t, pos + 4] = var
            out[t, pos + 5] = _quantile(buf, 0.95)
            out[t, pos + 6] = _quantile(buf, 0.99)
            out[t, pos + 7] = _quantile(buf, 0.05)
            out[t, pos + 8] = _quantile(buf, 0.01)
            pos += 9
        energy = 0.0
        for i in range(t + 1):
            energy += imputed[i] * imputed[i]
        diff = 0.0
        if t > 0:
            for i in range(t):
                diff += imputed[i + 1] - imputed[i]
            diff /= t
        out[t, pos] = energy
        out[t, pos + 1] = _prefix_entropy(imputed[: t + 1].copy())
        out[t, pos + 2] = diff
        pos += 3
        count = 0.0
        for i in range(t + 1):
            count += observed[i]
        out[t, pos] = count
        out[t, pos + 1] = imputed[t]


def lab_matrix(split: str, means: np.ndarray) -> np.ndarray:
    path = PROCESSED / f"lab_windows_causal_{split}.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        print(f"lab windows {split} {cached.shape}", flush=True)
        return cached
    values, _, offsets = load_raw(split)
    columns = [_index(name) for name in LABS]
    matrix = np.zeros((int(offsets[-1]), len(LABS) * PER_LAB), dtype=np.float32)
    block = np.zeros((336, PER_LAB), dtype=np.float64)
    for patient, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw = np.asarray(values[int(start) : int(end)], dtype=np.float64)
        length = raw.shape[0]
        for lab, column in enumerate(columns):
            series = raw[:, column]
            imputed = _causal_fill(series, float(means[column]))
            observed = np.isfinite(series).astype(np.float64)
            _lab_block(imputed, observed, block[:length])
            matrix[int(start) : int(end), lab * PER_LAB : (lab + 1) * PER_LAB] = block[:length]
        if (patient + 1) % 4000 == 0:
            print(f"lab windows {split} {patient + 1}/{offsets.size - 1}", flush=True)
    np.save(path, matrix)
    print(f"cached {path} {matrix.shape}", flush=True)
    return np.load(path, mmap_mode="r")


def standardization(separatrix, labs):
    width = separatrix.shape[1] + labs.shape[1]
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    count = separatrix.shape[0]
    for start in range(0, count, 200_000):
        end = min(count, start + 200_000)
        block = np.concatenate([np.asarray(separatrix[start:end]), np.asarray(labs[start:end])], axis=1)
        values = block.astype(np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    mu = (total / count).astype(np.float32)
    variance = total_sq / count - np.square(total / count)
    sd = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    return mu, sd


class CombinedStayDataset(Dataset):
    def __init__(self, split: str, separatrix, labs, mu: np.ndarray, sd: np.ndarray):
        self.base = StayDataset(split)
        self.offsets = self.base.offsets
        self.separatrix = separatrix
        self.labs = labs
        self.mu = mu
        self.sd = sd
        if separatrix.shape[0] != int(self.offsets[-1]) or labs.shape[0] != int(self.offsets[-1]):
            raise RuntimeError(f"{split} feature rows do not match the stays")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        _, labels = self.base[index]
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        block = np.concatenate(
            [np.asarray(self.separatrix[start:end]), np.asarray(self.labs[start:end])], axis=1
        )
        block = (block - self.mu) / self.sd
        return torch.from_numpy(np.ascontiguousarray(block.T)), labels


def train_epoch(model, loader, optimizer, pos_weight, noise_std, device) -> float:
    model.train()
    total = 0.0
    hours = 0
    for features, labels, mask in loader:
        features = features.to(device, non_blocking=True)
        if noise_std > 0:
            features = features + torch.randn_like(features) * noise_std
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
def split_loss(model, dataset, pos_weight, device) -> float:
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
        print_scores(split, overall, sites)
    return scored, hospitals


def run_model(name, model, config, splits, dev_labels, pos_weight_value, device, cosine: bool):
    set_seed(config["seed"])
    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    sampler = LengthBatchSampler(np.diff(splits["train"].offsets), config["batch_size"], generator)
    loader = DataLoader(splits["train"], batch_sampler=sampler, collate_fn=collate, pin_memory=device.type == "cuda")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=config["lr"] / 20) if cosine else None
    pos_weight = torch.tensor([pos_weight_value], device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_threshold = None
    history = []
    print(f"\n{name}  parameters {parameters}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, pos_weight, config["noise_std"], device)
        if schedule is not None:
            schedule.step()
        held_out = split_loss(model, splits["dev"], pos_weight, device)
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
    scored, hospitals = score(model, splits, best_threshold, device)
    torch.save({"state_dict": best_state, "config": config, "threshold": best_threshold}, RUN / f"{name}.pt")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameters": parameters,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "history": history,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
    }


def main() -> None:
    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    means = training_means()
    separatrix = {split: cached_matrix(split, means) for split in ("train", "dev", "test")}
    labs = {split: lab_matrix(split, means) for split in ("train", "dev", "test")}
    mu, sd = standardization(separatrix["train"], labs["train"])
    splits = {
        split: CombinedStayDataset(split, separatrix[split], labs[split], mu, sd) for split in ("train", "dev", "test")
    }
    width = separatrix["train"].shape[1] + labs["train"].shape[1]
    print(f"input channels {width}", flush=True)
    dev_labels = load_label_sequences("dev")
    RUN.mkdir(parents=True, exist_ok=True)
    cnn = SimpleCausalCNN(
        width,
        hidden=CNN["hidden"],
        kernel_size=CNN["kernel_size"],
        n_layers=CNN["n_layers"],
        dropout=CNN["dropout"],
        activation=SWInSiLU(0.05, 0.5),
    )
    lstm = CausalLSTM(width, LSTM["hidden"], LSTM["n_layers"], LSTM["dropout"])
    results = {
        "cnn": run_model("cnn", cnn, CNN, splits, dev_labels, float(meta["pos_weight"]), device, cosine=False),
        "lstm": run_model("lstm", lstm, LSTM, splits, dev_labels, float(meta["pos_weight"]), device, cosine=True),
    }
    payload = {
        "features": "Separatrix 407 columns plus 23 window summaries for each of 8 labs",
        "labs": LABS,
        "per_lab": ["6h and 12h mean, min, max, median, variance, and four tail quantiles", "energy, entropy, drift", "times measured", "current value"],
        "channels": width,
        "checkpoint": "epoch with the highest dev utility",
        "regularized_cnn_test_utility": 0.3850805973453087,
        "searched_lstm_test_utility": 0.3752571288529807,
        "separatrix_test_utility": 0.4545682369775407,
        "cnn": results["cnn"],
        "lstm": results["lstm"],
    }
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"cnn test {results['cnn']['test']['utility']:.4f}  lstm test {results['lstm']['test']['utility']:.4f}",
        flush=True,
    )
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
