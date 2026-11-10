"""Oversample the positive class, with no extra positive loss weight.

The MLP repeats positive hours until they match the negative hours, on the
same Separatrix columns as before. The CNN draws septic and non-septic stays
with equal probability. Dev chooses the threshold. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import N_FEATURES, StayDataset, collate, load_label_sequences, prepare
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, predict, set_seed
from train_cnn_regularized import CONFIG, EPOCHS as CNN_EPOCHS
from train_cnn_regularized import SEED as CNN_SEED
from train_cnn_regularized import augment, split_loss, train_epoch
from train_lstm import by_hospital, hospital_names, print_scores
from train_separatrix_mlp import (
    BATCH,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    HourMLP,
    cached_matrix,
    kept_columns,
    predict_flat,
)
from train_winners import flat_labels, score_split
from utility import choose_threshold
from winner_data import labels_of, load_raw, sequences, training_means

MLP_RUN = Path(__file__).resolve().parent / "runs" / "separatrix_mlp_oversample"
CNN_RUN = Path(__file__).resolve().parent / "runs" / "cnn_oversample"


def positive_hour_index(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    copies = rng.choice(positive, size=negative.size, replace=True)
    order = np.concatenate([negative, copies])
    rng.shuffle(order)
    return order


def train_mlp(device) -> None:
    print("oversampled MLP on Separatrix features", flush=True)
    set_seed(230)
    means = training_means()
    x_train = cached_matrix("train", means)
    y_train = flat_labels("train")
    mu = np.asarray(x_train, dtype=np.float64).mean(axis=0).astype(np.float32)
    sd = np.asarray(x_train, dtype=np.float64).std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    standardized = (np.asarray(x_train, dtype=np.float32) - mu) / sd
    keep = kept_columns(standardized, y_train)
    train = standardized[:, keep]
    del x_train, standardized
    dev = ((np.asarray(cached_matrix("dev", means), dtype=np.float32) - mu) / sd)[:, keep]
    model = HourMLP(len(keep)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(230)
    dev_labels = labels_of("dev")
    _, _, dev_offsets = load_raw("dev")
    history = []
    best_utility = -np.inf
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = positive_hour_index(y_train, rng)
        total = 0.0
        seen = 0
        for start in range(0, order.size, BATCH):
            chosen = order[start : start + BATCH]
            features = torch.from_numpy(train[chosen]).to(device)
            labels = torch.from_numpy(y_train[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(features), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * chosen.size
            seen += chosen.size
        dev_scores = sequences(predict_flat(model, dev, device), dev_offsets)
        threshold, utility = choose_threshold(dev_labels, dev_scores)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "dev_utility": float(utility), "threshold": float(threshold)})
        print(f"mlp epoch {epoch:02d}  loss {total / seen:.4f}  dev utility {utility:.4f}", flush=True)
        if utility > best_utility:
            best_utility = float(utility)
            best_epoch = epoch
            best_threshold = float(threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    scored = {}
    hospitals = {}
    matrices = {"train": train, "dev": dev}
    for split in ("train", "dev", "test"):
        if split not in matrices:
            raw = cached_matrix(split, means)
            matrices[split] = ((np.asarray(raw, dtype=np.float32) - mu) / sd)[:, keep]
        _, _, offsets = load_raw(split)
        probabilities = sequences(predict_flat(model, matrices[split], device), offsets)
        labels = labels_of(split)
        overall = score_split(labels, probabilities, best_threshold)
        overall["patients"] = len(labels)
        by_site = by_hospital(labels, probabilities, hospital_names(split), best_threshold)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"oversampled MLP {split}", overall, by_site)
    MLP_RUN.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "Separatrix MLP, positive hours oversampled to match the negative hours, no pos_weight",
        "features": len(keep),
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "history": history,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "weighted_mlp_test_utility": 0.3918,
        "separatrix_tree_test_utility": 0.4545682369775407,
    }
    (MLP_RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "keep": keep, "threshold": best_threshold}, MLP_RUN / "model.pt")
    print(f"oversampled MLP dev {scored['dev']['utility']:.4f} test {scored['test']['utility']:.4f}", flush=True)


def train_cnn(device) -> None:
    print("oversampled causal CNN", flush=True)
    prepare()
    set_seed(CNN_SEED)
    labels = load_label_sequences("train")
    septic = np.array([bool(np.any(row)) for row in labels])
    n_pos = int(septic.sum())
    n_neg = int((~septic).sum())
    weights = np.where(septic, n_neg / n_pos, 1.0)
    generator = torch.Generator()
    generator.manual_seed(CNN_SEED)
    sampler = WeightedRandomSampler(weights, num_samples=n_neg * 2, replacement=True, generator=generator)
    batch = CONFIG["batch_size"]
    train_loader = DataLoader(StayDataset("train"), batch_size=batch, sampler=sampler, collate_fn=collate)
    dev_loader = DataLoader(StayDataset("dev"), batch_size=batch, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(StayDataset("test"), batch_size=batch, shuffle=False, collate_fn=collate)
    model = SimpleCausalCNN(
        N_FEATURES,
        hidden=CONFIG["hidden"],
        kernel_size=CONFIG["kernel_size"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
        activation=SWInSiLU(CONFIG["swin_a"], CONFIG["swin_omega"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    pos_weight = torch.tensor([1.0], device=device)
    dev_labels = load_label_sequences("dev")
    history = []
    for epoch in range(1, CNN_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        dev_loss = split_loss(model, dev_loader, pos_weight, device)
        dev_probabilities = predict(model, dev_loader, device)
        threshold, utility = choose_threshold(dev_labels, dev_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "dev_loss": dev_loss,
                "dev_utility": float(utility),
                "threshold": float(threshold),
            }
        )
        print(f"cnn epoch {epoch:02d}  loss {loss:.4f}  dev loss {dev_loss:.4f}  dev utility {utility:.4f}", flush=True)
    threshold = history[-1]["threshold"]
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    scored = {}
    hospitals = {}
    loaders = {"train": DataLoader(StayDataset("train"), batch_size=batch, shuffle=False, collate_fn=collate), "dev": dev_loader, "test": test_loader}
    for split, loader in loaders.items():
        split_labels = load_label_sequences(split)
        probabilities = predict(model, loader, device)
        overall = operating_metrics(split_labels, probabilities, threshold)
        overall["patients"] = len(split_labels)
        by_site = by_hospital(split_labels, probabilities, hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"oversampled CNN {split}", overall, by_site)
    CNN_RUN.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "regularized causal CNN, septic stays sampled as often as non-septic stays, no pos_weight",
        "checkpoint": "last epoch",
        "threshold": threshold,
        "history": history,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "regularized_cnn_test_utility": 0.3850805973453087,
    }
    torch.save({"state_dict": state, "config": CONFIG, "threshold": threshold}, CNN_RUN / "model.pt")
    (CNN_RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"oversampled CNN dev {scored['dev']['utility']:.4f} test {scored['test']['utility']:.4f}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    train_mlp(device)
    train_cnn(device)


if __name__ == "__main__":
    main()
