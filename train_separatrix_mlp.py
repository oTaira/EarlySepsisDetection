"""Train a small MLP on the Separatrix feature matrix.

The columns are the same ones the Separatrix trees kept. Standardization is
fit on the training hours. The alarm threshold is chosen on dev. Test is
scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from xgboost import XGBClassifier

from train_cnn_baseline import set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from train_winners import SEED as WINNER_SEED
from train_winners import _hours, _undersample, flat_labels, score_split
from utility import choose_threshold
from winner_data import PROCESSED, build_separatrix, labels_of, load_raw, sequences, training_means

EPOCHS = 10
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 128
DROPOUT = 0.2
SEED = 230
RUN = Path(__file__).resolve().parent / "runs" / "separatrix_mlp"


class HourMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def cached_matrix(split: str, means: np.ndarray) -> np.ndarray:
    return build_separatrix(split, means)


def kept_columns(x_train: np.ndarray, y_train: np.ndarray) -> list[int]:
    _, _, offsets = load_raw("train")
    rng = np.random.default_rng(WINNER_SEED)
    patients = np.arange(offsets.size - 1)
    rng.shuffle(patients)
    n_screen = max(1, int(0.10 * patients.size))
    screen = _undersample(y_train, _hours(patients[:n_screen], offsets), rng)
    screener = XGBClassifier(
        n_estimators=40, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=WINNER_SEED
    )
    screener.fit(np.asarray(x_train[screen]), y_train[screen], verbose=False)
    gain = screener.get_booster().get_score(importance_type="gain")
    keep = sorted(int(name[1:]) for name in gain if gain[name] > 0)
    if not keep:
        keep = list(range(x_train.shape[1]))
    print(f"kept {len(keep)} of {x_train.shape[1]} Separatrix features", flush=True)
    return keep


def predict_flat(model, matrix, device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], BATCH):
            batch = torch.from_numpy(np.asarray(matrix[start : start + BATCH])).to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}. MLP on Separatrix features.", flush=True)
    means = training_means()
    x_train = cached_matrix("train", means)
    y_train = flat_labels("train")
    mu = np.asarray(x_train, dtype=np.float64).mean(axis=0).astype(np.float32)
    sd = np.asarray(x_train, dtype=np.float64).std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    keep = kept_columns((np.asarray(x_train, dtype=np.float32) - mu) / sd, y_train)
    train = ((np.asarray(x_train, dtype=np.float32) - mu) / sd)[:, keep]
    del x_train
    dev = ((np.asarray(cached_matrix("dev", means), dtype=np.float32) - mu) / sd)[:, keep]
    y_dev = flat_labels("dev")
    positive = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - positive) / positive], device=device)
    model = HourMLP(len(keep)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(SEED)
    history = []
    best_utility = -np.inf
    best_state = None
    dev_labels = labels_of("dev")
    _, _, dev_offsets = load_raw("dev")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = generator.permutation(train.shape[0])
        total = 0.0
        seen = 0
        for start in range(0, order.size, BATCH):
            chosen = order[start : start + BATCH]
            features = torch.from_numpy(train[chosen]).to(device)
            labels = torch.from_numpy(y_train[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(features), labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * chosen.size
            seen += chosen.size
        dev_scores = sequences(predict_flat(model, dev, device), dev_offsets)
        threshold, utility = choose_threshold(dev_labels, dev_scores)
        history.append({"epoch": epoch, "train_loss": total / seen, "dev_utility": utility, "threshold": threshold})
        print(f"epoch {epoch:02d}  loss {total / seen:.4f}  dev utility {utility:.4f}", flush=True)
        if utility > best_utility:
            best_utility = float(utility)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_threshold = float(threshold)
    threshold = best_threshold
    state = best_state
    model.load_state_dict(state)
    scored = {}
    hospitals = {}
    matrices = {"train": train, "dev": dev}
    for split in ("train", "dev", "test"):
        if split not in matrices:
            raw = cached_matrix(split, means)
            matrices[split] = ((np.asarray(raw, dtype=np.float32) - mu) / sd)[:, keep]
        _, _, offsets = load_raw(split)
        probs = sequences(predict_flat(model, matrices[split], device), offsets)
        labels = labels_of(split)
        overall = score_split(labels, probs, threshold)
        overall["patients"] = len(labels)
        by_site = by_hospital(labels, probs, hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"separatrix MLP {split}", overall, by_site)
    payload = {
        "model": "2-layer MLP on the Separatrix features the trees kept",
        "features": len(keep),
        "kept_columns": keep,
        "epochs": EPOCHS,
        "checkpoint": "epoch with the highest dev utility",
        "best_epoch": best_epoch,
        "seed": SEED,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "threshold": threshold,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "separatrix_tree_test_utility": 0.4545682369775407,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "keep": keep, "mu": mu, "sd": sd, "threshold": threshold}, RUN / "model.pt")
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"MLP dev {scored['dev']['utility']:.4f}  test {scored['test']['utility']:.4f}. "
        f"Separatrix tree test 0.4546.",
        flush=True,
    )
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
