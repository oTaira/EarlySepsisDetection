"""CNN with a utility loss, positive-leaning mixup, and a small activation search.

The network predicts the utility gain of raising an alarm on that hour,
u(alarm) - u(quiet), from the official challenge score. Mixup builds a new
stay between a septic stay and another stay and keeps the septic stay's gain,
so the new example stays a positive-class target. It does not copy the septic
stay. The activation is chosen on dev utility. Test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from clinical_features import N_CLINICAL, ROOT as CLINICAL, build
from dataset import N_FEATURES, N_PHYSIO, StayDataset, collate, load_label_sequences, prepare
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import operating_metrics, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold, hour_utility

SEED = 230
EPOCHS = 10
NOISE_STD = 0.1
ACTIVATIONS = ("relu", "silu", "gelu", "mish", "swin_silu")
THRESHOLDS = np.linspace(-1.0, 1.5, 251)
RUN = Path(__file__).resolve().parent / "runs" / "cnn_utility"


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name == "swin_silu":
        return SWInSiLU(0.05, 0.5)
    raise ValueError(name)


class ClinicalStayDataset(StayDataset):
    def __init__(self, split: str):
        super().__init__(split)
        self.extra = np.load(CLINICAL / f"{split}.npy", mmap_mode="r")

    def __getitem__(self, index: int):
        features, labels = super().__getitem__(index)
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        clinical = torch.from_numpy(np.ascontiguousarray(self.extra[start:end].T))
        return torch.cat((features, clinical), dim=0), labels


def augment(features: torch.Tensor) -> torch.Tensor:
    observed = features[:, N_PHYSIO : 2 * N_PHYSIO]
    noise = torch.randn_like(features[:, :N_PHYSIO]) * NOISE_STD
    out = features.clone()
    out[:, :N_PHYSIO] = features[:, :N_PHYSIO] + noise * observed
    return out


def utility_targets(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Utility gained by alarming, on the real prefix of each stay."""
    gain = torch.zeros_like(labels)
    label_rows = labels.detach().cpu().numpy()
    valid = mask.detach().cpu().numpy()
    for index in range(labels.shape[0]):
        length = int(valid[index].sum())
        alarm, quiet = hour_utility(label_rows[index, :length])
        gain[index, :length] = torch.from_numpy(np.asarray(alarm - quiet, dtype=np.float32))
    return gain


def utility_loss(scores, gain, mask) -> torch.Tensor:
    return F.smooth_l1_loss(scores[mask], gain[mask])


def mix_septic(features, mask, gain, labels):
    """New stays leaned toward a septic stay, labeled with that stay's utility gain."""
    septic = ((labels > 0.5) & mask).any(dim=1)
    origins = septic.nonzero(as_tuple=False).flatten()
    if origins.numel() == 0:
        return None
    partners = torch.randint(0, features.shape[0], (origins.numel(),), device=features.device)
    weights = torch.empty(origins.numel(), device=features.device).uniform_(0.5, 0.9)
    pieces = []
    targets = []
    lengths = []
    for origin, partner, weight in zip(origins.tolist(), partners.tolist(), weights.tolist()):
        left = int(mask[origin].sum())
        right = int(mask[partner].sum())
        span = min(left, right)
        windows = [(slice(0, span), slice(0, span))]
        if left != right:
            windows.append((slice(left - span, left), slice(right - span, right)))
        for source, other in windows:
            pieces.append(weight * features[origin, :, source] + (1.0 - weight) * features[partner, :, other])
            targets.append(gain[origin, source])
            lengths.append(span)
    width = features.shape[1]
    longest = max(lengths)
    mixed = features.new_zeros(len(pieces), width, longest)
    mixed_gain = gain.new_zeros(len(pieces), longest)
    mixed_mask = torch.zeros(len(pieces), longest, dtype=torch.bool, device=features.device)
    for index, (piece, target, length) in enumerate(zip(pieces, targets, lengths)):
        mixed[index, :, :length] = piece
        mixed_gain[index, :length] = target
        mixed_mask[index, :length] = True
    return mixed, mixed_gain, mixed_mask


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for features, labels, mask in loader:
        features = augment(features.to(device))
        labels = labels.to(device)
        mask = mask.to(device)
        gain = utility_targets(labels, mask).to(device)
        optimizer.zero_grad(set_to_none=True)
        real = utility_loss(model(features), gain, mask)
        mixed = mix_septic(features, mask, gain, labels)
        if mixed is None:
            loss = real
        else:
            mixed_features, mixed_gain, mixed_mask = mixed
            synthetic = utility_loss(model(mixed_features), mixed_gain, mixed_mask)
            loss = 0.5 * real + 0.5 * synthetic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        hours = int(mask.sum().item())
        total += float(loss.detach()) * hours
        seen += hours
    return total / max(seen, 1)


@torch.no_grad()
def predict_scores(model, loader, device) -> list:
    model.eval()
    sequences = []
    for features, _labels, mask in loader:
        scores = model(features.to(device)).cpu().numpy()
        valid = mask.numpy()
        for index in range(scores.shape[0]):
            length = int(valid[index].sum())
            sequences.append(scores[index, :length].astype("float64"))
    return sequences


def evaluate(model, split: str, threshold: float, device):
    loader = DataLoader(ClinicalStayDataset(split), batch_size=128, shuffle=False, collate_fn=collate)
    labels = load_label_sequences(split)
    scores = predict_scores(model, loader, device)
    overall = operating_metrics(labels, scores, threshold)
    overall["patients"] = len(labels)
    hospitals = by_hospital(labels, scores, hospital_names(split), threshold)
    return overall, hospitals


def train_activation(name: str, loaders, dev_labels, device) -> dict:
    set_seed(SEED)
    model = SimpleCausalCNN(
        N_FEATURES + N_CLINICAL,
        hidden=64,
        kernel_size=9,
        n_layers=3,
        dropout=0.4,
        activation=activation_module(name),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00022868030122031387, weight_decay=1e-3)
    train_loader, dev_loader = loaders
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        dev_scores = predict_scores(model, dev_loader, device)
        threshold, utility = choose_threshold(dev_labels, dev_scores, THRESHOLDS)
        history.append({"epoch": epoch, "train_loss": loss, "dev_utility": utility, "threshold": threshold})
        print(
            f"  epoch {epoch:02d}  utility loss {loss:.4f}  dev utility {utility:.4f}  threshold {threshold:.3f}",
            flush=True,
        )
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "activation": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "dev_utility": history[-1]["dev_utility"],
        "threshold": history[-1]["threshold"],
        "state": state,
    }


def main() -> None:
    prepare()
    build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}. Utility-gain loss, septic mixup, {len(ACTIVATIONS)} activations.", flush=True)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        ClinicalStayDataset("train"), batch_size=128, shuffle=True, collate_fn=collate, generator=generator
    )
    dev_loader = DataLoader(ClinicalStayDataset("dev"), batch_size=128, shuffle=False, collate_fn=collate)
    dev_labels = load_label_sequences("dev")
    RUN.mkdir(parents=True, exist_ok=True)
    trials = []
    best = None
    best_state = None
    for name in ACTIVATIONS:
        print(f"\n{name}", flush=True)
        generator.manual_seed(SEED)
        result = train_activation(name, (train_loader, dev_loader), dev_labels, device)
        state = result.pop("state")
        trials.append(result)
        (RUN / "search.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
        print(f"{name} epoch {EPOCHS} dev utility {result['dev_utility']:.4f}", flush=True)
        if best is None or result["dev_utility"] > best["dev_utility"]:
            best = result
            best_state = state
    model = SimpleCausalCNN(
        N_FEATURES + N_CLINICAL,
        hidden=64,
        kernel_size=9,
        n_layers=3,
        dropout=0.4,
        activation=activation_module(best["activation"]),
    ).to(device)
    model.load_state_dict(best_state)
    threshold = best["threshold"]
    scored = {}
    hospitals = {}
    for split in ("train", "dev", "test"):
        overall, by_site = evaluate(model, split, threshold, device)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"{best['activation']} {split}", overall, by_site)
    payload = {
        "model": "3-layer causal CNN, utility loss, septic mixup, six clinical summaries",
        "clinical_features": ["shock_z", "shock_known", "partial_sofa", "qsofa", "vital_rate_24", "lab_rate_24"],
        "mixup": "features of each septic stay mixed with another stay, lambda uniform on [0.5, 0.9]; the septic stay's utility gain is kept; half the loss",
        "loss": "smooth L1 on the utility gain of an alarm, u(alarm) - u(quiet)",
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint": "last epoch of the activation with the highest dev utility",
        "trials": trials,
        "selected_activation": best["activation"],
        "threshold": threshold,
        "train": scored["train"],
        "dev": scored["dev"],
        "test": scored["test"],
        "hospitals": hospitals,
        "regularized_cnn_test_utility": 0.385053,
    }
    torch.save(
        {"state_dict": best_state, "activation": best["activation"], "threshold": threshold},
        RUN / "model.pt",
    )
    (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nSelected {best['activation']}. Dev {scored['dev']['utility']:.4f}. Test {scored['test']['utility']:.4f}.",
        flush=True,
    )
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
