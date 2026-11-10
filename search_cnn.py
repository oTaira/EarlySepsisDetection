"""Random search over the causal CNN, with the SWIn-SiLU activation.

Each trial is scored by dev utility. The alarm threshold is chosen on dev.
Test is scored once, after the search, for the single best trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import N_FEATURES, StayDataset, collate, load_label_sequences, prepare
from model import SWInSiLU, SimpleCausalCNN
from train_cnn_baseline import (
    MAX_EPOCHS,
    PATIENCE,
    operating_metrics,
    predict,
    set_seed,
    train_epoch,
)
from utility import choose_threshold

SEED = 230
N_TRIALS = 20
RUN = Path(__file__).resolve().parent / "runs" / "cnn_search"


def sample_config(rng: np.random.Generator) -> dict:
    return {
        "hidden": int(rng.choice([32, 48, 64, 96, 128])),
        "kernel_size": int(rng.choice([3, 5, 7, 8, 9, 11])),
        "n_layers": int(rng.choice([2, 3, 4, 5])),
        "dropout": round(float(rng.uniform(0.0, 0.4)), 3),
        "lr": float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3)))),
        "weight_decay": float(rng.choice([0.0, 1e-6, 1e-5, 1e-4, 1e-3])),
        "batch_size": int(rng.choice([64, 128, 256])),
        "swin_a": float(rng.choice([0.05, 0.1, 0.2])),
        "swin_omega": float(rng.choice([0.5, 1.0, 2.0])),
    }


def run_trial(trial: int, config: dict, dev_labels, pos_weight_value: float, device) -> dict:
    set_seed(SEED + trial)
    generator = torch.Generator()
    generator.manual_seed(SEED + trial)
    train_loader = DataLoader(
        StayDataset("train"),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    dev_loader = DataLoader(
        StayDataset("dev"), batch_size=config["batch_size"], shuffle=False, collate_fn=collate
    )
    activation = SWInSiLU(config["swin_a"], config["swin_omega"])
    model = SimpleCausalCNN(
        N_FEATURES,
        hidden=config["hidden"],
        kernel_size=config["kernel_size"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        activation=activation,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    pos_weight = torch.tensor([pos_weight_value], device=device)
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_dev = None
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        dev_probs = predict(model, dev_loader, device)
        threshold, utility = choose_threshold(dev_labels, dev_probs)
        history.append({"epoch": epoch, "train_loss": loss, "dev_utility": utility, "threshold": threshold})
        print(
            f"  epoch {epoch:02d}  train loss {loss:.4f}  dev utility {utility:.4f}",
            flush=True,
        )
        if utility > best_utility:
            best_utility = utility
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_dev = operating_metrics(dev_labels, dev_probs, threshold)
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return {
        "trial": trial,
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "receptive_field_hours": model.receptive_field,
        "best_epoch": best_epoch,
        "dev": best_dev,
        "history": history,
        "state": best_state,
    }


def main() -> None:
    set_seed(SEED)
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}. {N_TRIALS} trials, SWIn-SiLU, selection on dev utility.", flush=True)
    dev_labels = load_label_sequences("dev")
    rng = np.random.default_rng(SEED)
    RUN.mkdir(parents=True, exist_ok=True)

    best = None
    records = []
    for trial in range(1, N_TRIALS + 1):
        config = sample_config(rng)
        print(f"\ntrial {trial}/{N_TRIALS}  {json.dumps(config)}", flush=True)
        result = run_trial(trial, config, dev_labels, meta["pos_weight"], device)
        state = result.pop("state")
        records.append(result)
        (RUN / "search.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        utility = result["dev"]["utility"]
        print(
            f"trial {trial} best epoch {result['best_epoch']}  dev utility {utility:.4f}  "
            f"params {result['parameters']}",
            flush=True,
        )
        if best is None or utility > best["dev"]["utility"]:
            best = result
            torch.save(
                {"state_dict": state, "config": config, "dev": result["dev"]},
                RUN / "best.pt",
            )

    test_labels = load_label_sequences("test")
    saved = torch.load(RUN / "best.pt", map_location=device, weights_only=False)
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
    test_loader = DataLoader(
        StayDataset("test"), batch_size=config["batch_size"], shuffle=False, collate_fn=collate
    )
    threshold = saved["dev"]["threshold"]
    test_metrics = operating_metrics(test_labels, predict(model, test_loader, device), threshold)
    summary = {
        "activation": "SWIn-SiLU",
        "activation_reference": "Ran et al., Neural Networks, 28 October 2026",
        "trials": N_TRIALS,
        "selection": "highest dev utility",
        "relu_baseline_dev_utility": 0.37800320049959096,
        "best": best,
        "test": test_metrics,
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("", flush=True)
    print(f"best trial {best['trial']}  dev {best['dev']['utility']:.4f}  test {test_metrics['utility']:.4f}", flush=True)
    print(f"config {json.dumps(config)}", flush=True)
    print(f"wrote {RUN / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
