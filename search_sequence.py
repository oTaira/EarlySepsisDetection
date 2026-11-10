"""Search a causal LSTM and a causal attention model, then blend the winners.

Every trial can use a constant learning rate or a cosine decay. The epoch
with the best dev utility is kept. The blend weight and the alarm threshold
are chosen on dev. Test is scored once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import N_FEATURES, N_PHYSIO, StayDataset, collate, load_label_sequences, prepare
from model import CausalAttentionNet, CausalLSTM
from search_cnn_broad import LengthBatchSampler, predict_ordered
from train_cnn_baseline import operating_metrics, set_seed
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold

SEED = 232
MAX_EPOCHS = 10
TRIALS_PER_FAMILY = 24
BUDGET_HOURS = 3
RUN = Path(__file__).resolve().parent / "runs" / "sequence_search"


def sample_config(rng: np.random.Generator, family: str) -> dict:
    config = {
        "family": family,
        "dropout": round(float(rng.uniform(0.1, 0.5)), 3),
        "lr": float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-3)))),
        "weight_decay": float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        "batch_size": int(rng.choice([64, 128])),
        "noise_std": float(rng.choice([0.0, 0.05, 0.1])),
        "schedule": str(rng.choice(["cosine", "constant"])),
    }
    if family == "lstm":
        config.update(
            hidden=int(rng.choice([32, 64, 96, 128])),
            n_layers=int(rng.choice([1, 2, 3])),
        )
    else:
        hidden = int(rng.choice([32, 48, 64, 96]))
        config.update(
            hidden=hidden,
            heads=int(rng.choice([2, 4])),
            n_layers=int(rng.choice([1, 2])),
            stem_kernel=int(rng.choice([3, 5, 9])),
            ffn_ratio=int(rng.choice([1, 2, 4])),
            activation=str(rng.choice(["relu", "gelu", "silu"])),
        )
    return config


def build_model(config: dict) -> torch.nn.Module:
    if config["family"] == "lstm":
        return CausalLSTM(N_FEATURES, config["hidden"], config["n_layers"], config["dropout"])
    activation = {"relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "silu": torch.nn.SiLU()}[config["activation"]]
    return CausalAttentionNet(
        N_FEATURES,
        hidden=config["hidden"],
        heads=config["heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        activation=activation,
        stem_kernel=config["stem_kernel"],
        ffn_ratio=config["ffn_ratio"],
    )


def augment(features: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0:
        return features
    observed = features[:, N_PHYSIO : 2 * N_PHYSIO]
    out = features.clone()
    out[:, :N_PHYSIO] = features[:, :N_PHYSIO] + torch.randn_like(features[:, :N_PHYSIO]) * noise_std * observed
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


def run_trial(trial, config, train_set, dev_set, dev_labels, pos_weight_value, device) -> dict:
    set_seed(SEED + trial)
    generator = torch.Generator()
    generator.manual_seed(SEED + trial)
    sampler = LengthBatchSampler(np.diff(train_set.offsets), config["batch_size"], generator)
    train_loader = DataLoader(
        train_set, batch_sampler=sampler, collate_fn=collate, pin_memory=device.type == "cuda"
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    schedule = None
    if config["schedule"] == "cosine":
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=config["lr"] / 20)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_dev = None
    history = []
    parameters = sum(parameter.numel() for parameter in model.parameters())
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, pos_weight, config["noise_std"], device)
            if schedule is not None:
                schedule.step()
            if not np.isfinite(loss):
                print(f"  epoch {epoch:02d}  non-finite loss", flush=True)
                break
            dev_probabilities = predict_ordered(model, dev_set, max(config["batch_size"], 64), device)
            threshold, utility = choose_threshold(dev_labels, dev_probabilities)
            history.append(
                {"epoch": epoch, "train_loss": float(loss), "dev_utility": float(utility), "threshold": float(threshold)}
            )
            print(f"  epoch {epoch:02d}  loss {loss:.4f}  dev utility {utility:.4f}", flush=True)
            if utility > best_utility:
                best_utility = float(utility)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_dev = operating_metrics(dev_labels, dev_probabilities, threshold)
                best_dev["patients"] = len(dev_labels)
    finally:
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "trial": trial,
        "config": config,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "dev": best_dev,
        "history": history,
        "state": best_state,
    }


def blend(left, right, weight: float, kind: str) -> list[np.ndarray]:
    mixed = []
    for first, second in zip(left, right):
        first = np.clip(np.asarray(first, dtype=np.float64), 1e-6, 1 - 1e-6)
        second = np.clip(np.asarray(second, dtype=np.float64), 1e-6, 1 - 1e-6)
        if kind == "probability":
            mixed.append(weight * first + (1.0 - weight) * second)
        else:
            combined = weight * np.log(first / (1.0 - first)) + (1.0 - weight) * np.log(second / (1.0 - second))
            mixed.append(1.0 / (1.0 + np.exp(-combined)))
    return mixed


def score_model(model, splits, threshold, device):
    scored = {}
    hospitals = {}
    probabilities = {}
    for split, dataset in splits.items():
        probabilities[split] = predict_ordered(model, dataset, 128, device)
        labels = load_label_sequences(split)
        overall = operating_metrics(labels, probabilities[split], threshold)
        overall["patients"] = len(labels)
        by_site = by_hospital(labels, probabilities[split], hospital_names(split), threshold)
        scored[split] = overall
        hospitals[split] = by_site
        print_scores(f"{split}", overall, by_site)
    return scored, hospitals, probabilities


def main() -> None:
    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device {device}. {TRIALS_PER_FAMILY} LSTM trials and {TRIALS_PER_FAMILY} attention trials, "
        f"{MAX_EPOCHS} epochs, cosine or constant learning rate.",
        flush=True,
    )
    train_set = StayDataset("train")
    dev_set = StayDataset("dev")
    test_set = StayDataset("test")
    dev_labels = load_label_sequences("dev")
    rng = np.random.default_rng(SEED)
    RUN.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    winners = {}
    trial = 0
    for family in ("lstm", "attention"):
        for index in range(1, TRIALS_PER_FAMILY + 1):
            trial += 1
            if trial > 1 and time.perf_counter() - started >= BUDGET_HOURS * 3600:
                print("budget reached", flush=True)
                break
            config = sample_config(rng, family)
            print(f"\n{family} {index}/{TRIALS_PER_FAMILY}  {json.dumps(config)}", flush=True)
            try:
                result = run_trial(
                    trial, config, train_set, dev_set, dev_labels, float(meta["pos_weight"]), device
                )
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"trial skipped: out of memory", flush=True)
                records.append({"trial": trial, "config": config, "error": "out of memory"})
                (RUN / "search.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                continue
            state = result.pop("state")
            records.append(result)
            (RUN / "search.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            if result["dev"] is None:
                continue
            utility = result["dev"]["utility"]
            print(
                f"{family} trial {index} epoch {result['best_epoch']}  dev utility {utility:.4f}  "
                f"params {result['parameters']}",
                flush=True,
            )
            if family not in winners or utility > winners[family]["dev"]["utility"]:
                winners[family] = result
                torch.save({"state_dict": state, "config": config, "dev": result["dev"]}, RUN / f"best_{family}.pt")
                print(f"new best {family} {utility:.4f}", flush=True)
        else:
            continue
        break

    if len(winners) < 2:
        print("need a finished LSTM and a finished attention model before blending", flush=True)
        return

    splits = {"train": train_set, "dev": dev_set, "test": test_set}
    stored = {}
    for family in ("lstm", "attention"):
        saved = torch.load(RUN / f"best_{family}.pt", map_location=device, weights_only=False)
        model = build_model(saved["config"]).to(device)
        model.load_state_dict(saved["state_dict"])
        print(f"\nbest {family}", flush=True)
        scored, hospitals, probabilities = score_model(model, splits, saved["dev"]["threshold"], device)
        stored[family] = {
            "config": saved["config"],
            "dev_selection": saved["dev"],
            "scored": scored,
            "hospitals": hospitals,
            "probabilities": probabilities,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_mix = None
    grid = []
    for kind in ("probability", "logit"):
        for weight in np.linspace(0.0, 1.0, 21):
            mixed = blend(stored["lstm"]["probabilities"]["dev"], stored["attention"]["probabilities"]["dev"], float(weight), kind)
            threshold, utility = choose_threshold(dev_labels, mixed)
            row = {"kind": kind, "lstm_weight": float(weight), "threshold": float(threshold), "dev_utility": float(utility)}
            grid.append(row)
            if best_mix is None or utility > best_mix["dev_utility"]:
                best_mix = row
    print(
        f"selected {best_mix['kind']} lstm weight {best_mix['lstm_weight']:.2f} "
        f"threshold {best_mix['threshold']:.3f} dev {best_mix['dev_utility']:.4f}",
        flush=True,
    )
    ensemble = {}
    ensemble_hospitals = {}
    for split in ("train", "dev", "test"):
        mixed = blend(
            stored["lstm"]["probabilities"][split],
            stored["attention"]["probabilities"][split],
            best_mix["lstm_weight"],
            best_mix["kind"],
        )
        labels = load_label_sequences(split)
        overall = operating_metrics(labels, mixed, best_mix["threshold"])
        overall["patients"] = len(labels)
        by_site = by_hospital(labels, mixed, hospital_names(split), best_mix["threshold"])
        ensemble[split] = overall
        ensemble_hospitals[split] = by_site
        print_scores(f"ensemble {split}", overall, by_site)
    payload = {
        "selection": "best dev utility within each family, then blend weight and threshold on dev",
        "lstm": {key: value for key, value in stored["lstm"].items() if key != "probabilities"},
        "attention": {key: value for key, value in stored["attention"].items() if key != "probabilities"},
        "blend": best_mix,
        "dev_grid": grid,
        "ensemble_train": ensemble["train"],
        "ensemble_dev": ensemble["dev"],
        "ensemble_test": ensemble["test"],
        "ensemble_hospitals": ensemble_hospitals,
        "regularized_cnn_test_utility": 0.3850805973453087,
        "cnn_separatrix_ensemble_test_utility": 0.4580,
    }
    (RUN / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"ensemble dev {ensemble['dev']['utility']:.4f} test {ensemble['test']['utility']:.4f}",
        flush=True,
    )
    print(f"wrote {RUN / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
