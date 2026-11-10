"""Compare three graphs with one fixed GAT.

The network is the best correlation-search trial. The graphs are self-loops
only, the 91-edge training correlation graph, and a hand-built clinical graph.
Each graph is trained on all hospitals, then again on hospital A alone.
Epoch and alarm threshold are chosen on dev (dev hospital A for the transfer
runs). Test, including hospital B, is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import PHYSIO, StayDataset, collate, load_label_sequences, prepare
from search_gat import LengthBatchSampler, build_model, predict_ordered, train_epoch
from train_cnn_baseline import operating_metrics, set_seed
from train_gat import correlation_matrix, edges_for
from train_lstm import by_hospital, hospital_names, print_scores
from utility import choose_threshold

RUN = Path(__file__).resolve().parent / "runs" / "graph_compare"
SEED = 230
EPOCHS = 10
CONFIG = {
    "hidden": 64,
    "heads": 4,
    "n_layers": 2,
    "dropout": 0.307,
    "lr": 0.0017077672618510144,
    "weight_decay": 0.0,
    "batch_size": 64,
    "noise_std": 0.0,
    "pos_weight_scale": 1.0,
    "negative_slope": 0.2,
    "pool": "mean",
    "activation": "silu",
}
CLINICAL_GROUPS = {
    "circulation": [
        ("HR", "SBP"), ("HR", "MAP"), ("HR", "DBP"),
        ("SBP", "MAP"), ("SBP", "DBP"), ("MAP", "DBP"),
        ("TroponinI", "HR"), ("Temp", "HR"),
    ],
    "respiration": [
        ("O2Sat", "SaO2"), ("O2Sat", "FiO2"), ("SaO2", "FiO2"),
        ("O2Sat", "Resp"), ("SaO2", "Resp"), ("Temp", "Resp"),
        ("Resp", "EtCO2"), ("Resp", "PaCO2"), ("EtCO2", "PaCO2"),
        ("pH", "PaCO2"), ("pH", "HCO3"), ("pH", "BaseExcess"),
        ("HCO3", "BaseExcess"), ("HCO3", "PaCO2"),
        ("Lactate", "pH"), ("Lactate", "BaseExcess"), ("Lactate", "HCO3"),
    ],
    "kidney": [
        ("BUN", "Creatinine"), ("Creatinine", "Potassium"),
        ("Creatinine", "Phosphate"), ("BUN", "Phosphate"),
    ],
    "liver": [
        ("AST", "Bilirubin_total"), ("AST", "Bilirubin_direct"),
        ("AST", "Alkalinephos"), ("Bilirubin_total", "Bilirubin_direct"),
        ("Alkalinephos", "Bilirubin_total"),
    ],
    "blood": [
        ("Hct", "Hgb"), ("Platelets", "Fibrinogen"), ("Platelets", "PTT"),
        ("WBC", "Temp"), ("Fibrinogen", "PTT"),
    ],
    "electrolytes": [
        ("Calcium", "Magnesium"), ("Potassium", "Magnesium"), ("Chloride", "HCO3"),
    ],
    "shock": [
        ("HR", "Lactate"), ("MAP", "Lactate"), ("SBP", "Lactate"),
        ("MAP", "Creatinine"), ("Lactate", "Creatinine"),
    ],
}


class IndexedStayDataset(Dataset):
    """A hospital subset with offsets measured on the subset itself."""

    def __init__(self, split: str, indices: np.ndarray):
        self.base = StayDataset(split)
        self.indices = np.asarray(indices, dtype=np.int64)
        lengths = np.diff(self.base.offsets)[self.indices]
        self.offsets = np.zeros(self.indices.size + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(lengths)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        return self.base[int(self.indices[index])]


def clinical_edges() -> torch.Tensor:
    index = {name: i for i, name in enumerate(PHYSIO)}
    src: list[int] = []
    dst: list[int] = []
    seen: set[tuple[int, int]] = set()
    for pairs in CLINICAL_GROUPS.values():
        for left, right in pairs:
            i, j = index[left], index[right]
            key = (min(i, j), max(i, j))
            if key in seen or i == j:
                raise RuntimeError(f"repeated clinical edge {left}-{right}")
            seen.add(key)
            src.extend((i, j))
            dst.extend((j, i))
    return torch.tensor([src, dst], dtype=torch.long)


def hospital_indices(split: str, hospital: str) -> np.ndarray:
    names = hospital_names(split)
    return np.array([i for i, name in enumerate(names) if name == hospital], dtype=np.int64)


def subset_labels(split: str, indices: np.ndarray) -> list[np.ndarray]:
    labels = load_label_sequences(split)
    return [labels[int(i)] for i in indices]


def positive_weight(labels: list[np.ndarray]) -> float:
    positive = sum(int(row.sum()) for row in labels)
    hours = sum(int(row.size) for row in labels)
    return (hours - positive) / positive


def score(model, dataset, labels, names, threshold, edge_index, device):
    probabilities = predict_ordered(model, dataset, 128, edge_index, device)
    overall = operating_metrics(labels, probabilities, threshold)
    overall["patients"] = len(labels)
    sites = by_hospital(labels, probabilities, names, threshold)
    return overall, sites


def train_one(name, edge_index, train_set, dev_set, dev_labels, pos_weight_value, device) -> dict:
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    sampler = LengthBatchSampler(np.diff(train_set.offsets), CONFIG["batch_size"], generator)
    loader = DataLoader(
        train_set, batch_sampler=sampler, collate_fn=collate, pin_memory=device.type == "cuda"
    )
    model = build_model(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    pos_weight = torch.tensor([pos_weight_value], device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    best_utility = -np.inf
    best_epoch = 0
    best_state = None
    best_threshold = None
    history = []
    print(
        f"\n{name}  edges {int(edge_index.shape[1] // 2)}  "
        f"train patients {len(train_set)}  parameters {parameters}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, pos_weight, edge_index, CONFIG["noise_std"], device)
        probabilities = predict_ordered(model, dev_set, 128, edge_index, device)
        threshold, utility = choose_threshold(dev_labels, probabilities)
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
            best_threshold = float(threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    del optimizer
    return {
        "model": model,
        "parameters": parameters,
        "undirected_edges": int(edge_index.shape[1] // 2),
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "history": history,
    }


def main() -> None:
    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    corr, _support = correlation_matrix()
    correlation = edges_for(corr, 0.15)
    if int(correlation.shape[1] // 2) != 91:
        raise RuntimeError(f"correlation graph has {int(correlation.shape[1] // 2)} edges, expected 91")
    graphs = {
        "self": torch.zeros(2, 0, dtype=torch.long),
        "correlation": correlation,
        "clinical": clinical_edges(),
    }
    print(
        "graphs "
        + " ".join(f"{name}={int(edges.shape[1] // 2)}" for name, edges in graphs.items()),
        flush=True,
    )
    full = {split: StayDataset(split) for split in ("train", "dev", "test")}
    labels = {split: load_label_sequences(split) for split in ("train", "dev", "test")}
    names = {split: hospital_names(split) for split in ("train", "dev", "test")}
    train_a = hospital_indices("train", "A")
    dev_a = hospital_indices("dev", "A")
    transfer_train = IndexedStayDataset("train", train_a)
    transfer_dev = IndexedStayDataset("dev", dev_a)
    transfer_labels = subset_labels("dev", dev_a)
    regimes = {
        "all_hospitals": (full["train"], full["dev"], labels["dev"], float(meta["pos_weight"])),
        "train_A": (transfer_train, transfer_dev, transfer_labels, positive_weight(subset_labels("train", train_a))),
    }
    RUN.mkdir(parents=True, exist_ok=True)
    results = {}
    for regime, (train_set, dev_set, dev_labels, pos_weight_value) in regimes.items():
        print(f"\nregime {regime}  pos_weight {pos_weight_value:.2f}", flush=True)
        results[regime] = {}
        for name, edges in graphs.items():
            trained = train_one(f"{regime}/{name}", edges, train_set, dev_set, dev_labels, pos_weight_value, device)
            model = trained.pop("model")
            record = dict(trained)
            record["splits"] = {}
            record["hospitals"] = {}
            for split in ("dev", "test"):
                overall, sites = score(model, full[split], labels[split], names[split], record["threshold"], edges, device)
                record["splits"][split] = overall
                record["hospitals"][split] = sites
                print_scores(f"{regime}/{name} {split}", overall, sites)
            if regime == "train_A":
                record["selection_patients"] = len(dev_labels)
                record["selection_utility"] = record["history"][record["best_epoch"] - 1]["dev_utility"]
            torch.save(
                {"state_dict": {key: value.cpu() for key, value in model.state_dict().items()}, "config": CONFIG},
                RUN / f"{regime}_{name}.pt",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results[regime][name] = record
            payload = {
                "config": CONFIG,
                "seed": SEED,
                "epochs": EPOCHS,
                "checkpoint": "epoch with the highest dev utility",
                "self_loops": "PhysioGAT always keeps a self-loop on each variable",
                "clinical_groups": CLINICAL_GROUPS,
                "results": results,
            }
            (RUN / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\ncomparison", flush=True)
    for regime, rows in results.items():
        for name, record in rows.items():
            test = record["splits"]["test"]
            sites = record["hospitals"]["test"]
            print(
                f"{regime:16} {name:12} epoch {record['best_epoch']:2d}  "
                f"dev {record['splits']['dev']['utility']:.4f}  test {test['utility']:.4f}  "
                f"A {sites['A']['utility']:.4f}  B {sites['B']['utility']:.4f}",
                flush=True,
            )
    print(f"wrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
