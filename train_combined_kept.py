"""CNN and LSTM on the columns the lab-window trees kept.

The gain screen was fit on training hours only. Both models use the same
settings as the run on all 591 columns. The best dev epoch is kept, and
test is scored once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import load_label_sequences, prepare
from model import SWInSiLU, CausalLSTM, SimpleCausalCNN
from train_combined import CNN, LSTM, CombinedStayDataset, lab_matrix, run_model, standardization
from train_separatrix_mlp import cached_matrix
from winner_data import training_means

import train_combined

RUN = Path(__file__).resolve().parent / "runs" / "combined_kept"
SOURCE = Path(__file__).resolve().parent / "runs" / "separatrix_labs" / "metrics.json"


class KeptColumns(Dataset):
    def __init__(self, base: CombinedStayDataset, keep: np.ndarray):
        self.base = base
        self.offsets = base.offsets
        self.keep = keep

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        features, labels = self.base[index]
        return features[self.keep], labels


def main() -> None:
    torch.set_float32_matmul_precision("high")
    meta = prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep = np.asarray(json.loads(SOURCE.read_text(encoding="utf-8"))["kept_columns"], dtype=np.int64)
    print(f"device {device}  kept columns {keep.size}", flush=True)
    means = training_means()
    separatrix = {split: cached_matrix(split, means) for split in ("train", "dev", "test")}
    labs = {split: lab_matrix(split, means) for split in ("train", "dev", "test")}
    mu, sd = standardization(separatrix["train"], labs["train"])
    full = {split: CombinedStayDataset(split, separatrix[split], labs[split], mu, sd) for split in ("train", "dev", "test")}
    splits = {split: KeptColumns(full[split], keep) for split in full}
    dev_labels = load_label_sequences("dev")
    RUN.mkdir(parents=True, exist_ok=True)
    train_combined.RUN = RUN
    cnn = SimpleCausalCNN(
        int(keep.size),
        hidden=CNN["hidden"],
        kernel_size=CNN["kernel_size"],
        n_layers=CNN["n_layers"],
        dropout=CNN["dropout"],
        activation=SWInSiLU(0.05, 0.5),
    )
    lstm = CausalLSTM(int(keep.size), LSTM["hidden"], LSTM["n_layers"], LSTM["dropout"])
    results = {
        "cnn": run_model("cnn", cnn, CNN, splits, dev_labels, float(meta["pos_weight"]), device, cosine=False),
        "lstm": run_model("lstm", lstm, LSTM, splits, dev_labels, float(meta["pos_weight"]), device, cosine=True),
    }
    payload = {
        "features": "223 columns kept by the Separatrix gain screen on the combined matrix",
        "kept_columns": keep.tolist(),
        "channels": int(keep.size),
        "checkpoint": "epoch with the highest dev utility",
        "full_matrix_cnn_test_utility": 0.3484,
        "full_matrix_lstm_test_utility": 0.3928,
        "separatrix_labs_test_utility": 0.5169,
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
