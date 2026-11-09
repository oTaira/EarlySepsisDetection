"""Train the five official PhysioNet/CinC 2019 winners on this split.

The alarm threshold is chosen on dev. Test is scored once. Train is scored
with that same frozen threshold, so it is not a separately tuned operating point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

from utility import choose_threshold, hour_utility, normalized_utility, self_check as check_utility
from winner_data import (
    N_PHYSIO,
    TASP_EARLY_COLUMNS,
    _index,
    build_ctl,
    build_separatrix,
    build_sepsyd,
    build_signature,
    build_tasp,
    ensure_raw,
    labels_of,
    load_raw,
    log_columns,
    self_check as check_features,
    sequences,
    shift_labels,
    training_means,
)

SEED = 230
RUN = Path(__file__).resolve().parent / "runs" / "winners"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_xgb(model, x_train, y_train, x_dev=None, y_dev=None):
    if x_dev is None:
        model.fit(x_train, y_train, verbose=False)
    else:
        model.fit(x_train, y_train, eval_set=[(x_dev, y_dev)], verbose=False)
    return model


def flat_labels(split: str) -> np.ndarray:
    return np.concatenate(labels_of(split)).astype(np.int32)


def utility_target(split: str) -> np.ndarray:
    parts = []
    for labels in labels_of(split):
        utility_if_alarm, utility_if_quiet = hour_utility(labels)
        parts.append(utility_if_alarm - utility_if_quiet)
    return np.concatenate(parts).astype(np.float32)


def choose_cutoff(label_seqs, score_seqs, probability: bool) -> float:
    if probability:
        grid = np.linspace(0.0, 1.0, 201)
    else:
        grid = np.unique(np.quantile(np.concatenate(score_seqs), np.linspace(0.0, 1.0, 251)))
    threshold, _ = choose_threshold(label_seqs, score_seqs, grid)
    return threshold


def score_split(label_seqs, score_seqs, threshold: float) -> dict:
    predictions = [np.asarray(scores) >= threshold for scores in score_seqs]
    y = np.concatenate(label_seqs).astype(np.int32)
    scores = np.concatenate([np.asarray(s, dtype=np.float64) for s in score_seqs])
    pred = np.concatenate(predictions)
    positive = max(int((y == 1).sum()), 1)
    true_pos = int(((y == 1) & pred).sum())
    false_alarms = 0
    nonseptic_hours = 0
    for labels, guessed in zip(label_seqs, predictions):
        if not np.any(labels):
            false_alarms += int(np.sum(guessed))
            nonseptic_hours += int(len(labels))
    return {
        "utility": float(normalized_utility(label_seqs, predictions)),
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "hour_sensitivity": float(true_pos / positive),
        "fp_per_100_nonseptic_hours": float(100.0 * false_alarms / nonseptic_hours),
    }


def evaluate_model(score_fn, probability: bool) -> dict:
    dev_labels = labels_of("dev")
    dev_scores = score_fn("dev")
    threshold = choose_cutoff(dev_labels, dev_scores, probability)
    results = {}
    for split, label_seqs, scores in (
        ("dev", dev_labels, dev_scores),
        ("train", labels_of("train"), score_fn("train")),
        ("test", labels_of("test"), score_fn("test")),
    ):
        results[split] = score_split(label_seqs, scores, threshold)
    return results


def positive_weight(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / max(pos, 1)


def train_signature(means) -> dict:
    print("training signature model", flush=True)
    x_train = np.asarray(build_signature("train", means))
    y_train = utility_target("train")
    model = LGBMRegressor(
        n_estimators=160,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        feature_fraction=0.4,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=-1,
        verbose=-1,
        random_state=SEED,
    )
    model.fit(x_train, y_train)

    def scores(split: str):
        if split == "train":
            matrix = x_train
        else:
            matrix = np.asarray(build_signature(split, means))
        flat = model.predict(matrix)
        return sequences(flat, load_raw(split)[2])

    results = evaluate_model(scores, probability=False)
    del x_train
    return results


def sepsyd_means(skewed: list[int]) -> np.ndarray:
    values, _, _ = load_raw("train")
    means = np.zeros(40, dtype=np.float64)
    skewed_set = set(skewed)
    for j in range(40):
        column = np.asarray(values[:, j], dtype=np.float64)
        good = column[np.isfinite(column)]
        if j in skewed_set:
            good = np.log1p(good)
        means[j] = float(good.mean()) if good.size else 0.0
    return means


def train_sepsyd(means) -> dict:
    print("training Sepsyd", flush=True)
    skewed = log_columns()
    print("log-transformed columns:", [__import__("dataset").PHYSIO[j] for j in skewed], flush=True)
    center = sepsyd_means(skewed)
    x_train = build_sepsyd("train", center, skewed)
    y_train = flat_labels("train")
    x_dev = build_sepsyd("dev", center, skewed)
    y_dev = flat_labels("dev")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=40,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=25,
        eval_metric="aucpr",
        random_state=SEED,
    )
    fit_xgb(model, x_train, y_train, x_dev, y_dev)

    def scores(split: str):
        matrix = {"train": x_train, "dev": x_dev}.get(split)
        if matrix is None:
            matrix = build_sepsyd(split, center, skewed)
        flat = model.predict_proba(matrix)[:, 1]
        return sequences(flat, load_raw(split)[2])

    return evaluate_model(scores, probability=True)


def _hours(patient_ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(int(offsets[i]), int(offsets[i + 1])) for i in patient_ids])


def _undersample(y: np.ndarray, candidates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    chosen = candidates[y[candidates] == 1]
    negatives = candidates[y[candidates] == 0]
    take = rng.choice(negatives, size=chosen.size, replace=False)
    sample = np.concatenate([chosen, take])
    rng.shuffle(sample)
    return sample


def train_separatrix(means) -> dict:
    print("training Separatrix", flush=True)
    x_train = build_separatrix("train", means)
    y_train = flat_labels("train")
    mu = x_train.mean(axis=0).astype(np.float32)
    sd = x_train.std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    x_train -= mu
    x_train /= sd
    _, _, offsets = load_raw("train")
    rng = np.random.default_rng(SEED)
    patients = np.arange(offsets.size - 1)
    rng.shuffle(patients)
    n_screen = max(1, int(0.10 * patients.size))
    screen = _undersample(y_train, _hours(patients[:n_screen], offsets), rng)
    screener = XGBClassifier(
        n_estimators=40, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=SEED
    )
    screener.fit(x_train[screen], y_train[screen], verbose=False)
    gain = screener.get_booster().get_score(importance_type="gain")
    # XGBoost names columns f0, f1, ...
    keep = sorted(int(name[1:]) for name in gain if gain[name] > 0)
    if not keep:
        keep = list(range(x_train.shape[1]))
    print(f"Separatrix kept {len(keep)} of 407 features", flush=True)
    rest = np.array_split(patients[n_screen:], 5)
    models = []
    for fold, ids in enumerate(rest):
        sample = _undersample(y_train, _hours(ids, offsets), rng)
        model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(x_train[sample][:, keep], y_train[sample], verbose=False)
        models.append(model)

    def scores(split: str):
        if split == "train":
            matrix = x_train
        else:
            matrix = build_separatrix(split, means)
            matrix -= mu
            matrix /= sd
        clipped = []
        for model in models:
            clipped.append(np.clip(model.predict_proba(matrix[:, keep])[:, 1], 1e-6, 1 - 1e-6))
        flat = np.exp(np.mean(np.log(np.vstack(clipped)), axis=0))
        return sequences(flat, load_raw(split)[2])

    results = evaluate_model(scores, probability=True)
    del x_train
    return results


def train_ctl() -> dict:
    print("training CTL", flush=True)
    x_train = build_ctl("train")
    y_train = np.concatenate(shift_labels(labels_of("train"))).astype(np.int32)
    x_dev = build_ctl("dev")
    y_dev = np.concatenate(shift_labels(labels_of("dev"))).astype(np.int32)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=positive_weight(y_train),
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="aucpr",
        random_state=SEED,
        missing=np.nan,
    )
    fit_xgb(model, x_train, y_train, x_dev, y_dev)

    def scores(split: str):
        matrix = {"train": x_train, "dev": x_dev}.get(split)
        if matrix is None:
            matrix = build_ctl(split)
        flat = model.predict_proba(matrix)[:, 1]
        return sequences(flat, load_raw(split)[2])

    return evaluate_model(scores, probability=True)


class GRUD(nn.Module):
    def __init__(self, n_vars: int, hidden: int, extra: int):
        super().__init__()
        self.decay_x = nn.Linear(n_vars, n_vars)
        self.decay_h = nn.Linear(n_vars, hidden)
        self.cell = nn.GRUCell(n_vars * 2, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + extra, 32), nn.ReLU(), nn.Linear(32, 1))
        self.hidden = hidden

    def forward(self, values, mask, delta, extra, mean):
        batch, steps, _ = values.shape
        hidden = values.new_zeros(batch, self.hidden)
        last = mean.view(1, -1).expand(batch, -1).clone()
        logits = []
        for t in range(steps):
            observed = mask[:, t]
            gamma_x = torch.exp(-F.relu(self.decay_x(delta[:, t])))
            gamma_h = torch.exp(-F.relu(self.decay_h(delta[:, t])))
            decayed = gamma_x * last + (1.0 - gamma_x) * mean
            filled = observed * values[:, t] + (1.0 - observed) * decayed
            last = observed * values[:, t] + (1.0 - observed) * last
            hidden = gamma_h * hidden
            hidden = self.cell(torch.cat([filled, observed], dim=-1), hidden)
            logits.append(self.head(torch.cat([hidden, extra[:, t]], dim=-1)).squeeze(-1))
        return torch.stack(logits, dim=1)


def _time_since(observed: np.ndarray) -> np.ndarray:
    idx = np.where(observed, np.arange(observed.shape[0])[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    clock = np.arange(observed.shape[0])[:, None]
    since = np.where(idx >= 0, clock - idx, clock + 1)
    return np.minimum(since, 48).astype(np.float32)


class LateStayDataset(Dataset):
    def __init__(self, split: str, features: np.ndarray, mean: np.ndarray, train_only_long: bool):
        self.values, self.labels, self.offsets = load_raw(split)
        self.features = features
        self.mean = mean.astype(np.float32)
        self.ids = []
        for i, (a, b) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            if (not train_only_long) or (b - a) >= 50:
                self.ids.append(i)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        patient = self.ids[index]
        a, b = int(self.offsets[patient]), int(self.offsets[patient + 1])
        raw = np.asarray(self.values[a:b, :N_PHYSIO], dtype=np.float32)
        observed = np.isfinite(raw)
        values = np.where(observed, raw, 0.0).astype(np.float32)
        delta = _time_since(observed)
        extra = np.nan_to_num(np.asarray(self.features[a:b, 39:], dtype=np.float32), nan=0.0)
        labels = np.asarray(self.labels[a:b], dtype=np.float32)
        late = np.asarray(self.values[a:b, _index("ICULOS")], dtype=np.float32) >= 50
        return (
            torch.from_numpy(values),
            torch.from_numpy(observed.astype(np.float32)),
            torch.from_numpy(delta),
            torch.from_numpy(extra),
            torch.from_numpy(labels),
            torch.from_numpy(late),
        )


def _pad_batch(batch):
    steps = [item[0].shape[0] for item in batch]
    longest = max(steps)
    out = []
    for position in range(6):
        piece = batch[0][position]
        shape = (len(batch), longest, *piece.shape[1:])
        buf = torch.zeros(shape, dtype=piece.dtype)
        for i, item in enumerate(batch):
            buf[i, : steps[i]] = item[position]
        out.append(buf)
    out.append(torch.tensor(steps))
    return out


def train_tasp(means) -> dict:
    print("training TASP", flush=True)
    matrices = {split: build_tasp(split, means) for split in ("train", "dev", "test")}
    iculos = {
        split: np.asarray(load_raw(split)[0][:, _index("ICULOS")], dtype=np.float32) for split in ("train", "dev", "test")
    }
    y_train = flat_labels("train")
    early = iculos["train"] <= 9
    model_early = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.01,
        num_leaves=70,
        min_child_samples=1000,
        scale_pos_weight=1,
        n_jobs=-1,
        verbose=-1,
        random_state=SEED,
    )
    model_middle = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.01,
        num_leaves=70,
        min_child_samples=1000,
        scale_pos_weight=5,
        n_jobs=-1,
        verbose=-1,
        random_state=SEED,
    )
    dev_early = iculos["dev"] <= 9
    model_early.fit(
        matrices["train"][early][:, TASP_EARLY_COLUMNS],
        y_train[early],
        eval_set=[(matrices["dev"][dev_early][:, TASP_EARLY_COLUMNS], flat_labels("dev")[dev_early])],
        callbacks=[early_stopping(40, verbose=False)],
    )
    model_middle.fit(
        matrices["train"],
        y_train,
        eval_set=[(matrices["dev"], flat_labels("dev"))],
        callbacks=[early_stopping(40, verbose=False)],
    )
    phys_mean = torch.tensor(means[:N_PHYSIO], dtype=torch.float32, device=DEVICE)
    late = GRUD(N_PHYSIO, 39, matrices["train"].shape[1] - 39).to(DEVICE)
    late_labels = flat_labels("train")[iculos["train"] >= 50]
    weight = torch.tensor([positive_weight(late_labels)], device=DEVICE)
    optimizer = torch.optim.Adam(late.parameters(), lr=5e-4)
    train_set = LateStayDataset("train", matrices["train"], means[:N_PHYSIO], train_only_long=True)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=_pad_batch, generator=generator)
    best_state = None
    best_utility = -np.inf
    for epoch in range(1, 7):
        late.train()
        total = 0.0
        hours = 0
        for values, mask, delta, extra, labels, late_mask, lengths in loader:
            values, mask, delta, extra = (t.to(DEVICE) for t in (values, mask, delta, extra))
            labels, late_mask = labels.to(DEVICE), late_mask.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = late(values, mask, delta, extra, phys_mean)
            selected = late_mask.bool()
            if int(selected.sum()) == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(logits[selected], labels[selected], pos_weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(late.parameters(), 1.0)
            optimizer.step()
            n = int(selected.sum())
            total += float(loss.detach()) * n
            hours += n
        dev_probs = _tasp_probabilities(matrices, iculos, model_early, model_middle, late, phys_mean, "dev")
        _, utility = choose_threshold(labels_of("dev"), dev_probs)
        print(f"TASP epoch {epoch} late loss {total / max(hours, 1):.4f} dev utility {utility:.4f}", flush=True)
        if utility > best_utility:
            best_utility = utility
            best_state = {k: v.detach().cpu().clone() for k, v in late.state_dict().items()}
    late.load_state_dict(best_state)
    bundled = {}
    for split in ("train", "dev", "test"):
        bundled[split] = _tasp_probabilities(matrices, iculos, model_early, model_middle, late, phys_mean, split)
    dev_threshold = choose_cutoff(labels_of("dev"), bundled["dev"], probability=True)
    return {split: score_split(labels_of(split), bundled[split], dev_threshold) for split in ("train", "dev", "test")}


@torch.no_grad()
def _predict_gru(model, features, mean, split: str) -> np.ndarray:
    model.eval()
    data = LateStayDataset(split, features, mean.cpu().numpy(), train_only_long=False)
    loader = DataLoader(data, batch_size=64, shuffle=False, collate_fn=_pad_batch)
    # Dataset ids are all patients in order, so batches stay in patient order.
    flat = np.zeros(int(load_raw(split)[2][-1]), dtype=np.float64)
    cursor = 0
    for values, mask, delta, extra, _labels, _late, lengths in loader:
        values, mask, delta, extra = (t.to(DEVICE) for t in (values, mask, delta, extra))
        probs = torch.sigmoid(model(values, mask, delta, extra, mean)).cpu().numpy()
        for i, n in enumerate(lengths.tolist()):
            flat[cursor : cursor + n] = probs[i, :n]
            cursor += n
    return flat


def _tasp_probabilities(matrices, iculos, model_early, model_middle, late, mean, split: str):
    early = model_early.predict_proba(matrices[split][:, TASP_EARLY_COLUMNS])[:, 1]
    middle = model_middle.predict_proba(matrices[split])[:, 1]
    gru = _predict_gru(late, matrices[split], mean, split)
    hour = iculos[split]
    flat = np.where(hour <= 9, early, np.where(hour <= 49, middle, gru))
    return sequences(flat, load_raw(split)[2])


def _print_block(name: str, results: dict) -> None:
    print(f"\n{name}", flush=True)
    print(f"{'split':<8}{'utility':>10}{'AUPRC':>10}{'AUROC':>10}{'sens':>8}{'FP/100':>10}", flush=True)
    for split in ("train", "dev", "test"):
        row = results[split]
        print(
            f"{split:<8}{row['utility']:10.4f}{row['auprc']:10.4f}{row['auroc']:10.4f}"
            f"{row['hour_sensitivity']:8.3f}{row['fp_per_100_nonseptic_hours']:10.2f}",
            flush=True,
        )


def main() -> None:
    check_utility()
    check_features()
    ensure_raw()
    means = training_means()
    RUN.mkdir(parents=True, exist_ok=True)
    outcome = {}
    cnn_path = Path(__file__).resolve().parent / "runs" / "cnn_baseline" / "metrics.json"
    if cnn_path.exists():
        cnn = json.loads(cnn_path.read_text(encoding="utf-8"))
        outcome["cnn_baseline"] = {"dev": cnn["dev"], "test": cnn["test"], "note": "Previous causal CNN. Train utility was not recorded."}
    jobs = (
        ("sepsyd", lambda: train_sepsyd(means)),
        ("ctl", train_ctl),
        ("separatrix", lambda: train_separatrix(means)),
        ("tasp", lambda: train_tasp(means)),
        ("signature", lambda: train_signature(means)),
    )
    for name, job in jobs:
        outcome[name] = job()
        (RUN / "metrics.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        _print_block(name, outcome[name])
    print(f"\nwrote {RUN / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
