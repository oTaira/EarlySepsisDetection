"""Feature pipelines for the five official PhysioNet 2019 winners.

Every feature at an hour uses only that hour and the past. Statistics used
for imputation or scaling are fit on the training stays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

from dataset import N_PHYSIO, PHYSIO, PROCESSED, manifest_rows, read_stay, stay_path

RAW_VERSION = 1
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
# Separatrix's 13 covariates. EtCO2 is left out; it is missing for every Hospital A stay.
SEP_COVARIATES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
SEP_WINDOW = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
SEP_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]


def _index(name: str) -> int:
    if name in PHYSIO:
        return PHYSIO.index(name)
    return N_PHYSIO + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"].index(name)


def ensure_raw() -> dict:
    marker = PROCESSED / "raw_READY"
    if marker.exists():
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("version") == RAW_VERSION:
            return meta
    print("caching raw stays", flush=True)
    meta = {"version": RAW_VERSION, "splits": {}}
    for split in ("train", "dev", "test"):
        rows = manifest_rows(split)
        n_hours = sum(int(row["n_hours"]) for row in rows)
        values = np.lib.format.open_memmap(
            PROCESSED / f"raw_{split}_values.npy", mode="w+", dtype=np.float32, shape=(n_hours, 40)
        )
        labels = np.lib.format.open_memmap(
            PROCESSED / f"raw_{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(n_hours,)
        )
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        cursor = 0
        for i, row in enumerate(rows):
            raw = read_stay(stay_path(split, row))
            end = cursor + raw.shape[0]
            values[cursor:end] = raw[:, :40]
            labels[cursor:end] = raw[:, 40].astype(np.uint8)
            offsets[i + 1] = end
            cursor = end
            if (i + 1) % 5000 == 0:
                print(f"cached {split} {i + 1}/{len(rows)}", flush=True)
        values.flush()
        labels.flush()
        del values, labels
        np.save(PROCESSED / f"raw_{split}_offsets.npy", offsets)
        meta["splits"][split] = {"patients": len(rows), "hours": int(n_hours)}
    marker.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_raw(split: str):
    values = np.load(PROCESSED / f"raw_{split}_values.npy", mmap_mode="r")
    labels = np.load(PROCESSED / f"raw_{split}_labels.npy", mmap_mode="r")
    offsets = np.load(PROCESSED / f"raw_{split}_offsets.npy")
    return values, labels, offsets


def training_means() -> np.ndarray:
    values, _, _ = load_raw("train")
    means = np.zeros(40, dtype=np.float64)
    for j in range(40):
        col = np.asarray(values[:, j], dtype=np.float64)
        good = col[np.isfinite(col)]
        means[j] = float(good.mean()) if good.size else 0.0
    return means


def log_columns() -> list[int]:
    """Columns whose observed training values are nonnegative and right-skewed."""
    values, _, _ = load_raw("train")
    chosen = []
    for j in range(N_PHYSIO):
        col = np.asarray(values[:, j], dtype=np.float64)
        good = col[np.isfinite(col)]
        if good.size < 100 or good.min() < 0:
            continue
        std = float(good.std())
        if std < 1e-8:
            continue
        skew = float(np.mean(((good - good.mean()) / std) ** 3))
        if skew > 1.0:
            chosen.append(j)
    return chosen


def sequences(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(flat[int(a) : int(b)]) for a, b in zip(offsets[:-1], offsets[1:])]


def labels_of(split: str) -> list[np.ndarray]:
    labels, offsets = load_raw(split)[1:]
    return [np.asarray(labels[int(a) : int(b)], dtype=np.int32) for a, b in zip(offsets[:-1], offsets[1:])]


@njit
def _add_piece(s1, s2, s3, delta):
    d = delta.shape[0]
    t2 = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            t2[i, j] = 0.5 * delta[i] * delta[j]
    n1 = np.empty(d)
    n2 = np.empty((d, d))
    n3 = np.empty((d, d, d))
    for i in range(d):
        n1[i] = s1[i] + delta[i]
        for j in range(d):
            n2[i, j] = s2[i, j] + t2[i, j] + s1[i] * delta[j]
            for k in range(d):
                t3 = (1.0 / 6.0) * delta[i] * delta[j] * delta[k]
                n3[i, j, k] = s3[i, j, k] + t3 + s1[i] * t2[j, k] + s2[i, j] * delta[k]
    for i in range(d):
        s1[i] = n1[i]
        for j in range(d):
            s2[i, j] = n2[i, j]
            for k in range(d):
                s3[i, j, k] = n3[i, j, k]


@njit
def _flatten(s1, s2, s3, dest):
    d = s1.shape[0]
    pos = 0
    for i in range(d):
        dest[pos] = s1[i]
        pos += 1
    for i in range(d):
        for j in range(d):
            dest[pos] = s2[i, j]
            pos += 1
    for i in range(d):
        for j in range(d):
            for k in range(d):
                dest[pos] = s3[i, j, k]
                pos += 1


@njit
def sliding_leadlag_signature(path, window, out):
    """Order-3 lead-lag signature of each trailing window. path is (T, d)."""
    t_len, d = path.shape
    wide = 2 * d
    s1 = np.zeros(wide)
    s2 = np.zeros((wide, wide))
    s3 = np.zeros((wide, wide, wide))
    step = np.zeros(wide)
    for t in range(t_len):
        for i in range(wide):
            s1[i] = 0.0
            for j in range(wide):
                s2[i, j] = 0.0
                for k in range(wide):
                    s3[i, j, k] = 0.0
        start = 0 if t + 1 < window else t + 1 - window
        for i in range(start, t):
            for a in range(d):
                step[a] = path[i + 1, a] - path[i, a]
                step[d + a] = 0.0
            _add_piece(s1, s2, s3, step)
            for a in range(d):
                step[a] = 0.0
                step[d + a] = path[i + 1, a] - path[i, a]
            _add_piece(s1, s2, s3, step)
        _flatten(s1, s2, s3, out[t])


def signature_width(d: int) -> int:
    wide = 2 * d
    return wide + wide * wide + wide * wide * wide


@njit
def sliding_column_signatures(series, window, out):
    """Order-3 lead-lag signature of each 1D column. series is (T, C), out is (T, C, 14)."""
    t_len, n_col = series.shape
    path = np.empty((t_len, 1))
    column = np.empty((t_len, 14))
    for c in range(n_col):
        for t in range(t_len):
            path[t, 0] = series[t, c]
        sliding_leadlag_signature(path, window, column)
        for t in range(t_len):
            for k in range(14):
                out[t, c, k] = column[t, k]


def partial_sofa_row(platelets, bilirubin, map_, creatinine) -> float:
    score = 0.0
    if np.isfinite(platelets):
        score += 4 if platelets < 20 else 3 if platelets < 50 else 2 if platelets < 100 else 1 if platelets < 150 else 0
    if np.isfinite(bilirubin):
        score += 4 if bilirubin >= 12 else 3 if bilirubin >= 6 else 2 if bilirubin >= 2 else 1 if bilirubin >= 1.2 else 0
    if np.isfinite(map_):
        score += 1 if map_ < 70 else 0
    if np.isfinite(creatinine):
        score += 4 if creatinine >= 5 else 3 if creatinine >= 3.5 else 2 if creatinine >= 2 else 1 if creatinine >= 1.2 else 0
    return score


def forward_fill(x: np.ndarray, fill: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    idx = np.where(mask, np.arange(x.shape[0])[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    safe = np.where(mask, x, 0.0)
    gathered = np.take_along_axis(safe, np.clip(idx, 0, None), axis=0)
    return np.where(idx < 0, fill, gathered).astype(np.float32)


def _ratio(num, den) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) >= 1e-3)
    out[ok] = num[ok] / den[ok]
    return out


def build_signature(split: str, means: np.ndarray) -> np.ndarray:
    values, _, offsets = load_raw(split)
    n_hours = int(offsets[-1])
    width = 4 + 27 + 16 + 40 + signature_width(4) + N_PHYSIO * signature_width(1)
    features = np.lib.format.open_memmap(
        PROCESSED / f"sig_{split}.npy", mode="w+", dtype=np.float32, shape=(n_hours, width)
    )
    temp_i = _index("Temp")
    lab_idx = list(range(8, N_PHYSIO))
    count_idx = [temp_i] + lab_idx
    map_i, sbp_i, hr_i = _index("MAP"), _index("SBP"), _index("HR")
    bili_i, cr_i, plt_i = _index("Bilirubin_total"), _index("Creatinine"), _index("Platelets")
    joint = np.zeros((336, signature_width(4)), dtype=np.float64)
    per_col = np.zeros((336, N_PHYSIO, signature_width(1)), dtype=np.float64)
    for p, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw = np.asarray(values[int(a) : int(b)], dtype=np.float32)
        t_len = raw.shape[0]
        observed = np.isfinite(raw[:, :N_PHYSIO])
        filled = forward_fill(raw[:, :N_PHYSIO], means[:N_PHYSIO])
        sofa = np.array(
            [partial_sofa_row(filled[t, plt_i], filled[t, bili_i], filled[t, map_i], filled[t, cr_i]) for t in range(t_len)],
            dtype=np.float64,
        )
        base = np.empty(t_len)
        base[:24] = sofa[0]
        if t_len > 24:
            base[24:] = sofa[:-24]
        shock = _ratio(filled[:, hr_i], filled[:, sbp_i])
        bun_cr = _ratio(filled[:, bili_i], filled[:, cr_i])
        path = np.column_stack(
            [
                np.arange(t_len, dtype=np.float64),
                sofa,
                np.nan_to_num(filled[:, map_i], nan=0.0),
                np.nan_to_num(bun_cr, nan=0.0),
            ]
        )
        sliding_leadlag_signature(path, 7, joint[:t_len])
        block = np.zeros((t_len, width), dtype=np.float32)
        cursor = 0
        block[:, cursor] = np.nan_to_num(shock, nan=0.0)
        block[:, cursor + 1] = np.nan_to_num(bun_cr, nan=0.0)
        block[:, cursor + 2] = sofa
        block[:, cursor + 3] = (sofa - base) >= 2
        cursor = 4
        csum = np.cumsum(observed[:, count_idx], axis=0)
        if t_len > 8:
            block[:, cursor : cursor + len(count_idx)] = csum - np.vstack(
                [np.zeros((8, len(count_idx))), csum[:-8]]
            )
        else:
            block[:, cursor : cursor + len(count_idx)] = csum
        cursor += len(count_idx)
        window = filled[:, :8]
        for t in range(t_len):
            start = max(0, t + 1 - 6)
            block[t, cursor : cursor + 8] = window[start : t + 1].min(axis=0)
            block[t, cursor + 8 : cursor + 16] = window[start : t + 1].max(axis=0)
        cursor += 16
        current = raw.copy()
        block[:, cursor : cursor + 40] = current
        cursor += 40
        block[:, cursor : cursor + joint.shape[1]] = joint[:t_len]
        cursor += joint.shape[1]
        cumulative = np.cumsum(np.nan_to_num(filled, nan=0.0), axis=0)
        sliding_column_signatures(cumulative, 7, per_col[:t_len])
        block[:, cursor:] = per_col[:t_len].reshape(t_len, -1)
        features[int(a) : int(b)] = block
        if (p + 1) % 2000 == 0:
            print(f"signature features {split} {p + 1}/{len(offsets) - 1}", flush=True)
    features.flush()
    return features


def build_sepsyd(split: str, means: np.ndarray, skewed: list[int]) -> np.ndarray:
    values, _, offsets = load_raw(split)
    lookback = 5
    width = lookback * (40 + N_PHYSIO)
    features = np.zeros((int(offsets[-1]), width), dtype=np.float32)
    for a, b in zip(offsets[:-1], offsets[1:]):
        raw = np.asarray(values[int(a) : int(b)], dtype=np.float64)
        observed = np.isfinite(raw[:, :N_PHYSIO])
        transformed = raw.copy()
        for j in skewed:
            good = np.isfinite(transformed[:, j])
            transformed[good, j] = np.log1p(transformed[good, j])
        centered = transformed - means
        centered[:, :N_PHYSIO] = np.where(np.isfinite(raw[:, :N_PHYSIO]), centered[:, :N_PHYSIO], np.nan)
        filled = centered.copy()
        for j in range(40):
            col = filled[:, j]
            good = np.isfinite(col)
            if not good.any():
                filled[:, j] = 0.0
                continue
            first = int(np.argmax(good))
            col[:first] = 0.0
            idx = np.where(np.isfinite(col), np.arange(col.shape[0]), 0)
            np.maximum.accumulate(idx, out=idx)
            filled[:, j] = col[idx]
        hour = np.concatenate([filled, observed.astype(np.float64)], axis=1)
        block = np.zeros((raw.shape[0], width), dtype=np.float32)
        for lag in range(lookback):
            if lag == 0:
                block[:, : hour.shape[1]] = hour
            else:
                block[lag:, lag * hour.shape[1] : (lag + 1) * hour.shape[1]] = hour[:-lag]
        features[int(a) : int(b)] = block
    return features


@njit
def _quantile(sorted_v, q):
    n = sorted_v.shape[0]
    if n == 1:
        return sorted_v[0]
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    weight = pos - lo
    return sorted_v[lo] * (1.0 - weight) + sorted_v[hi] * weight


@njit
def _prefix_entropy(values):
    n = values.shape[0]
    lo = values[0]
    hi = values[0]
    for i in range(n):
        if values[i] < lo:
            lo = values[i]
        if values[i] > hi:
            hi = values[i]
    if hi - lo < 1e-8:
        return 0.0
    hist = np.zeros(10)
    for i in range(n):
        bin_id = int((values[i] - lo) / (hi - lo) * 10.0)
        if bin_id > 9:
            bin_id = 9
        hist[bin_id] += 1.0
    ent = 0.0
    for b in range(10):
        if hist[b] > 0:
            p = hist[b] / n
            ent -= p * np.log(p)
    return ent


@njit
def separatrix_type1(imputed, observed_counts_source, out):
    """245 causal features for the 13 Separatrix covariates.

    imputed: (T, 13) with age at 7 and gender at 8.
    observed_counts_source: (T, 13) 1 where the raw value was present.
    """
    t_len = imputed.shape[0]
    window_cols = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12], dtype=np.int64)
    vital_cols = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    for t in range(t_len):
        pos = 0
        for w in (5, 11):
            start = t + 1 - w
            for c in range(11):
                col = window_cols[c]
                buf = np.empty(w)
                for k in range(w):
                    src = start + k
                    buf[k] = imputed[0, col] if src < 0 else imputed[src, col]
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
        for c in range(7):
            col = vital_cols[c]
            energy = 0.0
            for i in range(t + 1):
                energy += imputed[i, col] * imputed[i, col]
            diff = 0.0
            if t > 0:
                for i in range(t):
                    diff += imputed[i + 1, col] - imputed[i, col]
                diff /= t
            series = imputed[: t + 1, col].copy()
            out[t, pos] = energy
            out[t, pos + 1] = _prefix_entropy(series)
            out[t, pos + 2] = diff
            pos += 3
        for c in range(13):
            count = 0.0
            for i in range(t + 1):
                count += observed_counts_source[i, c]
            out[t, pos] = count
            pos += 1
        for c in range(13):
            out[t, pos] = imputed[t, c]
            pos += 1


@njit
def separatrix_missingness(present, out):
    """162 causal missingness features. present is (T, 38) and 1 means observed."""
    t_len, n_cov = present.shape
    run_len = np.ones(n_cov)
    run_sum = np.ones(n_cov)
    run_sum2 = np.ones(n_cov)
    run_n = np.ones(n_cov)
    cur = present[0].copy()
    valid_open = present[0].copy()
    valid_len = np.zeros(n_cov)
    valid_sum = np.zeros(n_cov)
    valid_sum2 = np.zeros(n_cov)
    valid_n = np.zeros(n_cov)
    for c in range(n_cov):
        if present[0, c] > 0.5:
            valid_len[c] = 1.0
            valid_sum[c] = 1.0
            valid_sum2[c] = 1.0
            valid_n[c] = 1.0
    for t in range(t_len):
        if t > 0:
            for c in range(n_cov):
                bit = present[t, c]
                if bit == cur[c]:
                    old = run_len[c]
                    run_sum[c] -= old
                    run_sum2[c] -= old * old
                    run_len[c] = old + 1.0
                    run_sum[c] += run_len[c]
                    run_sum2[c] += run_len[c] * run_len[c]
                else:
                    cur[c] = bit
                    run_n[c] += 1.0
                    run_len[c] = 1.0
                    run_sum[c] += 1.0
                    run_sum2[c] += 1.0
                if bit > 0.5:
                    if valid_open[c] > 0.5:
                        old = valid_len[c]
                        valid_sum[c] -= old
                        valid_sum2[c] -= old * old
                        valid_len[c] = old + 1.0
                        valid_sum[c] += valid_len[c]
                        valid_sum2[c] += valid_len[c] * valid_len[c]
                    else:
                        valid_open[c] = 1.0
                        valid_n[c] += 1.0
                        valid_len[c] = 1.0
                        valid_sum[c] += 1.0
                        valid_sum2[c] += 1.0
                else:
                    valid_open[c] = 0.0
        for c in range(n_cov):
            mean = run_sum[c] / run_n[c]
            var = run_sum2[c] / run_n[c] - mean * mean
            out[t, c] = mean
            out[t, n_cov + c] = var if var > 0 else 0.0
            if valid_n[c] == 0:
                out[t, 2 * n_cov + c] = 0.0
                out[t, 3 * n_cov + c] = 0.0
            else:
                vmean = valid_sum[c] / valid_n[c]
                vvar = valid_sum2[c] / valid_n[c] - vmean * vmean
                out[t, 2 * n_cov + c] = valid_sum[c]
                out[t, 3 * n_cov + c] = vvar if vvar > 0 else 0.0
        for slot in range(5):
            hour = t - 4 + slot
            base = 4 * n_cov + 2 * slot
            if hour < 0:
                out[t, base] = 0.0
                out[t, base + 1] = 0.0
                continue
            # Run lengths across covariates at this one hour.
            length = 1.0
            total = 0.0
            total2 = 0.0
            runs = 0.0
            prev = present[hour, 0]
            for c in range(1, n_cov):
                if present[hour, c] == prev:
                    length += 1.0
                else:
                    total += length
                    total2 += length * length
                    runs += 1.0
                    prev = present[hour, c]
                    length = 1.0
            total += length
            total2 += length * length
            runs += 1.0
            mean = total / runs
            var = total2 / runs - mean * mean
            out[t, base] = mean
            out[t, base + 1] = var if var > 0 else 0.0


def _causal_fill(col: np.ndarray, train_mean: float) -> np.ndarray:
    """Last observed value. Hours before the first measurement use the training mean."""
    out = np.asarray(col, dtype=np.float64).copy()
    good = np.isfinite(out)
    if not good.any():
        return np.full(out.shape, train_mean)
    index = np.where(good, np.arange(out.shape[0]), 0)
    np.maximum.accumulate(index, out=index)
    filled = out[index]
    filled[: int(np.argmax(good))] = train_mean
    return filled


def _linear_fill(col: np.ndarray, train_mean: float) -> np.ndarray:
    out = col.astype(np.float64).copy()
    good = np.isfinite(out)
    if good.sum() == 0:
        return np.full(col.shape, train_mean)
    if good.sum() < 3:
        out[~good] = train_mean
        return out
    idx = np.arange(col.shape[0])
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def build_separatrix(split: str, means: np.ndarray) -> np.ndarray:
    """Window features filled from the last past measurement only."""
    path = PROCESSED / f"separatrix_causal_{split}.npy"
    if path.exists():
        return np.load(path)
    values, _, offsets = load_raw(split)
    cov_idx = [_index(name) for name in SEP_COVARIATES]
    miss_idx = [j for j in range(40) if j not in (_index("Age"), _index("Gender"))]
    features = np.zeros((int(offsets[-1]), 407), dtype=np.float32)
    type1 = np.zeros((336, 245), dtype=np.float64)
    type2 = np.zeros((336, 162), dtype=np.float64)
    for p, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw = np.asarray(values[int(a) : int(b)], dtype=np.float64)
        t_len = raw.shape[0]
        chosen = raw[:, cov_idx]
        imputed = np.column_stack([_causal_fill(chosen[:, j], means[cov_idx[j]]) for j in range(13)])
        present_cov = np.isfinite(chosen).astype(np.float64)
        separatrix_type1(imputed, present_cov, type1[:t_len])
        present_miss = np.isfinite(raw[:, miss_idx]).astype(np.float64)
        separatrix_missingness(present_miss, type2[:t_len])
        features[int(a) : int(b), :245] = type1[:t_len]
        features[int(a) : int(b), 245:] = type2[:t_len]
        if (p + 1) % 2000 == 0:
            print(f"separatrix features {split} {p + 1}/{len(offsets) - 1}", flush=True)
    np.save(path, features)
    return np.load(path)


def build_ctl(split: str) -> np.ndarray:
    """Five-hour window of the raw values, with NaNs left missing, plus 34 masks."""
    values, _, offsets = load_raw(split)
    lookback = 5
    width = lookback * (40 + N_PHYSIO)
    features = np.full((int(offsets[-1]), width), np.nan, dtype=np.float32)
    for a, b in zip(offsets[:-1], offsets[1:]):
        raw = np.asarray(values[int(a) : int(b)], dtype=np.float32)
        mask = np.isfinite(raw[:, :N_PHYSIO]).astype(np.float32)
        hour = np.concatenate([raw, mask], axis=1)
        block = np.zeros((raw.shape[0], width), dtype=np.float32)
        for lag in range(lookback):
            dest = block[:, lag * hour.shape[1] : (lag + 1) * hour.shape[1]]
            if lag == 0:
                dest[:] = hour
            else:
                dest[lag:] = hour[:-lag]
                dest[:lag] = 0.0
        # Restore NaNs for the current hour's unmeasured values. Padded past hours stay 0.
        block[:, :40] = raw
        features[int(a) : int(b)] = block
    return features


def shift_labels(labels: list[np.ndarray], hours: int = 3) -> list[np.ndarray]:
    """CTL training target: move the positive region three hours earlier."""
    shifted = []
    for y in labels:
        out = y.copy()
        if np.any(out):
            start = int(np.argmax(out))
            out[max(0, start - hours) :] = 1
        shifted.append(out)
    return shifted


@njit
def _rolling_vital_stats(x, window, out_min, out_max, out_mean, out_std, out_range):
    t_len, n_col = x.shape
    for t in range(t_len):
        start = 0 if t + 1 < window else t + 1 - window
        n = t - start + 1
        for c in range(n_col):
            lo = x[t, c]
            hi = x[t, c]
            total = 0.0
            for i in range(start, t + 1):
                v = x[i, c]
                total += v
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
            mean = total / n
            var = 0.0
            for i in range(start, t + 1):
                var += (x[i, c] - mean) * (x[i, c] - mean)
            var /= n
            out_min[t, c] = lo
            out_max[t, c] = hi
            out_mean[t, c] = mean
            out_std[t, c] = np.sqrt(var if var > 0 else 0.0)
            out_range[t, c] = hi - lo


def build_tasp(split: str, means: np.ndarray) -> np.ndarray:
    """269 features. The early-stage model uses a 139-column subset of these."""
    values, _, offsets = load_raw(split)
    width = 269
    features = np.zeros((int(offsets[-1]), width), dtype=np.float32)
    hr, sbp, resp = _index("HR"), _index("SBP"), _index("Resp")
    sao2, fio2 = _index("SaO2"), _index("FiO2")
    map_i, bili_i, cr_i, plt_i = _index("MAP"), _index("Bilirubin_total"), _index("Creatinine"), _index("Platelets")
    worst_high = np.array([True, False, True, False, False, False, True, False])
    for p, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        raw = np.asarray(values[int(a) : int(b)], dtype=np.float32)
        t_len = raw.shape[0]
        phys = raw[:, :N_PHYSIO].copy()
        bad_fio2 = (phys[:, fio2] < 0) | (phys[:, fio2] > 100)
        phys[bad_fio2, fio2] = np.nan
        observed = np.isfinite(phys)
        filled = forward_fill(phys, means[:N_PHYSIO])
        block = np.zeros((t_len, width), dtype=np.float32)
        block[:, :N_PHYSIO] = filled
        block[:, N_PHYSIO] = raw[:, _index("Age")]
        block[:, N_PHYSIO + 1] = raw[:, _index("Gender")]
        block[:, N_PHYSIO + 2] = np.nan_to_num(raw[:, _index("HospAdmTime")], nan=means[_index("HospAdmTime")])
        block[:, N_PHYSIO + 3] = raw[:, _index("ICULOS")]
        unit1 = raw[0, _index("Unit1")]
        unit2 = raw[0, _index("Unit2")]
        block[:, N_PHYSIO + 4] = 1.0 if unit1 == 1 else 2.0 if unit2 == 1 else 0.0
        shock = _ratio(filled[:, hr], filled[:, sbp])
        sao = _ratio(filled[:, sao2], filled[:, fio2])
        for t in range(t_len):
            sofa = partial_sofa_row(filled[t, plt_i], filled[t, bili_i], filled[t, map_i], filled[t, cr_i])
            qsofa = float((filled[t, resp] >= 22) + (filled[t, sbp] <= 100))
            block[t, 39] = 0.0 if not np.isfinite(shock[t]) else shock[t]
            block[t, 40] = 0.0 if not np.isfinite(sao[t]) else sao[t]
            block[t, 41] = sofa
            block[t, 42] = qsofa
        cursor = 43
        vitals = filled[:, :8]
        for window in (6, 12, 24):
            stats = [np.zeros((t_len, 8), dtype=np.float64) for _ in range(5)]
            _rolling_vital_stats(vitals.astype(np.float64), window, *stats)
            packed = np.concatenate(stats, axis=1)
            block[:, cursor : cursor + 40] = packed
            cursor += 40
        for vital in range(8):
            series = vitals[:, vital]
            for lag_i, lag in enumerate((1, 2)):
                base = 163 + vital * 6 + lag_i * 3
                prev = np.zeros(t_len, dtype=np.float32)
                prev[lag:] = series[:-lag]
                block[:, base] = prev
                denom = np.where(np.abs(prev) < 1e-3, np.nan, prev)
                ratio = series / denom
                block[:, base + 1] = np.nan_to_num(ratio, nan=0.0)
                block[:, base + 2] = series - prev
        cursor = 211
        for window in (6, 12, 24):
            for t in range(t_len):
                start = max(0, t + 1 - window)
                chunk = vitals[start : t + 1]
                for vital in range(8):
                    block[t, cursor + vital] = chunk[:, vital].max() if worst_high[vital] else chunk[:, vital].min()
            cursor += 8
        csum = np.cumsum(observed, axis=0)
        if t_len > 24:
            counts = csum - np.vstack([np.zeros((24, N_PHYSIO)), csum[:-24]])
        else:
            counts = csum
        block[:, 235:269] = counts
        features[int(a) : int(b)] = block
        if (p + 1) % 2000 == 0:
            print(f"tasp features {split} {p + 1}/{len(offsets) - 1}", flush=True)
    return features


TASP_EARLY_COLUMNS = np.concatenate([np.arange(0, 83), np.arange(163, 211), np.arange(211, 219)])


def self_check() -> None:
    path = np.array([[0.0], [2.0]])
    out = np.zeros((2, signature_width(1)))
    sliding_leadlag_signature(path, 7, out)
    # One step of size 2, lead then lag. Level-1 terms are the two increments.
    level1 = out[1, :2]
    if abs(level1.sum() - 4.0) > 1e-6:
        raise AssertionError(f"lead-lag level-1 sum was {level1.sum()}")

    present = np.array([[0], [1], [1], [0], [0], [1]], dtype=np.float64)
    # Pad to 38 covariates with a copy of this bit so the function's width is satisfied.
    wide = np.repeat(present, 38, axis=1)
    missing = np.zeros((6, 162))
    separatrix_missingness(wide, missing)
    # All 38 covariates share the run lengths 1, 2, 2, 1.
    if abs(missing[-1, 0] - 1.5) > 1e-6:
        raise AssertionError(f"run-length mean was {missing[-1, 0]}")
    sofa = partial_sofa_row(40, 1.5, 65, 2.5)
    if sofa != 7:
        raise AssertionError(f"partial SOFA was {sofa}")
    imputed = np.linspace(0, 1, 130).reshape(10, 13)
    observed = np.ones((10, 13))
    type1 = np.zeros((10, 245))
    separatrix_type1(imputed, observed, type1)
    if not np.isfinite(type1).all():
        raise AssertionError("Separatrix features were not finite")
    print("winner feature self-check passed", flush=True)
