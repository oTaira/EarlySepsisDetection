"""PhysioNet/CinC 2019 normalized utility.

This is the official challenge score. SepsisLabel is already 1 from six hours
before clinical onset, and this function undoes that shift itself. Do not
shift the labels before calling it.

A perfect alarm sequence scores 1. Predicting no alarms scores 0. False alarms
and late or missed sepsis score below that, and can be negative.
"""

from __future__ import annotations

import numpy as np

# Official challenge parameters.
DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3.0
MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0

M1 = float(MAX_U_TP) / float(DT_OPTIMAL - DT_EARLY)
B1 = -M1 * DT_EARLY
M2 = float(-MAX_U_TP) / float(DT_LATE - DT_OPTIMAL)
B2 = -M2 * DT_LATE
M3 = float(MIN_U_FN) / float(DT_LATE - DT_OPTIMAL)
B3 = -M3 * DT_OPTIMAL


def hour_utility(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-hour utility of predicting 1 and of predicting 0.

    Matches compute_prediction_utility from the 2019 challenge evaluator.
    Each hour is scored on its own; alarms do not cancel each other.
    """
    labels = np.asarray(labels)
    n = int(labels.shape[0])
    u_pos = np.zeros(n, dtype=np.float64)
    u_neg = np.zeros(n, dtype=np.float64)
    if np.any(labels):
        t_sepsis = float(np.argmax(labels) - DT_OPTIMAL)
        t = np.arange(n, dtype=np.float64)
        dt = t - t_sepsis
        in_window = t <= (t_sepsis + DT_LATE)
        early = in_window & (t <= (t_sepsis + DT_OPTIMAL))
        late = in_window & ~early
        u_pos[early] = np.maximum(M1 * dt[early] + B1, U_FP)
        u_pos[late] = M2 * dt[late] + B2
        u_neg[late] = M3 * dt[late] + B3
    else:
        u_pos[:] = U_FP
        u_neg[:] = U_TN
    return u_pos, u_neg


def prediction_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Total utility of one stay. Predictions are binary."""
    predictions = np.asarray(predictions).astype(bool)
    if predictions.shape[0] != np.asarray(labels).shape[0]:
        raise ValueError("labels and predictions must have the same length")
    u_pos, u_neg = hour_utility(labels)
    return float(np.where(predictions, u_pos, u_neg).sum())


def best_predictions(labels: np.ndarray) -> np.ndarray:
    """Official oracle: alarm on the hours whose true-positive utility is nonnegative."""
    labels = np.asarray(labels)
    n = int(labels.shape[0])
    predictions = np.zeros(n, dtype=bool)
    if np.any(labels):
        t_sepsis = float(np.argmax(labels) - DT_OPTIMAL)
        start = max(0, int(t_sepsis + DT_EARLY))
        end = min(n, int(t_sepsis + DT_LATE) + 1)
        predictions[start:end] = True
    return predictions


def normalized_utility(label_seqs, prediction_seqs) -> float:
    """Cohort utility scaled so the oracle is 1 and no alarms is 0."""
    observed = 0.0
    best = 0.0
    inaction = 0.0
    for labels, predictions in zip(label_seqs, prediction_seqs):
        labels = np.asarray(labels)
        u_pos, u_neg = hour_utility(labels)
        observed += float(np.where(np.asarray(predictions).astype(bool), u_pos, u_neg).sum())
        inaction += float(u_neg.sum())
        best += float(np.where(best_predictions(labels), u_pos, u_neg).sum())
    denom = best - inaction
    if denom == 0:
        raise ValueError("utility is undefined when no patient can score above inaction")
    return (observed - inaction) / denom


def choose_threshold(label_seqs, probability_seqs, thresholds: np.ndarray | None = None):
    """Pick the binary cutoff that maximizes normalized utility."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)
    prepared = []
    inaction = 0.0
    best = 0.0
    for labels, probs in zip(label_seqs, probability_seqs):
        labels = np.asarray(labels)
        probs = np.asarray(probs, dtype=np.float64)
        u_pos, u_neg = hour_utility(labels)
        prepared.append((probs, u_pos, u_neg))
        inaction += float(u_neg.sum())
        best += float(np.where(best_predictions(labels), u_pos, u_neg).sum())
    denom = best - inaction
    scores = np.zeros(len(thresholds), dtype=np.float64)
    for probs, u_pos, u_neg in prepared:
        alarm = probs[None, :] >= thresholds[:, None]
        scores += np.where(alarm, u_pos, u_neg).sum(axis=1)
    scores = (scores - inaction) / denom
    # On a tie, keep the higher threshold so the alarm is the less trigger-happy one.
    best_index = int(len(scores) - 1 - np.argmax(scores[::-1]))
    return float(thresholds[best_index]), float(scores[best_index])


def _reference_utility(labels, predictions) -> float:
    """Line-by-line port of the official loop, used only to check the vectorized score."""
    labels = [int(v) for v in np.asarray(labels).tolist()]
    predictions = [int(v) for v in np.asarray(predictions).tolist()]
    if any(labels):
        is_septic = True
        t_sepsis = labels.index(1) - DT_OPTIMAL
    else:
        is_septic = False
        t_sepsis = float("inf")
    total = 0.0
    for t, prediction in enumerate(predictions):
        if t <= t_sepsis + DT_LATE:
            if is_septic and prediction:
                if t <= t_sepsis + DT_OPTIMAL:
                    total += max(M1 * (t - t_sepsis) + B1, U_FP)
                elif t <= t_sepsis + DT_LATE:
                    total += M2 * (t - t_sepsis) + B2
            elif (not is_septic) and prediction:
                total += U_FP
            elif is_septic and not prediction:
                if t <= t_sepsis + DT_OPTIMAL:
                    total += 0.0
                elif t <= t_sepsis + DT_LATE:
                    total += M3 * (t - t_sepsis) + B3
            elif (not is_septic) and not prediction:
                total += U_TN
    return total


def self_check() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1, 1, 1])
    got = prediction_utility(labels, predictions)
    if abs(got - 3.388888888888889) > 1e-8:
        raise AssertionError(f"official example utility was {got}")
    if abs(got - _reference_utility(labels, predictions)) > 1e-8:
        raise AssertionError("vectorized utility disagrees with the official loop")

    rng = np.random.default_rng(230)
    cohort_labels = []
    cohort_preds = []
    for _ in range(30):
        n = int(rng.integers(8, 80))
        y = np.zeros(n, dtype=int)
        if rng.random() < 0.3:
            start = int(rng.integers(0, n))
            y[start:] = 1
        pred = rng.integers(0, 2, size=n)
        if abs(prediction_utility(y, pred) - _reference_utility(y, pred)) > 1e-6:
            raise AssertionError("utility mismatch on a random stay")
        cohort_labels.append(y)
        cohort_preds.append(pred)

    oracle = [best_predictions(y) for y in cohort_labels]
    none = [np.zeros(len(y), dtype=bool) for y in cohort_labels]
    if abs(normalized_utility(cohort_labels, oracle) - 1.0) > 1e-8:
        raise AssertionError("oracle utility is not 1")
    if abs(normalized_utility(cohort_labels, none) - 0.0) > 1e-8:
        raise AssertionError("no-alarm utility is not 0")

    quiet = prediction_utility(np.zeros(10, dtype=int), np.zeros(10, dtype=int))
    one_alarm = prediction_utility(np.zeros(10, dtype=int), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    if quiet != 0.0 or abs(one_alarm - U_FP) > 1e-12:
        raise AssertionError("false-alarm utility on a non-septic stay is wrong")


if __name__ == "__main__":
    self_check()
    print("utility self-check passed")
