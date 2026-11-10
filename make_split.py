"""Patient-level 70/15/15 split of the PhysioNet/CinC 2019 training set.

Each stay is kept whole. Assignment is stratified by hospital, whether the
stay ever has a positive SepsisLabel, and stay length (duration quintile),
so those distributions match across train, dev, and test.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

SEED = 230
TRAIN_FRAC = 0.70
DEV_FRAC = 0.15
TEST_FRAC = 0.15
N_DURATION_BINS = 5

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPLITS = ("train", "dev", "test")


def summarize_psv(raw: bytes) -> tuple[int, int]:
    """Return (n_hours, n_positive_hours) from a pipe-separated stay."""
    lines = raw.splitlines()
    if not lines or not lines[0].endswith(b"SepsisLabel"):
        raise ValueError("unexpected PSV header")
    n_hours = 0
    n_positive = 0
    for line in lines[1:]:
        if not line:
            continue
        n_hours += 1
        if line.endswith(b"|1"):
            n_positive += 1
    return n_hours, n_positive


def load_patients(zf: zipfile.ZipFile) -> list[dict]:
    patients = []
    members = [
        name
        for name in zf.namelist()
        if name.endswith(".psv") and "/training_set" in name.replace("\\", "/")
    ]
    for i, name in enumerate(members, start=1):
        hospital = "A" if "training_setA" in name else "B"
        patient_id = Path(name).stem
        with zf.open(name) as handle:
            n_hours, n_positive = summarize_psv(handle.read())
        patients.append(
            {
                "patient_id": patient_id,
                "hospital": hospital,
                "zip_name": name,
                "filename": f"training_set{hospital}/{patient_id}.psv",
                "n_hours": n_hours,
                "sepsis": int(n_positive > 0),
                "n_positive_hours": n_positive,
            }
        )
        if i % 5000 == 0:
            print(f"scanned {i} stays", flush=True)
    return patients


def assign_duration_quintiles(patients: list[dict]) -> None:
    order = sorted(
        range(len(patients)),
        key=lambda i: (patients[i]["n_hours"], patients[i]["hospital"], patients[i]["patient_id"]),
    )
    n = len(order)
    for rank, idx in enumerate(order):
        patients[idx]["duration_quintile"] = min(N_DURATION_BINS - 1, rank * N_DURATION_BINS // n)

    for q in range(N_DURATION_BINS):
        hours = [p["n_hours"] for p in patients if p["duration_quintile"] == q]
        label = f"Q{q + 1}:{min(hours)}-{max(hours)}h"
        for p in patients:
            if p["duration_quintile"] == q:
                p["duration_bin"] = label


def allocate(n: int) -> tuple[int, int, int]:
    """Largest-remainder split of n patients into 70/15/15."""
    weights = (TRAIN_FRAC, DEV_FRAC, TEST_FRAC)
    raw = [w * n for w in weights]
    counts = [math.floor(x) for x in raw]
    leftover = n - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
    for k in range(leftover):
        counts[order[k]] += 1
    return counts[0], counts[1], counts[2]


def assign_splits(patients: list[dict]) -> None:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(patients):
        key = (p["hospital"], p["sepsis"], p["duration_quintile"])
        groups[key].append(i)

    rng = random.Random(SEED)
    for key in sorted(groups):
        idxs = groups[key]
        rng.shuffle(idxs)
        n_train, n_dev, n_test = allocate(len(idxs))
        cuts = (n_train, n_train + n_dev, n_train + n_dev + n_test)
        names = (idxs[: cuts[0]], idxs[cuts[0] : cuts[1]], idxs[cuts[1] : cuts[2]])
        for split, group in zip(SPLITS, names):
            for i in group:
                patients[i]["split"] = split


def write_manifests(patients: list[dict]) -> None:
    fields = [
        "split",
        "patient_id",
        "hospital",
        "filename",
        "n_hours",
        "sepsis",
        "n_positive_hours",
        "duration_quintile",
        "duration_bin",
    ]
    manifest_dir = DATA / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    patients_sorted = sorted(patients, key=lambda p: (p["split"], p["hospital"], p["patient_id"]))
    for split in ("all", *SPLITS):
        rows = patients_sorted if split == "all" else [p for p in patients_sorted if p["split"] == split]
        path = manifest_dir / ("all_patients.csv" if split == "all" else f"{split}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def extract_splits(zf: zipfile.ZipFile, patients: list[dict]) -> None:
    for split in SPLITS:
        for hospital in ("A", "B"):
            (DATA / split / f"training_set{hospital}").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patients, start=1):
        dest = DATA / p["split"] / p["filename"]
        if not dest.exists() or dest.stat().st_size == 0:
            with zf.open(p["zip_name"]) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        if i % 5000 == 0:
            print(f"extracted {i} stays", flush=True)


def _rate(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def write_summary(patients: list[dict]) -> None:
    lines = []
    lines.append("Patient-level split of PhysioNet/CinC 2019 training stays.")
    lines.append(f"Seed {SEED}. Fractions {TRAIN_FRAC:.0%}/{DEV_FRAC:.0%}/{TEST_FRAC:.0%}.")
    lines.append("Stratified by hospital x ever-septic x duration quintile.")
    lines.append("A stay is septic if any hour has SepsisLabel=1. Duration is the number of hourly rows.")
    lines.append("")

    header = f"{'split':<8}{'patients':>10}{'% patients':>12}{'% septic':>10}{'% hosp A':>10}{'hours':>12}{'% +hours':>10}{'median h':>10}{'mean h':>10}"
    lines.append(header)
    for split in ("all", *SPLITS):
        rows = patients if split == "all" else [p for p in patients if p["split"] == split]
        hours = [p["n_hours"] for p in rows]
        hours_sorted = sorted(hours)
        mid = len(hours_sorted) // 2
        median = hours_sorted[mid] if len(hours_sorted) % 2 else (hours_sorted[mid - 1] + hours_sorted[mid]) / 2
        n_pos_hours = sum(p["n_positive_hours"] for p in rows)
        n_hours = sum(hours)
        lines.append(
            f"{split:<8}{len(rows):10d}{_rate(len(rows), len(patients)):12.2f}"
            f"{_rate(sum(p['sepsis'] for p in rows), len(rows)):10.2f}"
            f"{_rate(sum(p['hospital'] == 'A' for p in rows), len(rows)):10.2f}"
            f"{n_hours:12d}{_rate(n_pos_hours, n_hours):10.2f}{median:10.1f}{n_hours / len(rows):10.1f}"
        )

    lines.append("")
    lines.append("Sepsis rate (%) by hospital")
    lines.append(f"{'split':<8}{'A patients':>12}{'A % septic':>12}{'B patients':>12}{'B % septic':>12}")
    for split in ("all", *SPLITS):
        rows = patients if split == "all" else [p for p in patients if p["split"] == split]
        cells = []
        for hospital in ("A", "B"):
            group = [p for p in rows if p["hospital"] == hospital]
            cells.append(f"{len(group):12d}{_rate(sum(p['sepsis'] for p in group), len(group)):12.2f}")
        lines.append(f"{split:<8}{''.join(cells)}")

    lines.append("")
    lines.append("Patients by duration quintile")
    bins = sorted({p["duration_bin"] for p in patients})
    lines.append(f"{'split':<8}" + "".join(f"{b:>16}" for b in bins))
    for split in ("all", *SPLITS):
        rows = patients if split == "all" else [p for p in patients if p["split"] == split]
        counts = []
        for b in bins:
            group = [p for p in rows if p["duration_bin"] == b]
            counts.append(f"{len(group):16d}")
        lines.append(f"{split:<8}{''.join(counts)}")

    lines.append("")
    lines.append("Share of each split inside each duration quintile (%)")
    lines.append(f"{'split':<8}" + "".join(f"{b:>16}" for b in bins))
    for split in SPLITS:
        rows = [p for p in patients if p["split"] == split]
        shares = []
        for b in bins:
            group = [p for p in rows if p["duration_bin"] == b]
            shares.append(f"{_rate(len(group), len(rows)):16.2f}")
        lines.append(f"{split:<8}{''.join(shares)}")

    text = "\n".join(lines) + "\n"
    (DATA / "split_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fixed patient-level split from the public PhysioNet 2019 training ZIP.")
    parser.add_argument("zip_path", type=Path, help="Path to PhysioNet_2019_Full_Training_Data.zip")
    zip_path = parser.parse_args().zip_path.expanduser()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    DATA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for extra in ("README.md", "LICENSE.txt", "manifest.json"):
            name = f"PhysioNet_2019/{extra}"
            if name in zf.namelist():
                dest = DATA / extra
                if not dest.exists():
                    with zf.open(name) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        print("scanning stays", flush=True)
        patients = load_patients(zf)
        assign_duration_quintiles(patients)
        assign_splits(patients)
        ids = [(p["hospital"], p["patient_id"]) for p in patients]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate hospital/patient ids")
        if any(p.get("split") not in SPLITS for p in patients):
            raise RuntimeError("a stay was left unassigned")
        write_manifests(patients)
        print("extracting stays into train/dev/test", flush=True)
        extract_splits(zf, patients)
        write_summary(patients)
        print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
