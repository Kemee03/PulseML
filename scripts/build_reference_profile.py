"""Build a compact per-feature PSI reference from KeplerVision's exoTrain.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COUNT = 3197
LABELS = ("LABEL",)
FEATURES = tuple(f"FLUX.{i}" for i in range(1, FEATURE_COUNT + 1))
QUANTILES = np.linspace(0.0, 1.0, 11)


def build_profile(source: Path, destination: Path, *, normal_sample: int = 4000) -> None:
    labels = pd.read_csv(source, usecols=list(LABELS))["LABEL"].to_numpy()
    normal_rows = np.flatnonzero(labels == 1)
    planet_rows = np.flatnonzero(labels == 2)
    if not len(normal_rows) or not len(planet_rows):
        raise ValueError("The training CSV must contain both LABEL 1 and LABEL 2 rows.")
    rng = np.random.default_rng(42)
    chosen = np.concatenate((
        rng.choice(normal_rows, size=min(normal_sample, len(normal_rows)), replace=False),
        planet_rows,
    ))
    selected = set(map(int, chosen))
    frame = pd.read_csv(source, skiprows=lambda line: line > 0 and line - 1 not in selected)
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing {len(missing)} required flux columns.")
    values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Training flux values must all be finite numbers.")

    quantile_values = np.quantile(values, QUANTILES, axis=0)
    cut_points: list[float] = []
    edge_offsets = [0]
    expected_counts: list[int] = []
    count_offsets = [0]
    for index in range(FEATURE_COUNT):
        cuts = np.unique(quantile_values[1:-1, index])
        bins = np.concatenate(([-np.inf], cuts, [np.inf]))
        counts = np.histogram(values[:, index], bins=bins)[0]
        cut_points.extend(cuts.tolist())
        edge_offsets.append(len(cut_points))
        expected_counts.extend(counts.tolist())
        count_offsets.append(len(expected_counts))

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=np.asarray(FEATURES),
        cut_points=np.asarray(cut_points, dtype=np.float64),
        edge_offsets=np.asarray(edge_offsets, dtype=np.int32),
        expected_counts=np.asarray(expected_counts, dtype=np.int64),
        count_offsets=np.asarray(count_offsets, dtype=np.int32),
        sample_count=np.asarray([len(values)], dtype=np.int32),
        source_rows=np.asarray([len(labels)], dtype=np.int32),
        normal_sample=np.asarray([min(normal_sample, len(normal_rows))], dtype=np.int32),
        planet_sample=np.asarray([len(planet_rows)], dtype=np.int32),
    )
    print(f"Saved reference for {FEATURE_COUNT} features and {len(values):,} rows: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path, help="KeplerVision exoTrain.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "kepler_reference.npz")
    args = parser.parse_args()
    build_profile(args.training_csv, args.output)


if __name__ == "__main__":
    main()
