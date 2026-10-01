"""Population Stability Index against the sampled KeplerVision training rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_COUNT = 3197
MINIMUM_PRODUCTION_ROWS = 1000


class ReferenceProfile:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.features = data["features"]
            self.cut_points = data["cut_points"]
            self.edge_offsets = data["edge_offsets"]
            self.expected_counts = data["expected_counts"]
            self.count_offsets = data["count_offsets"]
            self.sample_count = int(data["sample_count"][0])
            self.source_rows = int(data["source_rows"][0])

        if len(self.features) != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} reference features, found {len(self.features)}.")
        if len(self.edge_offsets) != FEATURE_COUNT + 1 or len(self.count_offsets) != FEATURE_COUNT + 1:
            raise ValueError("Reference profile offsets are invalid.")

    def score(self, observations: np.ndarray) -> dict:
        if observations.ndim != 2 or observations.shape[1] != FEATURE_COUNT:
            raise ValueError(f"Expected a matrix with {FEATURE_COUNT} features.")
        if len(observations) < MINIMUM_PRODUCTION_ROWS:
            return {
                "status": "insufficient_data",
                "severity": "unknown",
                "sample_count": int(len(observations)),
                "minimum_sample_count": MINIMUM_PRODUCTION_ROWS,
                "reference_sample_count": self.sample_count,
                "source_training_rows": self.source_rows,
                "top_features": [],
                "message": f"Collect at least {MINIMUM_PRODUCTION_ROWS} production predictions for a drift check.",
            }

        feature_scores: list[tuple[str, float]] = []
        for index, name in enumerate(self.features):
            edge_start, edge_end = self.edge_offsets[index:index + 2]
            count_start, count_end = self.count_offsets[index:index + 2]
            cuts = self.cut_points[edge_start:edge_end]
            expected = self.expected_counts[count_start:count_end].astype(np.float64)
            actual = np.histogram(observations[:, index], bins=np.concatenate(([-np.inf], cuts, [np.inf])))[0]
            if len(actual) != len(expected):
                raise ValueError(f"Reference bins are invalid for feature {name}.")
            # A small pseudocount keeps empty bins finite with limited live samples.
            expected_share = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
            actual_share = (actual + 0.5) / (actual.sum() + 0.5 * len(actual))
            psi = float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))
            feature_scores.append((str(name), psi))

        feature_scores.sort(key=lambda item: item[1], reverse=True)
        worst_psi = feature_scores[0][1]
        if worst_psi >= 0.25:
            severity = "high"
        elif worst_psi >= 0.10:
            severity = "moderate"
        else:
            severity = "normal"
        return {
            "status": "ready",
            "severity": severity,
            "sample_count": int(len(observations)),
            "minimum_sample_count": MINIMUM_PRODUCTION_ROWS,
            "reference_sample_count": self.sample_count,
            "source_training_rows": self.source_rows,
            "top_features": [
                {"feature": name, "psi": round(score, 4),
                 "severity": "high" if score >= 0.25 else "moderate" if score >= 0.10 else "normal"}
                for name, score in feature_scores[:10]
            ],
            "message": "PSI compares current production feature distributions with a sampled AstroLens training reference. It is a signal to investigate, not proof of model failure.",
        }
