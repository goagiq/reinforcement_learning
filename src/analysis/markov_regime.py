"""Offline Markov-style regime analysis utilities.

Provides a lightweight pipeline that can be run while PPO training is in
progress. The analyzer extracts historical features, clusters market regimes,
and estimates a transition matrix that can later inform downstream agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data_extraction import DataExtractor


@dataclass
class RegimeConfig:
    """Configuration for the Markov regime analyzer."""

    instrument: str = "ES"
    timeframes: Iterable[int] = (1,)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rolling_vol_window: int = 50
    volume_zscore_window: int = 100
    num_regimes: int = 3
    random_state: int = 42
    min_samples: int = 500


@dataclass
class RegimeAnalysisResult:
    """Container for regime analysis outputs."""

    transition_matrix: pd.DataFrame
    stationary_distribution: pd.Series
    regime_summary: pd.DataFrame
    clustered_data: pd.DataFrame = field(repr=False)

    def to_dict(self) -> Dict:
        """Serialize results to JSON-serializable dict."""
        return {
            "transition_matrix": self.transition_matrix.to_dict(),
            "stationary_distribution": self.stationary_distribution.to_dict(),
            "regime_summary": self.regime_summary.to_dict(orient="index"),
        }


class MarkovRegimeAnalyzer:
    """Run offline clustering and transition estimation."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._extractor = DataExtractor()

    def run(self) -> RegimeAnalysisResult:
        """Execute the full analysis pipeline."""
        df = self._load_primary_timeframe()
        prepared = self._prepare_features(df)
        labeled = self._cluster_regimes(prepared)
        transition_matrix = self._estimate_transition_matrix(labeled["regime_id"].values)
        stationary = self._estimate_stationary_distribution(transition_matrix)
        summary = self._summarize_regimes(labeled)

        return RegimeAnalysisResult(
            transition_matrix=pd.DataFrame(
                transition_matrix,
                index=summary.index,
                columns=summary.index,
            ),
            stationary_distribution=pd.Series(stationary, index=summary.index, name="stationary_prob"),
            regime_summary=summary,
            clustered_data=labeled,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_primary_timeframe(self) -> pd.DataFrame:
        """Load the highest-frequency timeframe from disk."""
        timeframes = list(self.config.timeframes)
        if not timeframes:
            raise ValueError("At least one timeframe must be specified for regime analysis.")

        primary_tf = min(timeframes)
        df = self._extractor.load_historical_data(
            instrument=self.config.instrument,
            timeframe=primary_tf,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        if len(df) < self.config.min_samples:
            raise ValueError(
                f"Insufficient samples for regime clustering ({len(df)} < {self.config.min_samples}). "
                "Extend the historical window before running analysis."
            )

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute returns, volatility and normalized volume."""
        prepared = df.copy()
        prepared["log_close"] = np.log(prepared["close"])
        prepared["log_return"] = prepared["log_close"].diff()
        prepared["return"] = prepared["close"].pct_change()
        prepared["abs_return"] = prepared["return"].abs()

        # Rolling volatility proxy
        vol_window = max(5, self.config.rolling_vol_window)
        prepared["rolling_vol"] = prepared["return"].rolling(vol_window).std()

        # Volume z-score to highlight participation spikes
        vol_z_window = max(20, self.config.volume_zscore_window)
        volume_rolling_mean = prepared["volume"].rolling(vol_z_window).mean()
        volume_rolling_std = prepared["volume"].rolling(vol_z_window).std()
        prepared["volume_zscore"] = (prepared["volume"] - volume_rolling_mean) / volume_rolling_std

        # Fill missing values with reasonable defaults
        prepared = prepared.dropna(subset=["log_return"]).copy()
        prepared.loc[:, "rolling_vol"] = prepared["rolling_vol"].fillna(prepared["rolling_vol"].median())
        prepared.loc[:, "volume_zscore"] = prepared["volume_zscore"].fillna(0.0)
        prepared.loc[:, "return"] = prepared["return"].fillna(0.0)
        prepared.loc[:, "abs_return"] = prepared["abs_return"].fillna(0.0)

        return prepared

    def _cluster_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cluster the market regimes using KMeans."""
        features = df[["return", "rolling_vol", "volume_zscore"]].values
        kmeans = KMeans(
            n_clusters=self.config.num_regimes,
            random_state=self.config.random_state,
            n_init="auto",
        )
        raw_labels = kmeans.fit_predict(features)

        # Order regimes by mean return for consistent semantics
        label_order = self._order_labels_by_return(df["return"].values, raw_labels)
        label_map = {raw_label: f"Regime_{rank}" for raw_label, rank in label_order.items()}
        ordered_ids = np.vectorize(label_order.get)(raw_labels)

        labeled = df.copy()
        labeled["regime_id"] = ordered_ids
        labeled["regime_label"] = np.vectorize(label_map.get)(raw_labels)
        return labeled

    @staticmethod
    def _order_labels_by_return(returns: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
        """Assign deterministic ranking: lowest return -> 0 ... highest -> k-1."""
        unique_labels = np.unique(labels)
        avg_returns = {label: returns[labels == label].mean() for label in unique_labels}
        sorted_labels = sorted(avg_returns.items(), key=lambda x: x[1])
        return {label: rank for rank, (label, _) in enumerate(sorted_labels)}

    def _estimate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Compute empirical transition probabilities."""
        k = self.config.num_regimes
        counts = np.zeros((k, k), dtype=np.float64)

        for current_state, next_state in zip(regimes[:-1], regimes[1:]):
            counts[current_state, next_state] += 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = counts.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

        return transition_matrix

    @staticmethod
    def _estimate_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
        """Estimate the stationary distribution via eigen decomposition."""
        eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        stationary = np.clip(stationary, 0, None)
        stationary = stationary / stationary.sum()
        return stationary

    def _summarize_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate descriptive statistics for each regime."""
        summary = (
            df.groupby("regime_id")
            .agg(
                mean_return=("return", "mean"),
                median_return=("return", "median"),
                volatility=("rolling_vol", "mean"),
                avg_volume_z=("volume_zscore", "mean"),
                frequency=("return", "size"),
            )
            .sort_index()
        )
        summary["frequency_pct"] = summary["frequency"] / summary["frequency"].sum()
        return summary


def run_and_save_report(
    output_path: Path,
    config: Optional[RegimeConfig] = None,
) -> RegimeAnalysisResult:
    """Convenience wrapper to run the analyzer and persist a JSON report."""
    analyzer = MarkovRegimeAnalyzer(config=config)
    result = analyzer.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = result.to_dict()
    report["metadata"] = {
        "instrument": analyzer.config.instrument,
        "timeframes": list(analyzer.config.timeframes),
        "num_regimes": analyzer.config.num_regimes,
    }
    output_path.write_text(json.dumps(report, indent=2, default=float))

    return result


__all__ = [
    "RegimeConfig",
    "RegimeAnalysisResult",
    "MarkovRegimeAnalyzer",
    "run_and_save_report",
]


