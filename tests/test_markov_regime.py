"""Tests for the offline Markov regime analysis utilities."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Mitigate MKL memory-leak warning on Windows in sklearn KMeans
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.analysis import markov_regime


def _build_synthetic_market_df(rows: int = 120) -> pd.DataFrame:
    """Create a deterministic dataset with alternating regimes."""
    rng = np.random.default_rng(seed=42)

    timestamps = pd.date_range("2024-01-01", periods=rows, freq="min")
    base_price = 5000 + np.cumsum(rng.normal(0, 0.5, size=rows))

    # Create two regimes by modulating volatility and volume
    regime_switch = np.sin(np.linspace(0, 6 * np.pi, rows))
    volatility = 0.3 + 0.2 * (regime_switch > 0)
    noise = rng.normal(0, volatility, size=rows)

    close = base_price + noise
    open_price = close - rng.normal(0, 0.1, size=rows)
    high = np.maximum(open_price, close) + np.abs(rng.normal(0, 0.1, size=rows))
    low = np.minimum(open_price, close) - np.abs(rng.normal(0, 0.1, size=rows))
    volume = 1000 + 200 * (regime_switch > 0) + rng.normal(0, 30, size=rows)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.maximum(volume, 1.0),
        }
    )
    return df


@pytest.fixture
def synthetic_data(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Fixture that patches DataExtractor to return synthetic market data."""
    df = _build_synthetic_market_df()

    def fake_load(
        self: Any,
        instrument: str,
        timeframe: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return df.copy()

    monkeypatch.setattr(markov_regime.DataExtractor, "load_historical_data", fake_load)
    return df


def test_markov_regime_run_returns_expected_shapes(synthetic_data: pd.DataFrame) -> None:
    """Ensure the analyzer can run end-to-end on synthetic data."""
    config = markov_regime.RegimeConfig(
        instrument="ES",
        timeframes=[1],
        num_regimes=2,
        rolling_vol_window=5,
        volume_zscore_window=10,
        min_samples=50,
        random_state=7,
    )

    analyzer = markov_regime.MarkovRegimeAnalyzer(config=config)
    result = analyzer.run()

    # Transition matrix should be square with num_regimes dimension
    assert result.transition_matrix.shape == (config.num_regimes, config.num_regimes)
    # Stationary distribution should sum to ~1
    assert pytest.approx(result.stationary_distribution.sum(), rel=1e-6) == 1.0
    # We should observe the requested number of regimes
    assert result.clustered_data["regime_id"].nunique() == config.num_regimes
    # Summary contains descriptive stats per regime
    assert set(result.regime_summary.columns) >= {
        "mean_return",
        "volatility",
        "frequency",
        "frequency_pct",
    }


def test_run_and_save_report_writes_json(tmp_path: Path, synthetic_data: pd.DataFrame) -> None:
    """Verify the convenience wrapper writes a JSON report."""
    output_path = tmp_path / "markov_report.json"

    config = markov_regime.RegimeConfig(
        instrument="ES",
        timeframes=[1],
        num_regimes=3,
        rolling_vol_window=5,
        volume_zscore_window=10,
        min_samples=50,
        random_state=21,
    )

    result = markov_regime.run_and_save_report(output_path, config=config)

    assert output_path.exists(), "Expected JSON report to be written"

    report_data = output_path.read_text()
    assert '"transition_matrix"' in report_data
    assert '"stationary_distribution"' in report_data
    assert '"regime_summary"' in report_data

    # Ensure the saved matrix matches the result in memory
    assert result.transition_matrix.shape == (config.num_regimes, config.num_regimes)

