"""Momentum persistence ranking engine.

Momentum formula:
    0.4 * 6-month return
    + 0.3 * 3-month return
    + 0.2 * 1-month return

The engine rejects negative momentum, adds a normalized score, adds percentile
ranking, and can export the ranked dataframe.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calculate_momentum_persistence(
    stock_df: pd.DataFrame,
    return_6m_column: str = "return_6m",
    return_3m_column: str = "return_3m",
    return_1m_column: str = "return_1m",
    momentum_column: str = "momentum",
) -> pd.DataFrame:
    """Add the weighted momentum score to a processed stock dataframe."""

    required_columns = [return_6m_column, return_3m_column, return_1m_column]
    _validate_columns(stock_df, required_columns)

    ranked_df = stock_df.copy()
    for column in required_columns:
        ranked_df[column] = pd.to_numeric(ranked_df[column], errors="coerce")

    # Weighted momentum rewards persistence across multiple timeframes.
    ranked_df[momentum_column] = (
        0.4 * ranked_df[return_6m_column]
        + 0.3 * ranked_df[return_3m_column]
        + 0.2 * ranked_df[return_1m_column]
    )

    return ranked_df


def rank_momentum_persistence(
    stock_df: pd.DataFrame,
    return_6m_column: str = "return_6m",
    return_3m_column: str = "return_3m",
    return_1m_column: str = "return_1m",
    momentum_column: str = "momentum",
    normalized_score_column: str = "momentum_score",
    zscore_column: str = "momentum_zscore",
    percentile_column: str = "momentum_percentile",
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Return a ranked dataframe after rejecting negative momentum rows."""

    if stock_df.empty:
        logger.info("Momentum ranking received an empty dataframe.")
        return stock_df.copy()

    ranked_df = calculate_momentum_persistence(
        stock_df=stock_df,
        return_6m_column=return_6m_column,
        return_3m_column=return_3m_column,
        return_1m_column=return_1m_column,
        momentum_column=momentum_column,
    )

    before_count = len(ranked_df)

    ranked_df = ranked_df.dropna(subset=[momentum_column])

    # Negative or zero momentum means the stock is not persistent enough for
    # this model. This keeps the final book focused on positive leadership.
    ranked_df = ranked_df.loc[ranked_df[momentum_column] > 0].copy()

    if ranked_df.empty:
        logger.info("Momentum ranking rejected all %s rows.", before_count)
        return ranked_df

    ranked_df[zscore_column] = _zscore(ranked_df[momentum_column])
    ranked_df[normalized_score_column] = _normalize_to_100(ranked_df[momentum_column])

    # Percentile rank shows where each stock stands versus the current universe.
    ranked_df[percentile_column] = (
        ranked_df[momentum_column].rank(method="average", pct=True) * 100
    )

    sort_columns = [momentum_column, percentile_column]
    ascending = [False, False]
    if "symbol" in ranked_df.columns:
        sort_columns.append("symbol")
        ascending.append(True)

    ranked_df = ranked_df.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df[rank_column] = np.arange(1, len(ranked_df) + 1)

    logger.info(
        "Momentum ranking retained %s of %s rows.",
        len(ranked_df),
        before_count,
    )

    return ranked_df


def export_ranked_momentum_dataframe(
    ranked_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export the ranked dataframe to CSV and return the saved path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(path, index=False)

    logger.info("Exported ranked momentum dataframe to %s.", path)
    return path


def build_and_export_momentum_ranking(
    stock_df: pd.DataFrame,
    output_path: str | Path,
    return_6m_column: str = "return_6m",
    return_3m_column: str = "return_3m",
    return_1m_column: str = "return_1m",
) -> pd.DataFrame:
    """Rank stocks by momentum persistence, export them, and return the result."""

    ranked_df = rank_momentum_persistence(
        stock_df=stock_df,
        return_6m_column=return_6m_column,
        return_3m_column=return_3m_column,
        return_1m_column=return_1m_column,
    )
    export_ranked_momentum_dataframe(ranked_df, output_path)
    return ranked_df


def _normalize_to_100(values: pd.Series) -> pd.Series:
    """Scale a numeric series to a 0-100 score."""

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        # If all passing stocks have equal momentum, give them full credit.
        return pd.Series(100.0, index=values.index)

    return ((values - minimum) / (maximum - minimum)) * 100


def _zscore(values: pd.Series) -> pd.Series:
    """Return z-score normalization with division-by-zero protection."""

    standard_deviation = values.std(ddof=0)

    if standard_deviation == 0 or pd.isna(standard_deviation):
        return pd.Series(0.0, index=values.index)

    return (values - values.mean()) / standard_deviation


def _validate_columns(stock_df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error if required return columns are missing."""

    missing_columns = [
        column for column in required_columns if column not in stock_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Momentum ranking missing required columns: "
            + ", ".join(missing_columns)
        )
