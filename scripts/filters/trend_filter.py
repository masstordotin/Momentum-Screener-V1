"""Trend filter for structurally bullish stocks.

The rule is simple:
    Close > EMA50 > EMA200

Only rows that satisfy this rule are retained.
"""

from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def filter_structurally_bullish_stocks(
    stock_df: pd.DataFrame,
    close_column: str = "close",
    ema50_column: str = "ema50",
    ema200_column: str = "ema200",
) -> pd.DataFrame:
    """Return only stocks where Close > EMA50 > EMA200.

    Args:
        stock_df: Processed stock dataframe containing close and EMA columns.
        close_column: Name of the close-price column.
        ema50_column: Name of the EMA 50 column.
        ema200_column: Name of the EMA 200 column.

    Returns:
        A filtered copy of the input dataframe.
    """

    required_columns = [close_column, ema50_column, ema200_column]
    missing_columns = [
        column for column in required_columns if column not in stock_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Trend filter missing required columns: "
            + ", ".join(missing_columns)
        )

    if stock_df.empty:
        logger.info("Trend filter received an empty dataframe.")
        return stock_df.copy()

    # This boolean mask is True only for rows in a bullish long-term structure.
    bullish_mask = (
        (stock_df[close_column] > stock_df[ema50_column])
        & (stock_df[ema50_column] > stock_df[ema200_column])
    )

    filtered_df = stock_df.loc[bullish_mask].copy()

    logger.info(
        "Trend filter retained %s of %s rows.",
        len(filtered_df),
        len(stock_df),
    )

    return filtered_df

