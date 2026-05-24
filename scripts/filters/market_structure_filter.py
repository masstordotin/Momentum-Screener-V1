"""Higher High Higher Low market structure filter.

The filter keeps stocks whose recent swing highs and swing lows are rising.
This is a simple way to reject structurally weak or sideways trends.
"""

from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def detect_local_peaks(
    stock_df: pd.DataFrame,
    high_column: str = "high",
    window: int = 2,
) -> pd.DataFrame:
    """Return rows where the high is greater than nearby highs."""

    _validate_window(window)
    _validate_columns(stock_df, [high_column], "peak detection")

    highs = pd.to_numeric(stock_df[high_column], errors="coerce")
    peak_mask = pd.Series(True, index=stock_df.index)

    # Compare each high with the same number of candles on the left and right.
    for offset in range(1, window + 1):
        peak_mask &= highs > highs.shift(offset)
        peak_mask &= highs > highs.shift(-offset)

    return stock_df.loc[peak_mask].copy()


def detect_local_troughs(
    stock_df: pd.DataFrame,
    low_column: str = "low",
    window: int = 2,
) -> pd.DataFrame:
    """Return rows where the low is lower than nearby lows."""

    _validate_window(window)
    _validate_columns(stock_df, [low_column], "trough detection")

    lows = pd.to_numeric(stock_df[low_column], errors="coerce")
    trough_mask = pd.Series(True, index=stock_df.index)

    # A trough is a local low: lower than nearby candles on both sides.
    for offset in range(1, window + 1):
        trough_mask &= lows < lows.shift(offset)
        trough_mask &= lows < lows.shift(-offset)

    return stock_df.loc[trough_mask].copy()


def has_higher_high_higher_low_structure(
    stock_df: pd.DataFrame,
    high_column: str = "high",
    low_column: str = "low",
    date_column: str | None = "date",
    window: int = 2,
    required_swings: int = 2,
) -> bool:
    """Return True when recent peaks and troughs are both moving upward."""

    _validate_window(window)
    if required_swings < 2:
        raise ValueError("required_swings must be at least 2.")

    required_columns = [high_column, low_column]
    if date_column is not None:
        required_columns.append(date_column)
    _validate_columns(stock_df, required_columns, "HH-HL structure filter")

    if len(stock_df) < (window * 2) + required_swings:
        return False

    working_df = stock_df.copy()
    if date_column is not None:
        working_df = working_df.sort_values(date_column)
    working_df[high_column] = pd.to_numeric(working_df[high_column], errors="coerce")
    working_df[low_column] = pd.to_numeric(working_df[low_column], errors="coerce")

    peaks = detect_local_peaks(working_df, high_column, window)
    troughs = detect_local_troughs(working_df, low_column, window)

    if len(peaks) < required_swings or len(troughs) < required_swings:
        return False

    recent_peaks = peaks.tail(required_swings)
    recent_troughs = troughs.tail(required_swings)

    # HH-HL structure requires strictly rising swings. Equal highs/lows are
    # not structural progress and should not pass as leadership.
    higher_highs = (recent_peaks[high_column].diff().dropna() > 0).all()
    higher_lows = (recent_troughs[low_column].diff().dropna() > 0).all()

    return bool(higher_highs and higher_lows)


def filter_higher_high_higher_low_stocks(
    stock_df: pd.DataFrame,
    symbol_column: str = "symbol",
    high_column: str = "high",
    low_column: str = "low",
    date_column: str | None = "date",
    window: int = 2,
    required_swings: int = 2,
) -> pd.DataFrame:
    """Return latest rows for stocks with a valid HH-HL upward structure."""

    required_columns = [symbol_column, high_column, low_column]
    if date_column is not None:
        required_columns.append(date_column)
    _validate_columns(stock_df, required_columns, "HH-HL structure filter")

    if stock_df.empty:
        logger.info("HH-HL structure filter received an empty dataframe.")
        return stock_df.copy()

    passing_symbols: list[str] = []

    for symbol, symbol_df in stock_df.groupby(symbol_column):
        if has_higher_high_higher_low_structure(
            symbol_df,
            high_column=high_column,
            low_column=low_column,
            date_column=date_column,
            window=window,
            required_swings=required_swings,
        ):
            passing_symbols.append(symbol)

    if not passing_symbols:
        logger.info("HH-HL structure filter retained 0 symbols.")
        return stock_df.iloc[0:0].copy()

    passing_df = stock_df.loc[stock_df[symbol_column].isin(passing_symbols)].copy()

    sort_columns = [symbol_column]
    if date_column is not None:
        sort_columns.append(date_column)
    passing_df = passing_df.sort_values(sort_columns)

    latest_rows = passing_df.groupby(symbol_column, as_index=False).tail(1)

    logger.info(
        "HH-HL structure filter retained %s of %s symbols.",
        len(passing_symbols),
        stock_df[symbol_column].nunique(),
    )

    return latest_rows.copy()


def _validate_columns(
    stock_df: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    """Raise a clear error if the dataframe is missing needed columns."""

    missing_columns = [
        column for column in required_columns if column not in stock_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{label} missing required columns: "
            + ", ".join(missing_columns)
        )


def _validate_window(window: int) -> None:
    """Make sure local swing detection has a usable look-around window."""

    if window < 1:
        raise ValueError("window must be at least 1.")
