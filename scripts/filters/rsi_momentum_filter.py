"""RSI momentum filter.

The filter keeps stocks where:
    1. RSI is inside the configured mode bounds.
    2. Today's RSI has not weakened meaningfully versus 3 days ago.

The second rule is:
    RSI today >= RSI 3 days ago - 1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "screener_config.json"

DEFAULT_RSI_CONFIG: dict[str, Any] = {
    "active_mode": "moderate",
    "modes": {
        "conservative": {"min": 55, "max": 70},
        "moderate": {"min": 50, "max": 75},
        "aggressive": {"min": 45, "max": 80},
    },
    "lookback_days": 3,
    "allowed_drop": 1,
}


def load_rsi_filter_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read RSI filter settings from config/screener_config.json."""

    if not config_path.exists() or config_path.stat().st_size == 0:
        logger.warning(
            "RSI config file is missing or empty. Using default RSI settings."
        )
        return DEFAULT_RSI_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    rsi_config = config.get("rsi_momentum_filter", DEFAULT_RSI_CONFIG)
    return rsi_config


def get_rsi_mode_settings(
    rsi_config: dict[str, Any],
    mode: str | None = None,
) -> tuple[str, float, float, int, float]:
    """Return normalized RSI settings for the selected mode."""

    selected_mode = (mode or rsi_config.get("active_mode", "moderate")).lower()
    modes = rsi_config.get("modes", {})

    if selected_mode not in modes:
        raise ValueError(
            f"Unknown RSI mode '{selected_mode}'. "
            f"Available modes: {', '.join(sorted(modes))}"
        )

    mode_settings = modes[selected_mode]
    lower_bound = float(mode_settings["min"])
    upper_bound = float(mode_settings["max"])
    lookback_days = int(rsi_config.get("lookback_days", 3))
    allowed_drop = float(rsi_config.get("allowed_drop", 1))

    if lower_bound > upper_bound:
        raise ValueError("RSI lower bound cannot be greater than upper bound.")

    if lookback_days < 1:
        raise ValueError("RSI lookback_days must be at least 1.")

    return selected_mode, lower_bound, upper_bound, lookback_days, allowed_drop


def filter_rsi_momentum_stocks(
    stock_df: pd.DataFrame,
    mode: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    symbol_column: str = "symbol",
    rsi_column: str = "rsi",
    date_column: str | None = "date",
) -> pd.DataFrame:
    """Return stocks passing the configured RSI momentum filter.

    The input should be a processed stock dataframe with one row per symbol per
    date, or one latest row per symbol plus enough recent RSI history.
    """

    rsi_config = load_rsi_filter_config(config_path)
    selected_mode, lower_bound, upper_bound, lookback_days, allowed_drop = (
        get_rsi_mode_settings(rsi_config, mode)
    )

    required_columns = [symbol_column, rsi_column]
    if date_column is not None:
        required_columns.append(date_column)

    missing_columns = [
        column for column in required_columns if column not in stock_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "RSI momentum filter missing required columns: "
            + ", ".join(missing_columns)
        )

    if stock_df.empty:
        logger.info("RSI momentum filter received an empty dataframe.")
        return stock_df.copy()

    working_df = stock_df.copy()

    # Sorting makes groupby().shift() compare each row with the same stock's RSI
    # from 3 rows earlier. With daily processed data, that means 3 trading days.
    sort_columns = [symbol_column]
    if date_column is not None:
        sort_columns.append(date_column)
    working_df = working_df.sort_values(sort_columns)

    previous_rsi_column = f"{rsi_column}_{lookback_days}_days_ago"
    working_df[previous_rsi_column] = (
        working_df.groupby(symbol_column)[rsi_column].shift(lookback_days)
    )

    latest_rows = working_df.groupby(symbol_column, as_index=False).tail(1)

    rsi_in_bounds = latest_rows[rsi_column].between(
        lower_bound,
        upper_bound,
        inclusive="both",
    )
    rsi_not_weakening = (
        latest_rows[rsi_column]
        >= latest_rows[previous_rsi_column] - allowed_drop
    )

    filtered_df = latest_rows.loc[rsi_in_bounds & rsi_not_weakening].copy()

    logger.info(
        "RSI momentum filter (%s) retained %s of %s symbols.",
        selected_mode,
        len(filtered_df),
        latest_rows[symbol_column].nunique(),
    )

    return filtered_df.drop(columns=[previous_rsi_column])

