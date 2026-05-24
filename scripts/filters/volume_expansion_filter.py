"""Volume expansion filter.

The filter keeps stocks where:
    Today's volume > multiplier * 30-day average volume

This helps retain stocks where participation is expanding.
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

DEFAULT_VOLUME_CONFIG: dict[str, Any] = {
    "multiplier": 1.5,
    "average_days": 30,
}


def load_volume_filter_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read volume filter settings from config/screener_config.json."""

    if not config_path.exists() or config_path.stat().st_size == 0:
        logger.warning(
            "Volume config file is missing or empty. Using default settings."
        )
        return DEFAULT_VOLUME_CONFIG.copy()

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read volume config. Using defaults: %s", exc)
        return DEFAULT_VOLUME_CONFIG.copy()

    volume_config = config.get("volume_expansion_filter", DEFAULT_VOLUME_CONFIG)
    if not isinstance(volume_config, dict):
        logger.warning("Volume config is malformed. Using default settings.")
        return DEFAULT_VOLUME_CONFIG.copy()

    return volume_config


def filter_volume_expansion_stocks(
    stock_df: pd.DataFrame,
    config_path: Path = DEFAULT_CONFIG_PATH,
    symbol_column: str = "symbol",
    volume_column: str = "volume",
    date_column: str | None = "date",
    multiplier: float | None = None,
    average_days: int | None = None,
) -> pd.DataFrame:
    """Return stocks with volume above the configured 30-day average multiple.

    The input should be a processed dataframe with historical volume rows for
    each stock. The output contains only the latest passing row for each symbol.
    """

    volume_config = load_volume_filter_config(config_path)
    configured_multiplier = float(
        multiplier
        if multiplier is not None
        else volume_config.get("multiplier", DEFAULT_VOLUME_CONFIG["multiplier"])
    )
    configured_average_days = int(
        average_days
        if average_days is not None
        else volume_config.get("average_days", DEFAULT_VOLUME_CONFIG["average_days"])
    )

    if configured_multiplier <= 0:
        raise ValueError("Volume multiplier must be greater than 0.")

    if configured_average_days < 1:
        raise ValueError("Volume average_days must be at least 1.")

    required_columns = [symbol_column, volume_column]
    if date_column is not None:
        required_columns.append(date_column)

    missing_columns = [
        column for column in required_columns if column not in stock_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Volume expansion filter missing required columns: "
            + ", ".join(missing_columns)
        )

    if stock_df.empty:
        logger.info("Volume expansion filter received an empty dataframe.")
        return stock_df.copy()

    working_df = stock_df.copy()
    working_df[volume_column] = pd.to_numeric(
        working_df[volume_column],
        errors="coerce",
    )

    # Sorting ensures the rolling average uses each stock's volume history in
    # chronological order.
    sort_columns = [symbol_column]
    if date_column is not None:
        sort_columns.append(date_column)
    working_df = working_df.sort_values(sort_columns)

    average_volume_column = f"avg_volume_{configured_average_days}"
    working_df[average_volume_column] = (
        working_df.groupby(symbol_column)[volume_column]
        .transform(
            lambda volume_series: volume_series.rolling(
                window=configured_average_days,
                min_periods=configured_average_days,
            ).mean().shift(1)
        )
    )

    latest_rows = working_df.groupby(symbol_column, as_index=False).tail(1).copy()

    # Stocks without enough prior history have NaN average volume and will not
    # pass. The current day's volume is not included in its own average.
    volume_expanded = (
        latest_rows[volume_column]
        > configured_multiplier * latest_rows[average_volume_column]
    )
    latest_rows["volume_expansion"] = (
        latest_rows[volume_column] / latest_rows[average_volume_column]
    )

    filtered_df = latest_rows.loc[volume_expanded].copy()

    logger.info(
        "Volume expansion filter retained %s of %s symbols using multiplier %s.",
        len(filtered_df),
        latest_rows[symbol_column].nunique(),
        configured_multiplier,
    )

    return filtered_df
