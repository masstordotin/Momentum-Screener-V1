"""Build the processed stock dataframe from raw NSE bhavcopy files.

This module is intentionally compact and boring: raw NSE CSV files are cleaned,
validated, de-duplicated, and then enriched with the indicators required by the
screening pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nifty500_eod"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_stocks.csv"

logger = logging.getLogger(__name__)

RAW_COLUMN_MAP = {
    "SYMBOL": "symbol",
    "DATE1": "date",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "CLOSE_PRICE": "close",
    "TTL_TRD_QNTY": "volume",
}


def build_processed_stock_dataframe(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Create and save the processed dataframe used by the main pipeline."""

    raw_df = load_raw_bhavcopy_data(raw_dir)
    clean_df = clean_bhavcopy_dataframe(raw_df)
    processed_df = add_indicators(clean_df)
    validate_processed_dataframe(processed_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    logger.info("Saved processed stock dataframe to %s.", output_path)

    return processed_df


def load_raw_bhavcopy_data(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Read all accumulated NIFTY 500 bhavcopy CSV files."""

    files = sorted(raw_dir.glob("*/nifty500_bhavcopy/nifty500_eod_*.csv"))
    if not files:
        raise FileNotFoundError(f"No NIFTY 500 bhavcopy files found in {raw_dir}")

    frames = [pd.read_csv(file) for file in files]
    logger.info("Loaded %s bhavcopy files.", len(frames))
    return pd.concat(frames, ignore_index=True)


def clean_bhavcopy_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NSE columns and remove rows that cannot be trusted."""

    raw_df = raw_df.rename(columns=lambda column: str(column).strip())
    missing_columns = [
        column for column in RAW_COLUMN_MAP if column not in raw_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Raw bhavcopy data missing required columns: "
            + ", ".join(missing_columns)
        )

    clean_df = raw_df[list(RAW_COLUMN_MAP)].rename(columns=RAW_COLUMN_MAP).copy()
    clean_df["symbol"] = clean_df["symbol"].astype(str).str.strip().str.upper()
    clean_df["date"] = pd.to_datetime(
        clean_df["date"].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    for column in ["high", "low", "close", "volume"]:
        clean_df[column] = pd.to_numeric(
            clean_df[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    clean_df = clean_df.dropna(subset=["symbol", "date", "high", "low", "close"])
    clean_df = clean_df.loc[clean_df["symbol"] != ""]
    clean_df = clean_df.loc[clean_df["close"] > 0]
    clean_df = clean_df.loc[clean_df["high"] >= clean_df["low"]]
    clean_df = clean_df.loc[clean_df["volume"].fillna(0) >= 0]

    # NSE archives can be re-downloaded in multiple runs. Keep one row per
    # symbol/date so indicators are not distorted by duplicate data.
    clean_df = clean_df.drop_duplicates(
        subset=["symbol", "date"],
        keep="last",
    ).sort_values(["symbol", "date"])

    if clean_df.empty:
        raise ValueError("Cleaned bhavcopy dataframe is empty.")

    return clean_df.reset_index(drop=True)


def add_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, RSI, returns, average volume, and volume expansion columns."""

    processed_df = stock_df.copy()
    grouped = processed_df.groupby("symbol", group_keys=False)

    processed_df["ema50"] = grouped["close"].transform(
        lambda series: series.ewm(span=50, adjust=False, min_periods=50).mean()
    )
    processed_df["ema200"] = grouped["close"].transform(
        lambda series: series.ewm(span=200, adjust=False, min_periods=200).mean()
    )
    processed_df["rsi"] = grouped["close"].transform(calculate_rsi)
    processed_df["return_1m"] = grouped["close"].transform(
        lambda series: series.pct_change(21) * 100
    )
    processed_df["return_3m"] = grouped["close"].transform(
        lambda series: series.pct_change(63) * 100
    )
    processed_df["return_6m"] = grouped["close"].transform(
        lambda series: series.pct_change(126) * 100
    )
    processed_df["avg_volume_30"] = grouped["volume"].transform(
        lambda series: series.rolling(window=30, min_periods=30).mean().shift(1)
    )
    processed_df["volume_expansion"] = (
        processed_df["volume"] / processed_df["avg_volume_30"]
    )

    return processed_df


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI without looking into future rows."""

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))


def validate_processed_dataframe(processed_df: pd.DataFrame) -> None:
    """Fail early if required pipeline columns are missing or unusable."""

    required_columns = [
        "symbol",
        "date",
        "close",
        "high",
        "low",
        "volume",
        "ema50",
        "ema200",
        "rsi",
        "return_1m",
        "return_3m",
        "return_6m",
        "avg_volume_30",
        "volume_expansion",
    ]
    missing_columns = [
        column for column in required_columns if column not in processed_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed dataframe missing required columns: "
            + ", ".join(missing_columns)
        )

    if processed_df[["symbol", "date", "close"]].isna().any().any():
        raise ValueError("Processed dataframe has missing symbol/date/close values.")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed stock indicators from raw NSE bhavcopy data."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    build_processed_stock_dataframe(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
