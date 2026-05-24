"""Master momentum screening pipeline.

Pipeline order:
    1. Trend Filter
    2. RSI Filter
    3. Volume Filter
    4. HH-HL Filter
    5. Momentum Persistence
    6. Ranking
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from .filters.market_structure_filter import (
        filter_higher_high_higher_low_stocks,
    )
    from .filters.rsi_momentum_filter import filter_rsi_momentum_stocks
    from .filters.trend_filter import filter_structurally_bullish_stocks
    from .filters.volume_expansion_filter import filter_volume_expansion_stocks
    from .rankings.momentum_persistence import rank_momentum_persistence
except ImportError:
    # Allows direct execution: python scripts/main.py
    from filters.market_structure_filter import filter_higher_high_higher_low_stocks
    from filters.rsi_momentum_filter import filter_rsi_momentum_stocks
    from filters.trend_filter import filter_structurally_bullish_stocks
    from filters.volume_expansion_filter import filter_volume_expansion_stocks
    from rankings.momentum_persistence import rank_momentum_persistence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_stocks.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "final_results.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "momentum_screener.log"

REQUIRED_PIPELINE_COLUMNS = [
    "symbol",
    "date",
    "close",
    "ema50",
    "ema200",
    "rsi",
    "volume",
    "high",
    "low",
    "return_1m",
    "return_3m",
    "return_6m",
]

logger = logging.getLogger(__name__)


def configure_logging(log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Configure readable console logging for the screening pipeline."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def load_processed_stock_dataframe(input_path: Path) -> pd.DataFrame:
    """Load the processed stock dataframe from CSV, JSON, or Parquet."""

    if not input_path.exists():
        raise FileNotFoundError(f"Processed input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        stock_df = pd.read_csv(input_path)
        return validate_processed_stock_dataframe(stock_df)

    if suffix == ".json":
        stock_df = pd.read_json(input_path)
        return validate_processed_stock_dataframe(stock_df)

    if suffix == ".parquet":
        stock_df = pd.read_parquet(input_path)
        return validate_processed_stock_dataframe(stock_df)

    raise ValueError(
        "Unsupported processed input format. Use .csv, .json, or .parquet."
    )


def validate_processed_stock_dataframe(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize processed data before any screening logic runs."""

    missing_columns = [
        column for column in REQUIRED_PIPELINE_COLUMNS if column not in stock_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed dataframe missing required columns: "
            + ", ".join(missing_columns)
        )

    if stock_df.empty:
        raise ValueError("Processed dataframe is empty.")

    clean_df = stock_df.copy()
    clean_df["symbol"] = clean_df["symbol"].astype(str).str.strip().str.upper()
    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")

    numeric_columns = [
        "close",
        "ema50",
        "ema200",
        "rsi",
        "volume",
        "high",
        "low",
        "return_1m",
        "return_3m",
        "return_6m",
    ]
    for column in numeric_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.dropna(subset=["symbol", "date", "close"])
    clean_df = clean_df.loc[clean_df["symbol"] != ""]
    clean_df = clean_df.loc[clean_df["close"] > 0]
    clean_df = clean_df.loc[clean_df["high"] >= clean_df["low"]]
    clean_df = clean_df.drop_duplicates(subset=["symbol", "date"], keep="last")

    if clean_df.empty:
        raise ValueError("Processed dataframe has no valid rows after validation.")

    logger.info("Validated processed dataframe with %s rows.", len(clean_df))
    return clean_df.sort_values(["symbol", "date"]).reset_index(drop=True)


def save_final_results_json(
    ranked_df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save the final ranked dataframe into data/output/final_results.json."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataframe rows into plain dictionaries so the JSON is easy to read.
    records = ranked_df.to_dict(orient="records")
    payload = {
        "row_count": len(records),
        "generated_by": "momentum_screening_pipeline",
        "results": records,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)

    logger.info("Saved final results to %s.", output_path)
    return output_path


def run_momentum_screening_pipeline(
    stock_df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    rsi_mode: str | None = None,
) -> pd.DataFrame:
    """Run all filters sequentially and return the final ranked dataframe."""

    try:
        stock_df = validate_processed_stock_dataframe(stock_df)
        logger.info("Starting pipeline with %s rows.", len(stock_df))

        filtered_df = _run_symbol_filter_step(
            "Trend Filter",
            lambda df: filter_structurally_bullish_stocks(
                _latest_rows(df, symbol_column="symbol", date_column="date")
            ),
            stock_df,
        )
        filtered_df = _run_symbol_filter_step(
            "RSI Filter",
            lambda df: filter_rsi_momentum_stocks(df, mode=rsi_mode),
            filtered_df,
        )
        filtered_df = _run_symbol_filter_step(
            "Volume Filter",
            filter_volume_expansion_stocks,
            filtered_df,
        )
        filtered_df = _run_symbol_filter_step(
            "HH-HL Filter",
            filter_higher_high_higher_low_stocks,
            filtered_df,
        )

        # Momentum persistence is a ranking step, so we rank only the latest row
        # for each symbol after all history-aware filters have selected symbols.
        latest_filtered_df = _latest_rows(
            filtered_df,
            symbol_column="symbol",
            date_column="date",
        )
        ranked_df = _run_filter_step(
            "Momentum Persistence and Ranking",
            rank_momentum_persistence,
            latest_filtered_df,
        )

        save_final_results_json(ranked_df, output_path)
        logger.info("Pipeline completed with %s final rows.", len(ranked_df))

        return ranked_df
    except Exception:
        logger.exception("Momentum screening pipeline failed.")
        raise


def _run_symbol_filter_step(
    step_name: str,
    step_function: Callable[[pd.DataFrame], pd.DataFrame],
    stock_df: pd.DataFrame,
    symbol_column: str = "symbol",
    date_column: str = "date",
) -> pd.DataFrame:
    """Run a filter, then keep full history for symbols that passed.

    Several filters need historical rows to make a decision, but return only the
    latest passing row. The next filter may still need history, so this helper
    converts each filter result into a symbol list and keeps all rows for those
    symbols.
    """

    before_symbols = stock_df[symbol_column].nunique()
    before_rows = len(stock_df)
    logger.info(
        "Running %s on %s symbols and %s rows.",
        step_name,
        before_symbols,
        before_rows,
    )

    try:
        passing_df = step_function(stock_df)
    except Exception as exc:
        raise RuntimeError(f"{step_name} failed: {exc}") from exc

    if symbol_column not in passing_df.columns:
        raise RuntimeError(f"{step_name} output did not include {symbol_column}.")

    passing_symbols = set(passing_df[symbol_column].dropna().unique())
    filtered_df = stock_df.loc[stock_df[symbol_column].isin(passing_symbols)].copy()
    filtered_df = _merge_filter_annotations(
        filtered_df,
        passing_df,
        symbol_column=symbol_column,
        date_column=date_column,
    )

    logger.info(
        "%s retained %s of %s symbols.",
        step_name,
        len(passing_symbols),
        before_symbols,
    )
    return filtered_df


def _merge_filter_annotations(
    full_history_df: pd.DataFrame,
    passing_df: pd.DataFrame,
    symbol_column: str,
    date_column: str,
) -> pd.DataFrame:
    """Carry useful computed filter columns onto the preserved history.

    The pipeline keeps full history for later filters, but some filters compute
    fields needed by the final output. This merge keeps those new fields on the
    latest passing row without mutating older history.
    """

    if date_column not in full_history_df.columns or date_column not in passing_df.columns:
        return full_history_df

    annotation_columns = [
        column
        for column in passing_df.columns
        if column not in full_history_df.columns
    ]
    if not annotation_columns:
        return full_history_df

    merge_columns = [symbol_column, date_column] + annotation_columns
    return full_history_df.merge(
        passing_df[merge_columns],
        on=[symbol_column, date_column],
        how="left",
    )


def _run_filter_step(
    step_name: str,
    step_function: Callable[[pd.DataFrame], pd.DataFrame],
    stock_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run one pipeline step with consistent logging and error context."""

    before_count = len(stock_df)
    logger.info("Running %s on %s rows.", step_name, before_count)

    try:
        filtered_df = step_function(stock_df)
    except Exception as exc:
        raise RuntimeError(f"{step_name} failed: {exc}") from exc

    logger.info(
        "%s retained %s of %s rows.",
        step_name,
        len(filtered_df),
        before_count,
    )
    return filtered_df


def _latest_rows(
    stock_df: pd.DataFrame,
    symbol_column: str = "symbol",
    date_column: str | None = "date",
) -> pd.DataFrame:
    """Return the latest row for each symbol."""

    if stock_df.empty:
        return stock_df.copy()

    if symbol_column not in stock_df.columns:
        raise ValueError(f"Missing required column: {symbol_column}")

    sort_columns = [symbol_column]
    if date_column is not None and date_column in stock_df.columns:
        sort_columns.append(date_column)

    sorted_df = stock_df.sort_values(sort_columns)
    return sorted_df.groupby(symbol_column, as_index=False).tail(1).copy()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running the pipeline manually."""

    parser = argparse.ArgumentParser(
        description="Run the master momentum screening pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Processed stock dataframe path. Supports .csv, .json, .parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Final JSON output path.",
    )
    parser.add_argument(
        "--rsi-mode",
        choices=("conservative", "moderate", "aggressive"),
        default=None,
        help="Optional RSI mode override. Defaults to config active_mode.",
    )
    return parser.parse_args()


def generate_processed_stock_dataframe(output_path: Path) -> Path:
    """
    Generate processed stock dataframe if it does not exist.

    This is a temporary bootstrap implementation until a dedicated
    processing pipeline module is added.
    """

    logger.info("Generating processed stock dataframe...")

    raw_path = PROJECT_ROOT / "data" / "raw" / "nifty500.json"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw NSE data not found: {raw_path}"
        )

    try:
        raw_df = pd.read_json(raw_path)

        # Normalize column names
        raw_df.columns = [column.strip().lower() for column in raw_df.columns]

        required_columns = [
            "symbol",
            "date",
            "close",
            "volume",
            "high",
            "low",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in raw_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required raw columns: "
                + ", ".join(missing_columns)
            )

        processed_df = raw_df.copy()

        # Placeholder EMA calculations
        processed_df["ema50"] = processed_df["close"]
        processed_df["ema200"] = processed_df["close"]

        # Placeholder RSI
        processed_df["rsi"] = 50

        # Placeholder returns
        processed_df["return_1m"] = 0
        processed_df["return_3m"] = 0
        processed_df["return_6m"] = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        processed_df.to_csv(output_path, index=False)

        logger.info(
            "Generated processed dataframe at %s",
            output_path,
        )

        return output_path

    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate processed dataframe: {exc}"
        ) from exc


def main() -> None:
    """CLI entry point."""

    configure_logging()

    args = parse_args()

    # Ensure required directories exist
    for directory in [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "output",
        PROJECT_ROOT / "logs",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Auto-generate processed data if missing
    if not args.input.exists():
        logger.warning(
            "Processed dataframe not found: %s",
            args.input,
        )

        generate_processed_stock_dataframe(args.input)

    processed_df = load_processed_stock_dataframe(args.input)

    run_momentum_screening_pipeline(
        processed_df,
        output_path=args.output,
        rsi_mode=args.rsi_mode,
    )


if __name__ == "__main__":
    main()
