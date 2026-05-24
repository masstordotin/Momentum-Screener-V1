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

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable console logging for the screening pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_processed_stock_dataframe(input_path: Path) -> pd.DataFrame:
    """Load the processed stock dataframe from CSV, JSON, or Parquet."""

    if not input_path.exists():
        raise FileNotFoundError(f"Processed input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path)

    if suffix == ".json":
        return pd.read_json(input_path)

    if suffix == ".parquet":
        return pd.read_parquet(input_path)

    raise ValueError(
        "Unsupported processed input format. Use .csv, .json, or .parquet."
    )


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

    logger.info(
        "%s retained %s of %s symbols.",
        step_name,
        len(passing_symbols),
        before_symbols,
    )
    return filtered_df


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


def main() -> None:
    """CLI entry point."""

    configure_logging()
    args = parse_args()

    processed_df = load_processed_stock_dataframe(args.input)
    run_momentum_screening_pipeline(
        processed_df,
        output_path=args.output,
        rsi_mode=args.rsi_mode,
    )


if __name__ == "__main__":
    main()
