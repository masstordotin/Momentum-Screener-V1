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
import math
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
DEFAULT_PICKS_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "momentum-picks.json"
DEFAULT_ROOT_PICKS_OUTPUT_PATH = PROJECT_ROOT / "momentum-picks.json"
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
    diagnostics: list[dict[str, int | str]] | None = None,
) -> Path:
    """Save the final ranked dataframe into data/output/final_results.json."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataframe rows into plain dictionaries so the JSON is easy to read
    # and safe for browsers to parse.
    records = _json_safe_records(ranked_df)
    payload = {
        "row_count": len(records),
        "generated_by": "momentum_screening_pipeline",
        "diagnostics": diagnostics or [],
        "results": records,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)

    logger.info("Saved final results to %s.", output_path)
    return output_path


def save_dashboard_momentum_picks_json(
    ranked_df: pd.DataFrame,
    output_path: Path = DEFAULT_PICKS_OUTPUT_PATH,
    mirror_output_path: Path | None = DEFAULT_ROOT_PICKS_OUTPUT_PATH,
) -> Path:
    """Save a flat dashboard-friendly JSON file.

    Some static dashboards read a simple list from momentum-picks.json instead
    of the richer final_results.json payload. Keeping both outputs avoids a
    front-end/back-end schema mismatch while preserving the main audit trail.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for row in _json_safe_records(ranked_df):
        symbol = str(row.get("symbol") or "-").upper()
        close = _as_float(row.get("close"))
        ema50 = _as_float(row.get("ema50"))

        records.append(
            {
                "rank": row.get("rank"),
                "name": symbol,
                "symbol": symbol,
                "sector": row.get("sector") or "Unclassified",
                "bucket": "Momentum Leader",
                "score": _round_or_none(row.get("momentum_score")),
                "price": _round_or_none(close),
                "rsi": _round_or_none(row.get("rsi")),
                "ema50": _round_or_none(ema50),
                "ema200": _round_or_none(row.get("ema200")),
                "volRatio": _round_or_none(row.get("volume_expansion")),
                "ret1m": _round_or_none(row.get("return_1m")),
                "ret3m": _round_or_none(row.get("return_3m")),
                "ret6m": _round_or_none(row.get("return_6m")),
                "setup": "Close > EMA50 > EMA200 with positive momentum persistence",
                "buyZone": _format_buy_zone(close),
                "invalid": _format_invalid_level(ema50),
                "thesis": (
                    f"{symbol} passed trend, RSI, volume expansion, HH-HL "
                    "structure, and positive momentum ranking filters."
                ),
                "indicator": (
                    f"RSI {_format_metric(row.get('rsi'))}, "
                    f"volume expansion {_format_metric(row.get('volume_expansion'))}x, "
                    f"momentum percentile {_format_metric(row.get('momentum_percentile'))}"
                ),
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, allow_nan=False)

    logger.info("Saved dashboard momentum picks to %s.", output_path)

    if mirror_output_path is not None:
        # A root-level mirror is useful for GitHub Pages dashboards that fetch
        # ./momentum-picks.json from the same folder as index.html.
        with mirror_output_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, allow_nan=False)
        logger.info("Saved root dashboard momentum picks to %s.", mirror_output_path)

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
        diagnostics: list[dict[str, int | str]] = [
            _diagnostic_step("Input", stock_df),
        ]

        filtered_df = _run_symbol_filter_step(
            "Trend Filter",
            lambda df: filter_structurally_bullish_stocks(
                _latest_rows(df, symbol_column="symbol", date_column="date")
            ),
            stock_df,
        )
        diagnostics.append(_diagnostic_step("Trend Filter", filtered_df))
        filtered_df = _run_symbol_filter_step(
            "RSI Filter",
            lambda df: filter_rsi_momentum_stocks(df, mode=rsi_mode),
            filtered_df,
        )
        diagnostics.append(_diagnostic_step("RSI Filter", filtered_df))
        filtered_df = _run_symbol_filter_step(
            "Volume Filter",
            filter_volume_expansion_stocks,
            filtered_df,
        )
        diagnostics.append(_diagnostic_step("Volume Filter", filtered_df))
        filtered_df = _run_symbol_filter_step(
            "HH-HL Filter",
            filter_higher_high_higher_low_stocks,
            filtered_df,
        )
        diagnostics.append(_diagnostic_step("HH-HL Filter", filtered_df))

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
        diagnostics.append(_diagnostic_step("Momentum Ranking", ranked_df))

        save_final_results_json(ranked_df, output_path, diagnostics=diagnostics)
        save_dashboard_momentum_picks_json(ranked_df)
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


def _diagnostic_step(
    step_name: str,
    stock_df: pd.DataFrame,
    symbol_column: str = "symbol",
) -> dict[str, int | str]:
    """Return small row/symbol counts for final_results.json diagnostics."""

    symbol_count = (
        int(stock_df[symbol_column].nunique())
        if symbol_column in stock_df.columns
        else 0
    )
    return {
        "step": step_name,
        "rows": int(len(stock_df)),
        "symbols": symbol_count,
    }


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


def _json_safe_records(stock_df: pd.DataFrame) -> list[dict[str, object]]:
    """Return records containing only JSON-safe Python values.

    Pandas can hold Timestamp, numpy numbers, NaN, and infinite values. Browsers
    and GitHub Pages need strict JSON, so this function normalizes those values
    before writing files.
    """

    records: list[dict[str, object]] = []
    for row in stock_df.to_dict(orient="records"):
        records.append(
            {key: _json_safe_value(value) for key, value in row.items()}
        )
    return records


def _json_safe_value(value: object) -> object:
    """Convert one pandas/numpy value into a strict JSON-compatible value."""

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _as_float(value: object) -> float | None:
    """Safely coerce dashboard numeric values."""

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def _round_or_none(value: object, digits: int = 2) -> float | None:
    """Round a numeric value for compact dashboard display."""

    numeric_value = _as_float(value)
    return None if numeric_value is None else round(numeric_value, digits)


def _format_metric(value: object) -> str:
    """Format a metric for the dashboard thesis text."""

    numeric_value = _as_float(value)
    return "N/A" if numeric_value is None else f"{numeric_value:.2f}"


def _format_buy_zone(close: float | None) -> str:
    """Create a simple reference zone around the latest close."""

    if close is None:
        return "N/A"
    return f"{close * 0.99:.2f}-{close * 1.01:.2f}"


def _format_invalid_level(ema50: float | None) -> str:
    """Use EMA50 as the practical trend invalidation reference."""

    if ema50 is None:
        return "N/A"
    return f"Daily close below EMA50 ({ema50:.2f})"


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

    # Ensure required directories exist
    for directory in [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "output",
        PROJECT_ROOT / "logs",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(
            "Processed dataframe not found. Run "
            "scripts/indicators/indicator_engine.py before scripts/main.py: "
            f"{args.input}"
        )

    processed_df = load_processed_stock_dataframe(args.input)

    run_momentum_screening_pipeline(
        processed_df,
        output_path=args.output,
        rsi_mode=args.rsi_mode,
    )


if __name__ == "__main__":
    main()
