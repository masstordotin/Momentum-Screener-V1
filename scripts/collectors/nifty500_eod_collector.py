"""Collect raw NIFTY 500 EOD data from official NSE APIs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from .nse_client import NSEClient, NSEClientError
except ImportError:
    # Allows direct execution: python scripts/collectors/nifty500_eod_collector.py
    from nse_client import NSEClient, NSEClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def default_from_date(days: int = 30) -> str:
    """Return dd-mm-yyyy for a date a few days before today."""

    return (date.today() - timedelta(days=days)).strftime("%d-%m-%Y")


def default_to_date() -> str:
    """Return today's date in NSE's dd-mm-yyyy format."""

    return date.today().strftime("%d-%m-%Y")


def save_raw_json(payload: dict[str, Any], path: Path) -> None:
    """Save raw API payload exactly as JSON so later steps can process it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_raw_text(payload: str, path: Path) -> None:
    """Save raw text payloads such as NSE's official CSV files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def extract_symbols_from_csv(csv_payload: str) -> list[str]:
    """Read equity symbols from NSE's official NIFTY 500 CSV."""

    symbols: list[str] = []
    reader = csv.DictReader(csv_payload.splitlines())

    for row in reader:
        symbol = row.get("Symbol")
        if isinstance(symbol, str) and symbol.strip():
            symbols.append(symbol.strip().upper())

    if not symbols:
        raise NSEClientError("No symbols were found in the NIFTY 500 payload.")

    return sorted(set(symbols))


def parse_nse_date(date_text: str) -> date:
    """Convert NSE's dd-mm-yyyy CLI date text into a date object."""

    return datetime.strptime(date_text, "%d-%m-%Y").date()


def iter_dates(from_date: str, to_date: str) -> list[date]:
    """Return all calendar dates between from_date and to_date, inclusive."""

    start = parse_nse_date(from_date)
    end = parse_nse_date(to_date)

    if start > end:
        raise ValueError("--from-date must be on or before --to-date")

    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def normalize_csv_row(row: dict[str, str]) -> dict[str, str]:
    """Strip NSE's extra spaces from CSV headers and values."""

    return {
        key.strip(): value.strip()
        for key, value in row.items()
        if key is not None and value is not None
    }


def filter_bhavcopy_rows(
    bhavcopy_csv: str,
    symbols: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Keep only EQ rows for the NIFTY 500 symbols."""

    reader = csv.DictReader(bhavcopy_csv.splitlines())
    fieldnames = [field.strip() for field in reader.fieldnames or []]
    filtered_rows: list[dict[str, str]] = []

    for raw_row in reader:
        row = normalize_csv_row(raw_row)
        if row.get("SYMBOL") in symbols and row.get("SERIES") == "EQ":
            filtered_rows.append(row)

    return fieldnames, filtered_rows


def rows_to_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    """Convert filtered bhavcopy rows back to CSV text."""

    if not fieldnames:
        return ""

    lines: list[str] = []
    writer = csv.DictWriter(
        CsvListWriter(lines),
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return "".join(lines)


class CsvListWriter:
    """Tiny file-like adapter so csv can write into a list of strings."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def write(self, text: str) -> None:
        self.lines.append(text)


def collect_nifty500_from_bhavcopy(
    client: NSEClient,
    symbols: list[str],
    from_date: str,
    to_date: str,
    run_dir: Path,
    pause_seconds: float,
    fail_if_no_data: bool = True,
    fallback_lookback_days: int = 10,
) -> None:
    """Collect daily NSE bhavcopy files and filter them to NIFTY 500."""

    symbol_set = set(symbols)
    failures: dict[str, str] = {}
    saved_dates = 0

    print("Using NSE daily bhavcopy fallback for EOD data...")

    requested_dates = iter_dates(from_date, to_date)
    trade_dates = _extend_with_fallback_dates(
        requested_dates,
        fallback_lookback_days=fallback_lookback_days,
    )

    for trade_date in trade_dates:
        date_text = trade_date.strftime("%d%m%Y")

        try:
            bhavcopy_csv = client.get_security_bhavcopy_csv(date_text)
            save_raw_text(
                bhavcopy_csv,
                run_dir / "bhavcopy" / f"sec_bhavdata_full_{date_text}.csv",
            )

            fieldnames, rows = filter_bhavcopy_rows(bhavcopy_csv, symbol_set)
            filtered_csv = rows_to_csv(fieldnames, rows)
            save_raw_text(
                filtered_csv,
                run_dir / "nifty500_bhavcopy" / f"nifty500_eod_{date_text}.csv",
            )

            saved_dates += 1
            print(f"Saved bhavcopy {date_text}: {len(rows)} NIFTY 500 rows")
        except Exception as exc:
            # Holidays and missing archive dates can return errors; keep going.
            failures[date_text] = str(exc)
            print(f"Skipped bhavcopy {date_text}: {exc}")

        time.sleep(pause_seconds)

    save_raw_json(
        {
            "source": "NSE official daily bhavcopy archive",
            "from_date": from_date,
            "to_date": to_date,
            "fallback_lookback_days": fallback_lookback_days,
            "saved_dates": saved_dates,
            "failed_dates": failures,
        },
        run_dir / "bhavcopy_manifest.json",
    )

    if saved_dates == 0 and fail_if_no_data:
        raise NSEClientError(
            "No bhavcopy files were downloaded. This can happen on NSE holidays "
            "or when archives are not published yet."
        )


def _extend_with_fallback_dates(
    requested_dates: list[date],
    fallback_lookback_days: int,
) -> list[date]:
    """Add recent prior dates so CI survives delayed same-day bhavcopy archives."""

    if not requested_dates or fallback_lookback_days <= 0:
        return requested_dates

    end_date = max(requested_dates)
    fallback_start = end_date - timedelta(days=fallback_lookback_days)
    fallback_dates = [
        fallback_start + timedelta(days=offset)
        for offset in range((end_date - fallback_start).days + 1)
    ]

    # Keep dates sorted and unique. Weekends/holidays will be skipped naturally
    # when NSE returns 404 for those archive files.
    return sorted(set(requested_dates + fallback_dates))


def collect_nifty500_eod(
    from_date: str,
    to_date: str,
    output_dir: Path = RAW_DATA_DIR,
    pause_seconds: float = 0.4,
    max_consecutive_failures: int = 5,
    eod_source: str = "auto",
    fallback_lookback_days: int = 10,
) -> None:
    """Collect index snapshot and EOD history for all NIFTY 500 symbols."""

    client = NSEClient()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "nifty500_eod" / run_id

    print("Fetching official NIFTY 500 stock list from NSE...")
    stock_list_csv = client.get_nifty500_stock_list_csv()
    save_raw_text(stock_list_csv, run_dir / "nifty500_stock_list.csv")

    symbols = extract_symbols_from_csv(stock_list_csv)
    save_raw_json(
        {
            "source": "NSE official API",
            "index": "NIFTY 500",
            "from_date": from_date,
            "to_date": to_date,
            "symbol_count": len(symbols),
            "symbols": symbols,
        },
        run_dir / "collection_manifest.json",
    )

    print(f"Found {len(symbols)} symbols. Fetching EOD history...")

    if eod_source == "bhavcopy":
        collect_nifty500_from_bhavcopy(
            client,
            symbols,
            from_date,
            to_date,
            run_dir,
            pause_seconds,
            fallback_lookback_days=fallback_lookback_days,
        )
        print(f"Raw NSE data saved to: {run_dir}")
        return

    failures: dict[str, str] = {}
    consecutive_failures = 0

    first_symbol = symbols[0]
    try:
        print(f"Checking EOD endpoint with {first_symbol}...")
        payload = client.get_equity_eod_history(first_symbol, from_date, to_date)
        save_raw_json(payload, run_dir / "symbols" / f"{first_symbol}.json")
        print(f"[1/{len(symbols)}] Saved {first_symbol}")
    except Exception as exc:
        failures[first_symbol] = str(exc)
        save_raw_json(
            {
                "failed_count": len(failures),
                "failures": failures,
                "stopped_early": True,
                "reason": (
                    "The first EOD request failed. NSE is likely blocking or "
                    "throttling the historical-data endpoint from this network."
                ),
            },
            run_dir / "failures.json",
        )
        print(f"[1/{len(symbols)}] Failed {first_symbol}: {exc}")
        print(
            "Stopping before the full symbol loop because the EOD endpoint "
            "failed on the preflight request."
        )
        if eod_source == "auto":
            print("Falling back to NSE daily bhavcopy archives.")
            collect_nifty500_from_bhavcopy(
                client,
                symbols,
                from_date,
                to_date,
                run_dir,
                pause_seconds,
                fallback_lookback_days=fallback_lookback_days,
            )
        print(f"Raw NSE data saved to: {run_dir}")
        return

    for position, symbol in enumerate(symbols, start=1):
        if symbol == first_symbol:
            continue

        try:
            payload = client.get_equity_eod_history(symbol, from_date, to_date)
            save_raw_json(payload, run_dir / "symbols" / f"{symbol}.json")
            print(f"[{position}/{len(symbols)}] Saved {symbol}")
            consecutive_failures = 0
        except Exception as exc:
            # Keep going so one bad symbol does not stop the whole run.
            failures[symbol] = str(exc)
            consecutive_failures += 1
            print(f"[{position}/{len(symbols)}] Failed {symbol}: {exc}")

            if consecutive_failures >= max_consecutive_failures:
                print(
                    "Stopping early because too many symbols failed in a row. "
                    "NSE may be blocking or throttling the EOD endpoint."
                )
                break

        # A small pause keeps the collector polite and helps avoid rate limits.
        time.sleep(pause_seconds)

    if failures:
        save_raw_json(
            {
                "failed_count": len(failures),
                "failures": failures,
            },
            run_dir / "failures.json",
        )

    print(f"Raw NSE data saved to: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw NIFTY 500 EOD data from NSE official APIs."
    )
    parser.add_argument(
        "--from-date",
        default=default_from_date(),
        help="Start date in dd-mm-yyyy format. Default: 30 days ago.",
    )
    parser.add_argument(
        "--to-date",
        default=default_to_date(),
        help="End date in dd-mm-yyyy format. Default: today.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.4,
        help="Pause between symbol requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="Stop after this many consecutive symbol failures.",
    )
    parser.add_argument(
        "--eod-source",
        choices=("auto", "api", "bhavcopy"),
        default="auto",
        help="EOD source: api, bhavcopy, or auto fallback.",
    )
    parser.add_argument(
        "--fallback-lookback-days",
        type=int,
        default=10,
        help=(
            "Extra recent calendar days to try when NSE has not published "
            "today's bhavcopy yet."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_nifty500_eod(
        from_date=args.from_date,
        to_date=args.to_date,
        pause_seconds=args.pause_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        eod_source=args.eod_source,
        fallback_lookback_days=args.fallback_lookback_days,
    )
