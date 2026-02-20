"""
run.py — MLOps Pipeline: Deterministic signal generation from OHLCV CSV data.

CLI:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: Tuple[str, ...] = ("timestamp", "open", "high", "close")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_file: str) -> logging.Logger:
    """Configure dual-sink logging to a file and stdout.

    Args:
        log_file: Absolute or relative path to the output log file.
            The file is created (or truncated) on each run.

    Returns:
        A configured :class:`logging.Logger` instance named
        ``"mlops_pipeline"``.
    """
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s"
    )

    # File sink — write mode so each run starts with a clean log.
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console sink — mirrors log output to stdout for Docker visibility.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Error output
# ---------------------------------------------------------------------------


def write_error_json(
    output_path: str,
    error_message: str,
    version: str = "unknown",
) -> None:
    """Write a structured error payload to the output JSON file.

    Args:
        output_path: Path where the error JSON will be written.
        error_message: Human-readable description of the failure.
        version: Pipeline version string extracted from config.
            Defaults to ``"unknown"`` when config is unavailable.
    """
    result = {
        "version": version,
        "status": "error",
        "error_message": error_message,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load and validate the YAML configuration file.

    Expected structure::

        seed: 42
        window: 5
        version: "v1"

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing ``seed`` (int), ``window`` (int),
        and ``version`` (str).

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the file is not a valid YAML mapping, is missing
            required keys, or contains values of the wrong type.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got: {type(config).__name__}"
        )

    required_keys = ["seed", "window", "version"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError(
            f"'seed' must be an integer, got: {type(config['seed']).__name__}"
        )
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(
            f"'window' must be a positive integer, got: {config['window']}"
        )
    if not isinstance(config["version"], str):
        raise ValueError(
            f"'version' must be a string, got: {type(config['version']).__name__}"
        )

    return config


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------


def load_data(input_path: str) -> pd.DataFrame:
    """Load, validate, and chronologically sort the input OHLCV CSV.

    Performs the following checks in order:

    1. File existence and readability.
    2. Successful CSV parse.
    3. Non-empty row count.
    4. Presence of all required OHLCV columns (timestamp, open, high, close).
    5. Timestamp conversion to :class:`pandas.Timestamp`.
    6. Chronological sort by timestamp (required for valid rolling statistics).

    Args:
        input_path: Path to the input CSV file.

    Returns:
        A validated and chronologically sorted :class:`pandas.DataFrame`
        containing at minimum the ``REQUIRED_COLUMNS``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If the CSV is unreadable, empty, or missing required
            columns.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError(f"Input CSV is empty (0 rows): {input_path}")

    # Validate all required OHLCV columns are present.
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s): {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    # Parse timestamps and sort chronologically for valid rolling statistics.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def compute_rolling_mean(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute a rolling arithmetic mean over the ``close`` column.

    Args:
        df: A validated OHLCV DataFrame sorted by timestamp.
        window: The number of periods in the rolling window.

    Returns:
        A :class:`pandas.Series` of rolling means aligned with ``df.index``.
        The first ``window - 1`` values are ``NaN`` by definition.
    """
    return df["close"].rolling(window=window).mean()


def generate_signals(
    close: pd.Series,
    rolling_mean: pd.Series,
) -> pd.Series:
    """Generate a binary trading signal by comparing close price to its rolling mean.

    Signal rules:

    * ``1`` — ``close > rolling_mean`` (bullish crossover).
    * ``0`` — ``close <= rolling_mean`` **or** ``rolling_mean`` is ``NaN``
      (insufficient lookback data for the initial window rows).

    Args:
        close: Series of closing prices aligned with ``rolling_mean``.
        rolling_mean: Series of rolling mean values (may contain leading NaNs).

    Returns:
        An integer-typed :class:`pandas.Series` containing only ``0`` and
        ``1`` values.
    """
    signals = (close > rolling_mean).astype(int)
    # Force initial NaN window rows to 0 — no signal without sufficient data.
    signals[rolling_mean.isna()] = 0
    return signals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        An :class:`argparse.Namespace` with attributes:
        ``input``, ``config``, ``output``, ``log_file``.
    """
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline — Deterministic signal generation from OHLCV data."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output", required=True, help="Path for output metrics JSON"
    )
    parser.add_argument(
        "--log-file", required=True, help="Path for execution log file"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Execute the full MLOps pipeline and return an exit code.

    Pipeline stages:

    1. Configuration loading and seed initialisation.
    2. OHLCV data ingestion, validation, and chronological sorting.
    3. Rolling mean computation.
    4. Binary signal generation.
    5. Metrics serialisation to JSON and stdout.

    Returns:
        ``0`` on successful completion, ``1`` on any handled exception.
    """
    args = parse_args()
    start_time = time.time()

    logger = setup_logging(args.log_file)
    logger.info("Job started")

    # Sentinel version for error payloads emitted before config is loaded.
    version = "unknown"

    try:
        # ------------------------------------------------------------------
        # Stage 1: Configuration
        # ------------------------------------------------------------------
        config = load_config(args.config)
        seed: int = config["seed"]
        window: int = config["window"]
        version: str = config["version"]

        np.random.seed(seed)
        logger.info(
            f"Configuration loaded: seed={seed}, window={window}, version={version}"
        )

        # ------------------------------------------------------------------
        # Stage 2: Data Ingestion
        # ------------------------------------------------------------------
        df = load_data(args.input)
        row_count: int = len(df)
        logger.info(f"Data loaded: {row_count} rows from {args.input}")
        logger.info("Data chronologically sorted by timestamp")

        # ------------------------------------------------------------------
        # Stage 3: Rolling Mean
        # ------------------------------------------------------------------
        rolling_mean = compute_rolling_mean(df, window)
        logger.info(f"Rolling mean computed with window={window}")

        # ------------------------------------------------------------------
        # Stage 4: Signal Generation
        # ------------------------------------------------------------------
        signals = generate_signals(df["close"], rolling_mean)
        signal_rate = round(float(signals.sum()) / row_count, 4)
        logger.info(
            f"Metrics: signal_rate={signal_rate:.4f}, rows_processed={row_count}"
        )

        # ------------------------------------------------------------------
        # Stage 5: Metrics Output
        # ------------------------------------------------------------------
        latency_ms: int = round((time.time() - start_time) * 1000)
        metrics = {
            "version": version,
            "rows_processed": row_count,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }

        print(json.dumps(metrics, indent=2))

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics written to {args.output}")
        logger.info(f"Job completed successfully in {latency_ms} ms")

        return 0

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"ERROR: {error_msg}")
        write_error_json(args.output, error_msg, version=version)
        logger.info(f"Error metrics written to {args.output}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
