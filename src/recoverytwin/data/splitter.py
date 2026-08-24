"""
Temporal data splitter.

Splits data by time periods, never randomly.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import yaml


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_end: str = "2024-08-31",
    val_end: str = "2024-10-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe temporally into train / validation / test.

    Args:
        df: Input dataframe with timestamp column
        timestamp_col: Name of the timestamp column
        train_end: End date for training (inclusive)
        val_end: End date for validation (inclusive)

    Returns:
        (train, validation, test) tuple
    """
    ts = pd.to_datetime(df[timestamp_col])

    train = df[ts <= pd.Timestamp(train_end)].copy()
    val = df[(ts > pd.Timestamp(train_end)) & (ts <= pd.Timestamp(val_end))].copy()
    test = df[ts > pd.Timestamp(val_end)].copy()

    return train, val, test


def split_and_save(
    df: pd.DataFrame,
    output_dir: str = "data/processed/debug",
    config_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Split data and save to parquet files.

    Returns:
        (train, val, test, split_info)
    """
    train_end = "2024-08-31"
    val_end = "2024-10-31"

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ts_cfg = cfg.get("temporal_split", {})
        train_end = ts_cfg.get("train_end", train_end)
        val_end = ts_cfg.get("validation_end", val_end)

    train, val, test = temporal_split(df, train_end=train_end, val_end=val_end)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_path / "train.parquet", index=False)
    val.to_parquet(output_path / "validation.parquet", index=False)
    test.to_parquet(output_path / "test.parquet", index=False)

    split_info = {
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "total_rows": len(train) + len(val) + len(test),
        "train_end": train_end,
        "val_end": val_end,
        "train_recovery_rate": float(train["recovered"].mean()) if len(train) > 0 else 0,
        "val_recovery_rate": float(val["recovered"].mean()) if len(val) > 0 else 0,
        "test_recovery_rate": float(test["recovered"].mean()) if len(test) > 0 else 0,
    }

    return train, val, test, split_info
