import pandas as pd
from pathlib import Path

from src.config import SKILL_COLS, FORM_COLS
from src.utils import ensure_dir

ID_COLS = ["league", "matchId", "playerId", "teamId", "eventSec", "matchPeriod"]


def build_skill_dataset(base_df: pd.DataFrame, skill_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join SKILL_COLS from skill_df onto base_df via ID_COLS.

    Validates: row count unchanged, no nulls in SKILL_COLS after merge,
    row order (ID_COLS) preserved.
    Returns base_df with SKILL_COLS appended.
    """
    result = base_df.merge(
        skill_df[ID_COLS + SKILL_COLS],
        on=ID_COLS, how="left", validate="one_to_one",
    )
    assert len(result) == len(base_df), "Row count changed after skill merge"
    for col in SKILL_COLS:
        assert result[col].notna().all(), f"Nulls in {col} after skill merge"
    assert result[ID_COLS].equals(base_df[ID_COLS]), "Row order changed after skill merge"
    return result


def build_form_dataset(base_df: pd.DataFrame, form_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join FORM_COLS (only) from form_df onto base_df via ID_COLS.

    Intentionally excludes SKILL_COLS even though form_df contains them, creating a
    pure form-only comparison dataset (36 baseline + 8 form = 44 cols).
    Validates: row count unchanged, no nulls in FORM_COLS, row order preserved.
    Returns base_df with FORM_COLS appended.
    """
    result = base_df.merge(
        form_df[ID_COLS + FORM_COLS],
        on=ID_COLS, how="left", validate="one_to_one",
    )
    assert len(result) == len(base_df), "Row count changed after form merge"
    for col in FORM_COLS:
        assert result[col].notna().all(), f"Nulls in {col} after form merge"
    assert result[ID_COLS].equals(base_df[ID_COLS]), "Row order changed after form merge"
    return result


def build_combined_dataset(
    skill_aware_df: pd.DataFrame, form_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join FORM_COLS from form_df onto skill_aware_df (which already has SKILL_COLS).

    Merges onto the already-built skill-aware dataset rather than onto the raw form
    parquet (which already contains SKILL_COLS and would produce duplicate columns).
    Validates: row count unchanged, no nulls in FORM_COLS after merge,
    row order (ID_COLS) preserved.
    Returns skill_aware_df with FORM_COLS appended.
    """
    result = skill_aware_df.merge(
        form_df[ID_COLS + FORM_COLS],
        on=ID_COLS, how="left", validate="one_to_one",
    )
    assert len(result) == len(skill_aware_df), "Row count changed after combined merge"
    for col in FORM_COLS:
        assert result[col].notna().all(), f"Nulls in {col} after combined merge"
    assert result[ID_COLS].equals(skill_aware_df[ID_COLS]), \
        "Row order changed after combined merge"
    return result


def save_dataset_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
    data_dir,
) -> None:
    """Save train/test parquets and a train sample CSV for a named dataset.

    Drops dataset_split column if present (consistent with NB05/NB06 convention).
    Creates data_dir if it does not already exist.

    Writes to data_dir:
      wyscout_train_xg_{name}.parquet  — train split (dataset_split excluded)
      wyscout_test_xg_{name}.parquet   — test split  (dataset_split excluded)
      wyscout_xg_{name}_sample.csv     — first 100 rows of train
    """
    data_dir = Path(data_dir)
    ensure_dir(data_dir)

    def _drop_split(df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if c != "dataset_split"]]

    _drop_split(train_df).to_parquet(
        data_dir / f"wyscout_train_xg_{name}.parquet", index=False
    )
    _drop_split(test_df).to_parquet(
        data_dir / f"wyscout_test_xg_{name}.parquet", index=False
    )
    _drop_split(train_df).head(100).to_csv(
        data_dir / f"wyscout_xg_{name}_sample.csv", index=False
    )
