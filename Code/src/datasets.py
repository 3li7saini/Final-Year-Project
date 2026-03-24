import pandas as pd
from collections.abc import Sequence, Iterable
from pathlib import Path

from src.config import SKILL_COLS, FORM_COLS
from src.data_loader import load_parquet
from src.utils import ensure_dir, assert_columns

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


def verify_saved_xg_split(
    path: str | Path,
    *,
    expected_rows: int,
    required_cols: Sequence[str] = (),
    absent_cols: Sequence[str] = (),
    expected_leagues: Iterable[str] | None = None,
    mandatory_cols: tuple[str, ...] = ("is_goal", "xg_pred"),
    forbidden_cols: tuple[str, ...] = ("dataset_split",),
    name: str = "",
) -> pd.DataFrame:
    """Reload a saved parquet and verify the persistence contract for
    post-baseline xG datasets (NB05+).

    Default mandatory_cols=("is_goal", "xg_pred") reflects the contract
    that every dataset from NB04 onward carries the target and the
    baseline prediction. mandatory_cols checks presence only, not
    non-null — matching the existing notebook behavior where is_goal
    and xg_pred are never null-checked explicitly. Callers can override
    if reusing for other dataset families.

    expected_leagues accepts any iterable of league name strings (list,
    set, frozenset) and normalizes to a set internally. Do NOT pass a
    bare string — set("England") would produce {"E","n","g",...}.

    Checks (in order):
      0. required_cols and absent_cols do not overlap (API misuse guard)
      1. Row count == expected_rows
      2. forbidden_cols not present (default: dataset_split)
      3. mandatory_cols present, presence only (default: is_goal, xg_pred)
      4. ID_COLS present (guard against KeyError before duplicate check)
      5. No duplicate rows on ID_COLS  [contract expansion — see note]
      6. All required_cols present and non-null
      7. All absent_cols NOT present
      8. expected_leagues is not a bare string (runtime guard)
      9. "league" column present if expected_leagues given
     10. League membership matches expected_leagues (if provided)

    Returns the reloaded DataFrame so callers can run additional
    notebook-specific assertions on it.
    """
    df = load_parquet(path)
    label = name or Path(path).stem

    # 0. Sanity: required and absent must not overlap
    overlap = set(required_cols) & set(absent_cols)
    assert not overlap, f"{label}: cols in both required and absent: {overlap}"

    # 1. Row count
    assert len(df) == expected_rows, \
        f"{label} row count mismatch: {len(df)} != {expected_rows}"

    # 2. Forbidden columns
    for col in forbidden_cols:
        assert col not in df.columns, f"{col} found in {label}"

    # 3. Mandatory columns (presence only, no null check)
    assert_columns(df, list(mandatory_cols), label)

    # 4-5. ID columns present + no duplicate IDs
    assert_columns(df, ID_COLS, label)
    assert not df[ID_COLS].duplicated().any(), f"Duplicate IDs in {label}"

    # 6. Required columns present and non-null
    assert_columns(df, list(required_cols), label)
    for col in required_cols:
        assert df[col].notna().all(), f"Nulls in {col} in {label}"

    # 7. Absent columns
    for col in absent_cols:
        assert col not in df.columns, f"{col} should not be in {label}"

    # 8-10. League membership
    if expected_leagues is not None:
        assert not isinstance(expected_leagues, (str, bytes)), \
            f"{label}: expected_leagues must be an iterable of strings, not a bare string/bytes"
        assert "league" in df.columns, f"league column missing from {label}"
        expected_set = set(expected_leagues)
        actual_set = set(df["league"].unique())
        assert actual_set == expected_set, \
            f"Wrong leagues in {label}: {actual_set} != {expected_set}"

    return df


def verify_saved_xg_bundle(
    name: str,
    data_dir: str | Path,
    *,
    expected_train_rows: int,
    expected_test_rows: int,
    required_cols: Sequence[str] = (),
    absent_cols: Sequence[str] = (),
    train_leagues: Iterable[str] | None = None,
    test_leagues: Iterable[str] | None = None,
    mandatory_cols: tuple[str, ...] = ("is_goal", "xg_pred"),
    forbidden_cols: tuple[str, ...] = ("dataset_split",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload and verify the two parquet splits of a saved dataset bundle.

    Convenience wrapper around verify_saved_xg_split for the common
    train/test pair pattern (matching save_dataset_bundle file naming).
    Verifies only the train/test parquets, not the sample CSV that
    save_dataset_bundle also writes.
    Forwards mandatory_cols and forbidden_cols to the split helper.

    Returns (train_df, test_df) for further notebook-specific checks.
    """
    data_dir = Path(data_dir)
    train_df = verify_saved_xg_split(
        data_dir / f"wyscout_train_xg_{name}.parquet",
        expected_rows=expected_train_rows,
        required_cols=required_cols,
        absent_cols=absent_cols,
        expected_leagues=train_leagues,
        mandatory_cols=mandatory_cols,
        forbidden_cols=forbidden_cols,
        name=f"{name}_train",
    )
    test_df = verify_saved_xg_split(
        data_dir / f"wyscout_test_xg_{name}.parquet",
        expected_rows=expected_test_rows,
        required_cols=required_cols,
        absent_cols=absent_cols,
        expected_leagues=test_leagues,
        mandatory_cols=mandatory_cols,
        forbidden_cols=forbidden_cols,
        name=f"{name}_test",
    )
    return train_df, test_df
