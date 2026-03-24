import pandas as pd
from collections.abc import Sequence, Iterable
from pathlib import Path

from src.data_loader import load_parquet
from src.utils import assert_columns

ID_COLS = ["league", "matchId", "playerId", "teamId", "eventSec", "matchPeriod"]


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
