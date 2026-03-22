import numpy as np
import pandas as pd
from src.config import TAG_GOAL
from src.utils import ensure_dir


KEEP_COLS = [
    "league", "matchId", "playerId", "teamId",
    "eventSec", "matchPeriod", "eventName", "subEventName",
    "x", "y", "is_goal", "is_penalty", "is_direct_free_kick",
]


def extract_x(positions: list):
    """Return the raw x value from the first position entry, or NaN.

    Returns the value as-is (not cast to float) so the notebook's
    pd.to_numeric(..., errors="coerce") chain is the single point of dtype
    conversion. Returns NaN on missing list, missing key, None, or any error.
    """
    try:
        val = positions[0]["x"]
        return val if val is not None else np.nan
    except (IndexError, KeyError, TypeError):
        return np.nan


def extract_y(positions: list):
    """Return the raw y value from the first position entry, or NaN.

    Returns the value as-is (not cast to float) so the notebook's
    pd.to_numeric(..., errors="coerce") chain is the single point of dtype
    conversion. Returns NaN on missing list, missing key, None, or any error.
    """
    try:
        val = positions[0]["y"]
        return val if val is not None else np.nan
    except (IndexError, KeyError, TypeError):
        return np.nan


def extract_goal(tags) -> int:
    """Return 1 if the goal tag (TAG_GOAL) is present in tags, else 0.

    Handles None gracefully.
    """
    tags = tags or []
    return int(any(t.get("id") == TAG_GOAL for t in tags))


def is_shot_attempt(event: dict) -> bool:
    """Return True if the event is an accepted shot attempt.

    Accepted (eventName, subEventName) combinations:
      - ("Shot", "Shot")                 — open-play shot
      - ("Free Kick", "Penalty")         — in-play penalty, matchPeriod != "P"
      - ("Free Kick", "Free kick shot")  — direct free-kick shot

    Penalty shootout kicks (matchPeriod == "P") are explicitly excluded.
    """
    name    = event.get("eventName", "")
    subname = event.get("subEventName", "")
    period  = event.get("matchPeriod", "")

    if name == "Shot" and subname == "Shot":
        return True
    if name == "Free Kick" and subname == "Penalty" and period != "P":
        return True
    if name == "Free Kick" and subname == "Free kick shot":
        return True
    return False


def label_attempt_type_flags(event: dict) -> dict:
    """Return integer 0/1 flags for is_penalty and is_direct_free_kick.

    These flags are mutually exclusive by construction:
      is_penalty = 1          only for Free Kick/Penalty with matchPeriod != "P"
      is_direct_free_kick = 1 only for Free Kick/Free kick shot

    Returns integer values (not booleans).
    """
    name    = event.get("eventName", "")
    subname = event.get("subEventName", "")
    period  = event.get("matchPeriod", "")
    return {
        "is_penalty":          int(name == "Free Kick" and subname == "Penalty" and period != "P"),
        "is_direct_free_kick": int(name == "Free Kick" and subname == "Free kick shot"),
    }


def extract_shot_attempts(events: list, league: str) -> pd.DataFrame:
    """Filter a list of raw events to shot attempts for one league.

    Pipeline:
      1. Filter with is_shot_attempt()
      2. Extract x, y from positions list (raw value or NaN; caller casts dtype via pd.to_numeric)
      3. Extract is_goal from tags list using TAG_GOAL constant
      4. Label is_penalty and is_direct_free_kick flags (integer 0/1)
      5. Build DataFrame, select only KEEP_COLS, return copy

    Returns an empty DataFrame with KEEP_COLS columns if no shot attempts are found.
    NOTE: positions and tags are not carried forward into the output.
    """
    rows = []
    for e in events:
        if not is_shot_attempt(e):
            continue
        flags = label_attempt_type_flags(e)
        rows.append({
            "league":              league,
            "matchId":             e.get("matchId"),
            "playerId":            e.get("playerId"),
            "teamId":              e.get("teamId"),
            "eventSec":            e.get("eventSec"),
            "matchPeriod":         e.get("matchPeriod"),
            "eventName":           e.get("eventName"),
            "subEventName":        e.get("subEventName"),
            "x":                   extract_x(e.get("positions", [])),
            "y":                   extract_y(e.get("positions", [])),
            "is_goal":             extract_goal(e.get("tags")),
            "is_penalty":          flags["is_penalty"],
            "is_direct_free_kick": flags["is_direct_free_kick"],
        })
    if not rows:
        return pd.DataFrame(columns=KEEP_COLS)
    df = pd.DataFrame(rows)
    return df[KEEP_COLS].copy()


def save_shot_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir,
) -> pd.DataFrame:
    """Write the combined and split shot parquets plus a sample CSV.

    Combines train_df and test_df in that order (train-first) to preserve the
    same row ordering as the pre-refactor wyscout_shots.parquet artifact.

    Validates that both inputs have exactly KEEP_COLS in order before writing,
    so a schema drift upstream raises an explicit error here rather than silently
    producing a malformed parquet.

    Creates data_dir if it does not already exist.

    Output files written to data_dir:
      wyscout_shots.parquet         — train + test combined (43,040 rows)
      wyscout_train_shots.parquet   — train leagues only    (34,159 rows)
      wyscout_test_shots.parquet    — test leagues only     ( 8,881 rows)
      wyscout_shots_sample.csv      — first 100 rows of combined

    Returns the combined DataFrame so the caller can run further checks.
    """
    from pathlib import Path
    data_dir = Path(data_dir)
    ensure_dir(data_dir)

    for label, df in [("train_df", train_df), ("test_df", test_df)]:
        if list(df.columns) != KEEP_COLS:
            raise ValueError(
                f"{label} columns do not match KEEP_COLS.\n"
                f"  Expected: {KEEP_COLS}\n"
                f"  Got:      {list(df.columns)}"
            )

    shots = pd.concat([train_df, test_df], ignore_index=True)

    shots.to_parquet(data_dir / "wyscout_shots.parquet", index=False)
    train_df.to_parquet(data_dir / "wyscout_train_shots.parquet", index=False)
    test_df.to_parquet(data_dir / "wyscout_test_shots.parquet", index=False)
    shots.head(100).to_csv(data_dir / "wyscout_shots_sample.csv", index=False)

    return shots
