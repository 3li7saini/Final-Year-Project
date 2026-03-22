import numpy as np
import pandas as pd


SKILL_SORT_COLS = [
    "match_datetime_utc",
    "matchId",
    "period_sort_key",
    "seconds_from_match_start",
    "shot_sequence_in_match",
    "league",
    "playerId",
    "teamId",
]


def add_career_skill_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Sort combined train+test globally and compute 5 cumulative career columns.

    Input is the unsorted concatenation of train and test DataFrames (including the
    dataset_split marker column added by the notebook before concat).

    Sorts by SKILL_SORT_COLS:
      match_datetime_utc, matchId, period_sort_key, seconds_from_match_start,
      shot_sequence_in_match, league, playerId, teamId.

    For each player (groupby playerId), computes BEFORE the current shot:
      career_shots_before_shot          (int32) — cumcount() gives 0-based prior count
      career_goals_before_shot          (int32) — cumsum() minus current is_goal
      career_non_goal_shots_before_shot (int32) — shots minus goals
      has_prior_shot_history            (int8)  — 1 if shots > 0, else 0
      career_conversion_rate_before_shot (float32) — goals/shots; 0.0 if no history

    CRITICAL: No data leakage — current shot is NOT included in any cumulative count.
    cumcount() returns 0 for the first row per player (sort order), so each value is
    always the count of rows before it. cumsum() - current_value achieves the same
    for goal counts.

    Validates that SKILL_SORT_COLS and is_goal are null-free before sorting.
    Returns a copy sorted by SKILL_SORT_COLS with 5 new columns appended.
    """
    df = combined_df.copy()

    for col in SKILL_SORT_COLS + ["is_goal"]:
        assert df[col].notna().all(), f"Null values in {col} before sort"

    df = df.sort_values(SKILL_SORT_COLS, ignore_index=True)

    df["career_shots_before_shot"] = (
        df.groupby("playerId").cumcount().astype("int32")
    )

    df["career_goals_before_shot"] = (
        df.groupby("playerId")["is_goal"]
        .cumsum()
        .sub(df["is_goal"])
        .astype("int32")
    )

    df["career_non_goal_shots_before_shot"] = (
        df["career_shots_before_shot"] - df["career_goals_before_shot"]
    ).astype("int32")

    # Dtype and range assertions matching cell-step3 (preserved from notebook)
    assert df["career_shots_before_shot"].dtype == "int32"
    assert df["career_goals_before_shot"].dtype == "int32"
    assert df["career_non_goal_shots_before_shot"].dtype == "int32"
    assert df["career_shots_before_shot"].ge(0).all(), "Negative career_shots_before_shot"
    assert df["career_goals_before_shot"].ge(0).all(), "Negative career_goals_before_shot"
    assert df["career_non_goal_shots_before_shot"].ge(0).all(), \
        "Negative career_non_goal_shots_before_shot"
    assert (df["career_goals_before_shot"] <= df["career_shots_before_shot"]).all(), \
        "Goals exceed shots in prior history"

    df["has_prior_shot_history"] = (
        df["career_shots_before_shot"] > 0
    ).astype("int8")

    df["career_conversion_rate_before_shot"] = np.where(
        df["career_shots_before_shot"] > 0,
        df["career_goals_before_shot"] / df["career_shots_before_shot"],
        0.0,
    ).astype("float32")

    # Dtype and range assertions matching cell-step4 (preserved from notebook)
    assert df["has_prior_shot_history"].dtype == "int8"
    assert df["career_conversion_rate_before_shot"].dtype == "float32"
    assert df["has_prior_shot_history"].isin([0, 1]).all(), \
        "has_prior_shot_history not binary"
    assert df["career_conversion_rate_before_shot"].between(0, 1).all(), \
        "career_conversion_rate_before_shot out of [0, 1]"
    assert (df.loc[
        df["career_shots_before_shot"] == 0,
        "career_conversion_rate_before_shot"
    ] == 0.0).all(), "Zero-history rows must have conversion rate = 0.0"

    return df
