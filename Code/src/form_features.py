import numpy as np
import pandas as pd


FORM_SORT_COLS = [
    "match_datetime_utc",
    "matchId",
    "period_sort_key",
    "seconds_from_match_start",
    "shot_sequence_in_match",
    "league",
    "playerId",
    "teamId",
]

PLAYER_MATCH_SORT = ["playerId", "match_datetime_utc", "matchId", "league"]


def build_player_match_form_table(shots_df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """Build player-match level table and compute 8 rolling form features.

    Input: combined (train+test) shots DataFrame. The notebook pre-sorts the combined
    DataFrame and validates monotonicity before calling this function. The function
    also sorts player_match internally by PLAYER_MATCH_SORT in Phase 2 — this is a
    defensive re-sort to make the rolling dependency explicit, not a reliance on the
    caller's ordering.

    Phase 1 — Aggregate shots to player-match level:
      Groups by (playerId, matchId), aggregates:
        league, match_datetime_utc, match_date, dataset_split (provenance only),
        match_shots (count of is_goal), match_goals (sum of is_goal),
        match_xg_sum (sum of xg_pred).
      Casts match_shots and match_goals to int32.
      Validates: one row per (playerId, matchId), no null keys, goals <= shots.

    Phase 2 — Sort by PLAYER_MATCH_SORT:
      ["playerId", "match_datetime_utc", "matchId", "league"]
      Required before rolling; re-sort is explicit to make the ordering dependency clear.

    Phase 3 — Compute 8 rolling form features using shift(1) + rolling(n_matches, min_periods=1):
      goals_last_5_matches           (int32)  — goals in prior n_matches
      shots_last_5_matches           (int32)  — shots in prior n_matches
      matches_in_form_window         (int8)   — prior matches in window
      conversion_rate_last_5_matches (float32) — goals/shots; 0.0 when no history
      xg_sum_last_5_matches          (float32) — sum of xg_pred in prior n_matches
      goals_minus_xg_last_5_matches  (float32) — goals - xg_sum in window
      shots_per_match_last_5_matches (float32) — shots / matches; 0.0 when no history
      xg_per_shot_last_5_matches     (float32) — xg_sum / shots; 0.0 when no shots

    CRITICAL: shift(1) ensures the current match never contributes to its own form
    window. The first match for every player has all form features = 0.

    Note: groupby().apply(lambda s: s.shift(1).rolling(...)) may emit FutureWarning
    in pandas 2.x — no redesign needed, suppress if noisy.

    Returns the player_match DataFrame (one row per (playerId, matchId)) with 8 new
    columns appended. Caller merges back to shot level via (playerId, matchId).
    """
    # Phase 1: Aggregate
    player_match = (
        shots_df.groupby(["playerId", "matchId"], as_index=False)
        .agg(
            league=("league", "first"),
            match_datetime_utc=("match_datetime_utc", "first"),
            match_date=("match_date", "first"),
            dataset_split=("dataset_split", "first"),
            match_shots=("is_goal", "count"),
            match_goals=("is_goal", "sum"),
            match_xg_sum=("xg_pred", "sum"),
        )
    )
    player_match["match_shots"] = player_match["match_shots"].astype("int32")
    player_match["match_goals"] = player_match["match_goals"].astype("int32")

    assert not player_match[["playerId", "matchId"]].duplicated().any(), \
        "player_match is not unique per (playerId, matchId)"
    assert player_match[["playerId", "matchId", "match_datetime_utc"]].notna().all().all(), \
        "Null key columns in player_match"
    assert (player_match["match_goals"] <= player_match["match_shots"]).all(), \
        "match_goals exceeds match_shots"

    # Phase 2: Sort
    player_match = player_match.sort_values(PLAYER_MATCH_SORT, ignore_index=True)

    # Phase 3: Rolling form
    g = player_match.groupby("playerId", group_keys=False)

    player_match["goals_last_5_matches"] = (
        g["match_goals"]
        .apply(lambda s: s.shift(1).rolling(n_matches, min_periods=1).sum())
        .fillna(0)
        .astype("int32")
    )
    player_match["shots_last_5_matches"] = (
        g["match_shots"]
        .apply(lambda s: s.shift(1).rolling(n_matches, min_periods=1).sum())
        .fillna(0)
        .astype("int32")
    )
    player_match["matches_in_form_window"] = (
        g["match_shots"]
        .apply(lambda s: s.shift(1).rolling(n_matches, min_periods=1).count())
        .fillna(0)
        .astype("int8")
    )
    player_match["conversion_rate_last_5_matches"] = np.where(
        player_match["shots_last_5_matches"] > 0,
        player_match["goals_last_5_matches"] / player_match["shots_last_5_matches"],
        0.0,
    ).astype("float32")
    player_match["xg_sum_last_5_matches"] = (
        g["match_xg_sum"]
        .apply(lambda s: s.shift(1).rolling(n_matches, min_periods=1).sum())
        .fillna(0)
        .astype("float32")
    )
    player_match["goals_minus_xg_last_5_matches"] = (
        player_match["goals_last_5_matches"] - player_match["xg_sum_last_5_matches"]
    ).astype("float32")
    player_match["shots_per_match_last_5_matches"] = np.where(
        player_match["matches_in_form_window"] > 0,
        player_match["shots_last_5_matches"] / player_match["matches_in_form_window"],
        0.0,
    ).astype("float32")
    player_match["xg_per_shot_last_5_matches"] = np.where(
        player_match["shots_last_5_matches"] > 0,
        player_match["xg_sum_last_5_matches"] / player_match["shots_last_5_matches"],
        0.0,
    ).astype("float32")

    return player_match
