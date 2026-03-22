import numpy as np
import pandas as pd
from src.config import (
    PITCH_LENGTH, PITCH_WIDTH, GOAL_WIDTH, GOAL_X,
    GOAL_Y_CENTER, GOAL_Y_LEFT, GOAL_Y_RIGHT,
    TAG_LEFT_FOOT, TAG_RIGHT_FOOT, TAG_HEAD,
)
from src.shot_filtering import is_shot_attempt


def extract_body_part_flags(event: dict) -> dict:
    """Return is_left_foot, is_right_foot, is_header booleans from a raw event's tags.

    Each value is True if the corresponding tag ID appears in the event's tags list:
      TAG_LEFT_FOOT=401, TAG_RIGHT_FOOT=402, TAG_HEAD=403

    Returns booleans (not integers). Caller may cast to a different dtype later if needed.
    Uses t.get("id") for robustness against malformed tag dicts.
    """
    tag_ids = {t.get("id") for t in (event.get("tags") or [])}
    return {
        "is_left_foot":  TAG_LEFT_FOOT  in tag_ids,
        "is_right_foot": TAG_RIGHT_FOOT in tag_ids,
        "is_header":     TAG_HEAD       in tag_ids,
    }


def build_tag_rows(leagues, load_events_fn, verbose=True) -> pd.DataFrame:
    """Extract body-part tag rows for all shot attempts across multiple leagues.

    Applies the same shot-attempt filter as notebook 01 (via is_shot_attempt from
    shot_filtering) so the returned rows align exactly with the shots parquet.

    Returns a DataFrame with columns:
      league, matchId, playerId, teamId, eventSec, matchPeriod,
      is_left_foot, is_right_foot, is_header

    When verbose=True (default), prints per-league attempt counts and total:
      "{league}: {n:,} attempts extracted"  (one per league)
      "\\nTotal tag rows: {total:,}"
    Set verbose=False to suppress prints when calling outside notebook contexts.
    """
    rows = []
    for league in leagues:
        events = load_events_fn(league)
        league_count = 0
        for ev in events:
            if not is_shot_attempt(ev):
                continue
            flags = extract_body_part_flags(ev)
            rows.append({
                "league":      league,
                "matchId":     ev["matchId"],
                "playerId":    ev["playerId"],
                "teamId":      ev["teamId"],
                "eventSec":    ev["eventSec"],
                "matchPeriod": ev["matchPeriod"],
                **flags,
            })
            league_count += 1
        if verbose:
            print(f"{league}: {league_count:,} attempts extracted")
    if verbose:
        print(f"\nTotal tag rows: {len(rows):,}")
    TAG_COLS = ["league", "matchId", "playerId", "teamId", "eventSec", "matchPeriod",
                "is_left_foot", "is_right_foot", "is_header"]
    if not rows:
        return pd.DataFrame(columns=TAG_COLS)
    return pd.DataFrame(rows, columns=TAG_COLS)


def add_body_part_label(tags_df: pd.DataFrame) -> pd.DataFrame:
    """Add shot_body_part column to tags_df using vectorised np.select.

    Precedence order: header > right_foot > left_foot > unknown.
    When shot_body_part == 'unknown', all three boolean flags are False.
    Returns a copy with shot_body_part appended.
    """
    df = tags_df.copy()
    df["shot_body_part"] = np.select(
        [df["is_header"], df["is_right_foot"], df["is_left_foot"]],
        ["head",          "right_foot",        "left_foot"],
        default="unknown",
    )
    return df


def add_shot_geometry(shots: pd.DataFrame) -> pd.DataFrame:
    """Compute geometry features from Wyscout normalised [0-100] coordinates.

    Wyscout convention: x increases toward the attacking goal (x=100), y is
    lateral. All shots attack in the same direction — no mirroring required.

    Uses pitch constants from config (PITCH_LENGTH=105m, PITCH_WIDTH=68m,
    GOAL_WIDTH=7.32m, GOAL_X=105m, GOAL_Y_CENTER=34m, GOAL_Y_LEFT=30.34m,
    GOAL_Y_RIGHT=37.66m). Shot angle is computed via law of cosines —
    robust for wide / behind-post positions.

    Adds three columns to a copy of shots:
      distance_to_goal  — Euclidean distance to goal centre (metres, float)
      shot_angle_rad    — Shot angle subtended by goal posts (radians, [0, π])
      shot_angle_deg    — Same angle in degrees (inspection/reporting only)

    Raises AssertionError if x or y are outside [0, 100].
    """
    assert shots["x"].between(0, 100).all(), "x out of [0, 100]"
    assert shots["y"].between(0, 100).all(), "y out of [0, 100]"

    df = shots.copy()
    x_m = df["x"] * PITCH_LENGTH / 100
    y_m = df["y"] * PITCH_WIDTH  / 100
    dx  = GOAL_X - x_m

    df["distance_to_goal"] = np.sqrt(dx**2 + (y_m - GOAL_Y_CENTER)**2)

    a2 = dx**2 + (y_m - GOAL_Y_LEFT)**2
    b2 = dx**2 + (y_m - GOAL_Y_RIGHT)**2
    cos_angle = (a2 + b2 - GOAL_WIDTH**2) / (2 * np.sqrt(a2 * b2))
    df["shot_angle_rad"] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    df["shot_angle_deg"] = np.degrees(df["shot_angle_rad"])
    return df
