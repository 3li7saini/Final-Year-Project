import pandas as pd
from src.config import PERIOD_ORDER, PERIOD_OFFSETS, TRAIN_LEAGUES, TEST_LEAGUES


def period_sort_key(period_series: pd.Series) -> pd.Series:
    """Map matchPeriod strings to integer sort order (1H→1, 2H→2, E1→3, E2→4, P→5).

    Uses PERIOD_ORDER from config. Returns NaN for unrecognised period strings,
    which the caller should assert against.
    """
    return period_series.map(PERIOD_ORDER)


def compute_match_seconds(df: pd.DataFrame) -> pd.Series:
    """Return seconds_from_match_start for every row (vectorised).

    Continuous match-time ordering key — NOT true ball-in-play time.
    Wyscout's eventSec resets at the start of each period; this function
    adds a fixed period offset (from PERIOD_OFFSETS in config) so rows
    can be sorted in a single global pass.
    """
    offsets = df["matchPeriod"].map(PERIOD_OFFSETS)
    return offsets + df["eventSec"]


def build_match_df(raw_matches: list, league: str) -> pd.DataFrame:
    """Flatten a list of raw Wyscout match dicts into a DataFrame.

    Extracts scalar fields only (no nested team/score arrays):
      league, matchId (from wyId), match_date_utc_raw, match_date_local_raw,
      gameweek, roundId, seasonId, competitionId, match_label,
      match_status, match_duration, winner.

    NOTE: match_date_utc_raw is a raw string — caller must parse it with
    pd.to_datetime(..., utc=True) and then drop match_date_utc_raw.
    """
    rows = []
    for m in raw_matches:
        rows.append({
            "league":               league,
            "matchId":              m["wyId"],
            "match_date_utc_raw":   m.get("dateutc"),
            "match_date_local_raw": m.get("date"),
            "gameweek":             m.get("gameweek"),
            "roundId":              m.get("roundId"),
            "seasonId":             m.get("seasonId"),
            "competitionId":        m.get("competitionId"),
            "match_label":          m.get("label"),
            "match_status":         m.get("status"),
            "match_duration":       m.get("duration"),
            "winner":               m.get("winner"),
        })
    return pd.DataFrame(rows)


def build_all_match_df(leagues, load_matches_fn, verbose=True) -> pd.DataFrame:
    """Load, flatten, and validate match metadata for multiple leagues.

    Calls build_match_df per league, then parses match_datetime_utc (UTC-aware)
    and derives match_date (date objects).

    Validates:
      - No null matchId
      - All match_datetime_utc parsed successfully (no NaT)
      - No duplicate (league, matchId) pairs

    When verbose=True (default), prints per-league counts and total to preserve
    current NB02 console output:
      "{league}: {n:,} matches"  (one per league)
      "\\nTotal matches: {total:,}"
    Set verbose=False to suppress prints when calling outside notebook contexts.
    """
    dfs = []
    for league in leagues:
        raw = load_matches_fn(league)
        df = build_match_df(raw, league)
        if verbose:
            print(f"{league}: {len(df):,} matches")
        dfs.append(df)

    matches_df = pd.concat(dfs, ignore_index=True)

    # dateutc strings are naive (no tz marker) but represent UTC by Wyscout convention.
    # utc=True attaches UTC after parsing rather than requiring it in the string.
    matches_df["match_datetime_utc"] = pd.to_datetime(
        matches_df["match_date_utc_raw"], utc=True, errors="coerce"
    )
    matches_df["match_date"] = matches_df["match_datetime_utc"].dt.date

    assert matches_df["matchId"].notna().all(), "Null matchId in matches"
    assert matches_df["match_datetime_utc"].notna().all(), "Null datetime in matches"
    assert not matches_df[["league", "matchId"]].duplicated().any(), \
        "Duplicate (league, matchId) in matches"

    if verbose:
        print(f"\nTotal matches: {len(matches_df):,}")
    return matches_df


def merge_match_metadata(shots_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join shots to match metadata on (league, matchId).

    Drops match_date_utc_raw from the join (already parsed into match_datetime_utc).
    Validates:
      - Row count unchanged
      - All shots matched (no null match_datetime_utc)
      - gameweek non-null on every row (required for downstream rolling form features)

    Returns merged DataFrame with all match metadata columns appended.
    """
    merged = shots_df.merge(
        matches_df.drop(columns=["match_date_utc_raw"]),
        on=["league", "matchId"],
        how="left",
        validate="many_to_one",
    )
    assert len(merged) == len(shots_df), "Row count changed after merge"
    assert merged["match_datetime_utc"].notna().all(), \
        "Unmatched shots (null match_datetime_utc)"
    assert merged["gameweek"].notna().all(), \
        "Null gameweek after merge — check match files"
    return merged


def add_chronological_fields(shots: pd.DataFrame) -> pd.DataFrame:
    """Add period_sort_key, seconds_from_match_start, sort chronologically,
    and add shot_sequence_in_match and shot_sequence_team_in_match.

    Sort key: (match_datetime_utc, matchId, period_sort_key, eventSec, playerId, teamId)
    Index is reset after sorting (ignore_index=True).

    Validates that period_sort_key has no NaN (catches unknown matchPeriod values).
    """
    shots = shots.copy()
    shots["period_sort_key"] = period_sort_key(shots["matchPeriod"])
    assert shots["period_sort_key"].notna().all(), (
        f"Unexpected matchPeriod values: "
        f"{shots[shots['period_sort_key'].isna()]['matchPeriod'].unique()}"
    )

    shots["seconds_from_match_start"] = compute_match_seconds(shots)

    shots = shots.sort_values(
        ["match_datetime_utc", "matchId", "period_sort_key", "eventSec", "playerId", "teamId"],
        ignore_index=True,
    )

    shots["shot_sequence_in_match"] = (
        shots.groupby(["league", "matchId"]).cumcount() + 1
    )
    shots["shot_sequence_team_in_match"] = (
        shots.groupby(["league", "matchId", "teamId"]).cumcount() + 1
    )
    return shots


def split_train_test(shots: pd.DataFrame, train_leagues=None, test_leagues=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a combined shots DataFrame into (train_df, test_df) by league.

    Defaults to TRAIN_LEAGUES and TEST_LEAGUES from config.
    Validates that each split contains exactly the expected leagues.
    Returns (train_df, test_df) as copies.
    """
    if train_leagues is None:
        train_leagues = TRAIN_LEAGUES
    if test_leagues is None:
        test_leagues = TEST_LEAGUES

    train = shots[shots["league"].isin(train_leagues)].copy()
    test  = shots[shots["league"].isin(test_leagues)].copy()

    assert set(train["league"].unique()) == set(train_leagues), \
        f"Wrong leagues in train: {set(train['league'].unique())}"
    assert set(test["league"].unique()) == set(test_leagues), \
        f"Wrong leagues in test: {set(test['league'].unique())}"
    return train, test
