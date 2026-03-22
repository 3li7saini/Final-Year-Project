import json
import pandas as pd
from pathlib import Path
from src.config import EVENT_DIR, MATCH_DIR, PLAYER_FILE, TEAM_FILE, ALL_LEAGUES, LEAGUE_FILE_MAP


def load_events(league: str) -> list:
    """Load raw Wyscout events for a single league. Returns list of event dicts.

    Uses LEAGUE_FILE_MAP to resolve the filename; raises ValueError for unknown leagues
    and FileNotFoundError with a clear message if the file does not exist.
    """
    if league not in LEAGUE_FILE_MAP:
        raise ValueError(
            f"Unknown league '{league}'. Valid options: {list(LEAGUE_FILE_MAP.keys())}"
        )
    path = EVENT_DIR / LEAGUE_FILE_MAP[league]
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_events(leagues=None) -> dict:
    """Load raw events for multiple leagues. Returns {league: events_list}.
    Defaults to ALL_LEAGUES if leagues is None."""
    if leagues is None:
        leagues = ALL_LEAGUES
    return {league: load_events(league) for league in leagues}


def load_matches(league: str) -> list:
    """Load raw Wyscout match metadata for a single league. Returns list of match dicts.

    Raises ValueError for unknown leagues and FileNotFoundError if the file is missing.
    """
    if league not in LEAGUE_FILE_MAP:
        raise ValueError(
            f"Unknown league '{league}'. Valid options: {list(LEAGUE_FILE_MAP.keys())}"
        )
    path = MATCH_DIR / f"matches_{league}.json"
    if not path.exists():
        raise FileNotFoundError(f"Matches file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_matches(leagues=None) -> dict:
    """Load raw match metadata for multiple leagues. Returns {league: matches_list}."""
    if leagues is None:
        leagues = ALL_LEAGUES
    return {league: load_matches(league) for league in leagues}


def load_players() -> pd.DataFrame:
    """Load Wyscout players.json as a DataFrame.

    Raises FileNotFoundError with a clear message if the file does not exist.
    """
    if not PLAYER_FILE.exists():
        raise FileNotFoundError(f"Players file not found: {PLAYER_FILE}")
    with open(PLAYER_FILE, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def load_teams() -> pd.DataFrame:
    """Load Wyscout teams.json as a DataFrame.

    Raises FileNotFoundError with a clear message if the file does not exist.
    """
    if not TEAM_FILE.exists():
        raise FileNotFoundError(f"Teams file not found: {TEAM_FILE}")
    with open(TEAM_FILE, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def load_parquet(path) -> pd.DataFrame:
    """Load a parquet file with path existence check."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)
