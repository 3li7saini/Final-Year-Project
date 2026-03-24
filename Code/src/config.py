from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR      = PROJECT_ROOT / "data"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = PROJECT_ROOT / "models" / "xg"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
TABLES_DIR    = OUTPUTS_DIR / "tables"
RAW_DIR      = DATA_DIR / "raw" / "wyscout"
EVENT_DIR    = RAW_DIR / "events"
MATCH_DIR    = RAW_DIR / "matches"
PLAYER_FILE  = RAW_DIR / "players.json"
TEAM_FILE    = RAW_DIR / "teams.json"

# ── League constants ───────────────────────────────────────────────────────
# ALL_LEAGUES drives extraction iteration order (alphabetical — matches current NB01
# console output order: England, France, Germany, Italy, Spain).
# The saved parquet is train-first because save_shot_splits receives train_df
# and test_df separately and does pd.concat([train_df, test_df]) internally.
TRAIN_LEAGUES = ["France", "Germany", "Italy", "Spain"]
TEST_LEAGUES  = ["England"]
ALL_LEAGUES   = ["England", "France", "Germany", "Italy", "Spain"]  # alphabetical, extraction order

LEAGUE_FILE_MAP = {
    "England": "events_England.json",
    "France":  "events_France.json",
    "Germany": "events_Germany.json",
    "Italy":   "events_Italy.json",
    "Spain":   "events_Spain.json",
}

RANDOM_SEED = 42

# ── Pitch geometry (used by feature_engineering.py in Step 3) ─────────────
PITCH_LENGTH  = 105.0
PITCH_WIDTH   =  68.0
GOAL_WIDTH    =   7.32
GOAL_X        = 105.0
GOAL_Y_CENTER =  34.0
GOAL_Y_LEFT   =  30.34
GOAL_Y_RIGHT  =  37.66

# ── Wyscout tag IDs ────────────────────────────────────────────────────────
# Used by shot_filtering.py and feature_engineering.py (Step 3).
TAG_GOAL       = 101
TAG_LEFT_FOOT  = 401
TAG_RIGHT_FOOT = 402
TAG_HEAD       = 403

# ── Locked feature lists (used from Step 4 onwards) ───────────────────────
BASELINE_FEATURES = [
    "distance_to_goal", "shot_angle_rad",
    "is_penalty", "is_direct_free_kick",
    "is_left_foot", "is_right_foot", "is_header",
]
SKILL_COLS = [
    "career_shots_before_shot",
    "career_goals_before_shot",
    "career_non_goal_shots_before_shot",
    "career_conversion_rate_before_shot",
    "has_prior_shot_history",
]
FORM_COLS = [
    "goals_last_5_matches",
    "shots_last_5_matches",
    "conversion_rate_last_5_matches",
    "matches_in_form_window",
    "xg_sum_last_5_matches",
    "goals_minus_xg_last_5_matches",
    "shots_per_match_last_5_matches",
    "xg_per_shot_last_5_matches",
]
TARGET_COL = "is_goal"

# ── Chronological ordering (used by preprocessing.py in Step 2) ───────────
PERIOD_ORDER   = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}
PERIOD_OFFSETS = {"1H": 0, "2H": 2700, "E1": 5400, "E2": 6300, "P": 7200}
