"""Page — Season Impact: league table, scatter, bar charts, match explorer."""

import streamlit as st
from src.dashboard import (
    init_page,
    load_league_comparison, load_team_match,
    fig_actual_vs_xg_scatter, fig_rank_difference, fig_points_difference,
)

init_page("Season Impact")

st.header("Season Impact")

st.markdown(
    "xG points replace actual match results with expected results: "
    "3 points if team xG > opponent xG, 1 point if team xG == opponent xG, "
    "0 points if team xG < opponent xG (strict comparison, no threshold). "
    "This section compares the 2017/18 EPL table under actual results vs "
    "xG-based results."
)

comparison = load_league_comparison()
team_match = load_team_match()

# ── League table ─────────────────────────────────────────────────────────
st.subheader("League Table Comparison")
display_cols = [
    "team_name", "actual_pts", "baseline_xg_pts", "combined_xg_pts",
    "actual_rank", "baseline_rank", "combined_rank",
]
st.dataframe(
    comparison[display_cols].sort_values("actual_rank"),
    use_container_width=True,
    hide_index=True,
)

# ── Scatter: actual vs xG points ────────────────────────────────────────
st.subheader("Actual vs xG Points")
model_choice = st.radio(
    "Model", ["baseline", "combined"], horizontal=True, format_func=str.title,
)
st.plotly_chart(
    fig_actual_vs_xg_scatter(comparison, model=model_choice),
    use_container_width=True,
)
st.caption(
    "Teams above the diagonal earned fewer actual points than the model "
    "predicted (underperformed expectations)."
)

# ── Bar charts: rank and points differences ──────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_rank_difference(comparison), use_container_width=True)
    st.caption("Positive = model ranks team lower than actual finish.")
with col2:
    st.plotly_chart(fig_points_difference(comparison), use_container_width=True)
    st.caption("Positive = model awards more xG points than actually earned.")

# ── Match explorer ───────────────────────────────────────────────────────
st.subheader("Match Explorer")

st.sidebar.divider()
st.sidebar.subheader("Match Explorer Filters")

all_teams = sorted(team_match["team_name"].unique())
team_filter = st.sidebar.multiselect("Teams", options=all_teams, default=all_teams)
gw_range = st.sidebar.slider("Gameweek range", 1, 38, (1, 38))

filtered = team_match[
    (team_match["team_name"].isin(team_filter))
    & (team_match["gameweek"].between(gw_range[0], gw_range[1]))
]

if filtered.empty:
    st.info("No matches found for the selected filters.")
else:
    explorer_cols = [
        "gameweek", "match_label", "team_name",
        "official_goals", "opponent_goals",
        "baseline_match_xg", "combined_match_xg",
        "baseline_xg_points", "combined_xg_points",
    ]
    st.dataframe(
        filtered[explorer_cols].sort_values(["gameweek", "match_label", "team_name"]),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

st.caption(
    "This table is team-centric: each match appears as one row per team. "
    "When both teams in a match are selected, the match shows twice "
    "(once from each team's perspective)."
)
