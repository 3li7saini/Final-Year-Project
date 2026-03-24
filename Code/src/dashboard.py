"""Shared data loaders (Streamlit-cached), Plotly figure builders, and styling for the xG dashboard."""

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

from sklearn.metrics import roc_curve

from src.config import PROCESSED_DIR, TABLES_DIR

# ── Display constants ────────────────────────────────────────────────────
MODEL_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "skill_aware": "Skill-aware",
    "form_aware": "Form-aware",
    "combined": "Combined",
}
MODEL_ORDER: list[str] = ["Baseline", "Skill-aware", "Form-aware", "Combined"]
MODEL_COLORS: dict[str, str] = {
    "Baseline": "#1f77b4",
    "Skill-aware": "#ff8011",
    "Form-aware": "#2ca02c",
    "Combined": "#d62728",
}
PRED_COLS: dict[str, str] = {
    "Baseline": "xg_pred",
    "Skill-aware": "xg_pred_skill_aware",
    "Form-aware": "xg_pred_form_aware",
    "Combined": "xg_pred_combined",
}
MODEL_MARKERS: dict[str, str] = {
    "Baseline": "circle",
    "Skill-aware": "square",
    "Form-aware": "triangle-up",
    "Combined": "diamond",
}
MODEL_DASHES: dict[str, str] = {
    "Baseline": "solid",
    "Skill-aware": "dash",
    "Form-aware": "dot",
    "Combined": "dashdot",
}
MODEL_PATTERNS: dict[str, str] = {
    "Baseline": "",
    "Skill-aware": "/",
    "Form-aware": "x",
    "Combined": "\\",
}

# ── Shared Plotly template (inherits from plotly_white) ──────────────────
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        template=pio.templates["plotly_white"],
        font=dict(family="Segoe UI, sans-serif"),
        title_font=dict(family="Georgia, serif", size=16),
        colorway=[MODEL_COLORS[m] for m in MODEL_ORDER],
    )
)


# ── Error guards ─────────────────────────────────────────────────────────
def _require_file(path: Path, label: str = "") -> None:
    """Show a user-friendly Streamlit error and stop if a mandatory file is missing."""
    if not path.exists():
        name = label or path.name
        st.error(
            f"Required data file not found: **{name}**\n\n"
            f"Expected at: `{path}`\n\n"
            "Please run the notebook pipeline (NB01\u2013NB11) to generate all artifacts."
        )
        st.stop()


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Show a user-friendly Streamlit error and stop if expected columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        st.error(
            f"**{label}** is missing expected columns: {sorted(missing)}\n\n"
            "The file may be from an older pipeline run. Re-run notebooks to regenerate."
        )
        st.stop()


def _require_model_set(
    df: pd.DataFrame, label: str, *, rows_per_model: int | None = None
) -> None:
    """Validate that the 'model' column contains exactly the expected four models."""
    if df["model"].isna().any():
        unknown = df.loc[df["model"].isna(), "model_name"].unique().tolist()
        st.error(f"**{label}** contains unrecognised model names: {unknown}")
        st.stop()
    actual = set(df["model"].unique())
    expected = set(MODEL_ORDER)
    if actual != expected:
        st.error(f"**{label}** has wrong model set: {sorted(actual)} (expected {sorted(expected)})")
        st.stop()
    if rows_per_model is not None:
        counts = df["model"].value_counts().to_dict()
        bad = {m: c for m, c in counts.items() if c != rows_per_model}
        if bad:
            st.error(f"**{label}** expected {rows_per_model} rows per model, got: {bad}")
            st.stop()


# ── Cache invalidation ───────────────────────────────────────────────────
def _file_mtime(path: Path) -> float:
    """Return file modification time, or 0.0 if file does not exist."""
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _cached_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with Streamlit caching keyed to file mtime."""
    return _cached_read_csv_impl(str(path), _file_mtime(path), **kwargs)


@st.cache_data
def _cached_read_csv_impl(path_str: str, _mtime: float, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path_str, **kwargs)


def _cached_read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet with Streamlit caching keyed to file mtime."""
    return _cached_read_parquet_impl(str(path), _file_mtime(path))


@st.cache_data
def _cached_read_parquet_impl(path_str: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


# ── Public loaders ───────────────────────────────────────────────────────

def load_metrics() -> pd.DataFrame:
    """4-row model comparison metrics with display label column."""
    path = PROCESSED_DIR / "wyscout_xg_model_comparison_metrics.csv"
    _require_file(path)
    df = _cached_read_csv(path)
    _require_columns(
        df,
        ["model_name", "roc_auc", "log_loss", "brier_score",
         "average_precision", "auc_gain", "n_shots"],
        "Metrics CSV",
    )
    df = df.copy()
    df["model"] = df["model_name"].map(MODEL_LABELS)
    _require_model_set(df, "Metrics CSV", rows_per_model=1)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def load_shot_predictions() -> pd.DataFrame:
    """8,881-row shot-level predictions parquet (used for ROC curve computation)."""
    path = PROCESSED_DIR / "wyscout_xg_model_comparison.parquet"
    _require_file(path)
    df = _cached_read_parquet(path)
    _require_columns(df, ["is_goal"] + list(PRED_COLS.values()), "Shot predictions parquet")
    return df


def load_calibration_table() -> pd.DataFrame:
    """40-row saved calibration table with display label column."""
    path = PROCESSED_DIR / "wyscout_xg_model_comparison_calibration_table.csv"
    _require_file(path)
    df = _cached_read_csv(path)
    _require_columns(
        df,
        ["model_name", "xg_bin", "n_shots", "mean_pred_xg", "observed_goal_rate"],
        "Calibration CSV",
    )
    df = df.copy()
    df["model"] = df["model_name"].map(MODEL_LABELS)
    _require_model_set(df, "Calibration CSV", rows_per_model=10)
    return df


def load_team_match() -> pd.DataFrame:
    """760-row team-match parquet (match explorer).
    Each match appears twice (one row per team) — the file is team-centric,
    not match-centric.
    """
    path = PROCESSED_DIR / "wyscout_xg_team_match.parquet"
    _require_file(path)
    df = _cached_read_parquet(path)
    _require_columns(
        df,
        ["team_name", "gameweek", "match_label", "official_goals",
         "opponent_goals", "baseline_match_xg", "combined_match_xg",
         "baseline_xg_points", "combined_xg_points"],
        "Team-match parquet",
    )
    return df


def load_league_comparison() -> pd.DataFrame:
    """20-row league table comparison — single source of truth for season displays."""
    path = TABLES_DIR / "final_league_table_comparison.csv"
    _require_file(path)
    df = _cached_read_csv(path)
    _require_columns(
        df,
        ["team_name", "actual_pts", "baseline_xg_pts", "combined_xg_pts",
         "actual_rank", "baseline_rank", "combined_rank",
         "baseline_points_diff", "combined_points_diff",
         "baseline_rank_diff", "combined_rank_diff"],
        "League comparison CSV",
    )
    if len(df) != 20:
        st.error(f"Expected 20 teams in league comparison, got {len(df)}")
        st.stop()
    if df["team_name"].nunique() != 20:
        st.error("Duplicate team names in league comparison CSV")
        st.stop()
    return df


def load_coefficients() -> tuple[pd.DataFrame | None, str]:
    """Load coefficient CSV. Returns (df, reason) where df is None if unavailable.
    reason is empty string on success, or a human-readable explanation on failure.
    Page code should call st.warning(reason) if df is None and reason is non-empty.
    """
    path = PROCESSED_DIR / "wyscout_xg_model_coefficients.csv"
    if not path.exists():
        return None, ""
    try:
        df = _cached_read_csv(path)
    except Exception as e:
        return None, f"Coefficients file could not be read: {e}"
    required = ["model_name", "feature", "coefficient", "abs_coefficient"]
    missing = set(required) - set(df.columns)
    if missing:
        return None, f"Coefficients file is missing columns: {sorted(missing)}"
    if not (df["model_name"] == "combined").any():
        return None, "No combined-model rows found in coefficients file"
    return df, ""


# ── Styling + shared sidebar ─────────────────────────────────────────────
def set_academic_style() -> None:
    """Inject CSS for academic presentation: serif headings, restrained sizing."""
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 1.3rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        h1, h2, h3 { font-family: "Georgia", serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the shared 'About' sidebar block. Call from every page for consistency."""
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "**Data:** Wyscout 2017-18, 5 European leagues\n\n"
            "**Train:** France, Germany, Italy, Spain (34,159 shots)\n\n"
            "**Test:** England Premier League (8,881 shots)\n\n"
            "**Models:** Logistic Regression (4 variants)"
        )


def init_page(title: str) -> None:
    """Standard page setup \u2014 must be the FIRST Streamlit call in every page file.
    Sets page config (wide layout), injects academic CSS, and renders the sidebar."""
    st.set_page_config(page_title=f"{title} \u2014 xG Dashboard", layout="wide")
    set_academic_style()
    render_sidebar()


# ── Overview figures ─────────────────────────────────────────────────────

def fig_metric_bars(metrics_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Horizontal bar chart for one metric across all four models."""
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        model = row["model"]
        fig.add_trace(go.Bar(
            y=[model],
            x=[row[metric]],
            orientation="h",
            marker_color=MODEL_COLORS[model],
            marker_pattern_shape=MODEL_PATTERNS[model],
            marker_pattern_fillmode="replace",
            marker_pattern_fgcolor="white",
            name=model,
            text=f"{row[metric]:.4f}",
            textposition="outside",
            showlegend=False,
        ))
    fig.update_layout(
        title=title,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(MODEL_ORDER))),
        height=300,
        margin=dict(l=20, r=60, t=40, b=20),
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_metric_table(metrics_df: pd.DataFrame) -> go.Figure:
    """Plotly Table showing all four models' key metrics."""
    display_df = metrics_df[["model", "roc_auc", "average_precision", "log_loss", "brier_score"]].copy()
    for c in ["roc_auc", "average_precision", "log_loss", "brier_score"]:
        display_df[c] = display_df[c].round(4)
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Model", "ROC AUC", "Avg Precision", "Log Loss", "Brier Score"],
            fill_color="#f0f2f6",
            align="left",
            font=dict(family="Georgia, serif", size=13),
        ),
        cells=dict(
            values=[display_df[c] for c in display_df.columns],
            align="left",
            font=dict(family="Segoe UI, sans-serif", size=12),
        ),
    )])
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ── Model Analysis figures ───────────────────────────────────────────────

def fig_roc_curves(shots_df: pd.DataFrame, metrics_df: pd.DataFrame) -> go.Figure:
    """ROC curves for all four models. Curves recomputed from shot-level predictions;
    AUC values in legend come from saved metrics (not sklearn.metrics.auc)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="grey"), showlegend=False,
    ))
    auc_lookup = dict(zip(metrics_df["model"], metrics_df["roc_auc"]))
    for model_label in MODEL_ORDER:
        pred_col = PRED_COLS[model_label]
        fpr, tpr, _ = roc_curve(shots_df["is_goal"], shots_df[pred_col])
        auc_val = auc_lookup[model_label]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{model_label} (AUC={auc_val:.4f})",
            line=dict(color=MODEL_COLORS[model_label], dash=MODEL_DASHES[model_label]),
        ))
    fig.update_layout(
        title="ROC Curves \u2014 Held-out EPL Shots",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        template=PLOTLY_TEMPLATE,
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def fig_calibration(cal_df: pd.DataFrame) -> go.Figure:
    """Calibration plot from saved calibration table (not recomputed)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="grey"), name="Perfect",
    ))
    for model_label in MODEL_ORDER:
        subset = cal_df[cal_df["model"] == model_label].sort_values("mean_pred_xg")
        fig.add_trace(go.Scatter(
            x=subset["mean_pred_xg"],
            y=subset["observed_goal_rate"],
            mode="lines+markers",
            name=model_label,
            line=dict(color=MODEL_COLORS[model_label]),
            marker=dict(symbol=MODEL_MARKERS[model_label], size=8),
        ))
    fig.update_layout(
        title="Calibration Curves (Quantile Bins)",
        xaxis_title="Mean Predicted xG",
        yaxis_title="Observed Goal Rate",
        height=500,
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_coefficient_importance(coeffs_df: pd.DataFrame, top_n: int = 15) -> go.Figure | None:
    """Horizontal bar chart of top-N combined-model features by |coefficient|.
    Returns None if no combined-model rows exist."""
    combined = coeffs_df[coeffs_df["model_name"] == "combined"].copy()
    if combined.empty:
        return None
    top = combined.nlargest(top_n, "abs_coefficient")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["coefficient"]]
    fig = go.Figure(go.Bar(
        y=top["feature"],
        x=top["coefficient"],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Combined Model \u2014 Top {top_n} Features by |Coefficient|",
        xaxis_title="Coefficient",
        yaxis=dict(autorange="reversed"),
        height=450,
        template=PLOTLY_TEMPLATE,
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    return fig


# ── Season Impact figures ────────────────────────────────────────────────

def fig_actual_vs_xg_scatter(comparison_df: pd.DataFrame, model: str = "combined") -> go.Figure:
    """Scatter: actual points vs xG points for one model, with team annotations."""
    pts_col = f"{model}_xg_pts"
    fig = go.Figure()
    lo = min(comparison_df["actual_pts"].min(), comparison_df[pts_col].min()) - 3
    hi = max(comparison_df["actual_pts"].max(), comparison_df[pts_col].max()) + 3
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color="grey"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=comparison_df["actual_pts"],
        y=comparison_df[pts_col],
        mode="markers+text",
        text=comparison_df["team_name"],
        textposition="top right",
        textfont=dict(size=9),
        marker=dict(size=8, color=MODEL_COLORS.get(model.title(), "#1f77b4")),
        showlegend=False,
    ))
    fig.update_layout(
        title=f"2017/18 EPL: Actual vs {model.replace('_', ' ').title()} xG Points",
        xaxis_title="Actual Points",
        yaxis_title="xG Points",
        xaxis=dict(range=[lo, hi]),
        yaxis=dict(range=[lo, hi], scaleanchor="x"),
        height=550,
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_rank_difference(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar chart: rank difference for baseline and combined models."""
    comp = comparison_df.sort_values("actual_rank")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=comp["team_name"], x=comp["baseline_rank_diff"],
        orientation="h", name="Baseline",
        marker_color=MODEL_COLORS["Baseline"],
        marker_pattern_shape=MODEL_PATTERNS["Baseline"],
        marker_pattern_fillmode="replace",
        marker_pattern_fgcolor="white",
    ))
    fig.add_trace(go.Bar(
        y=comp["team_name"], x=comp["combined_rank_diff"],
        orientation="h", name="Combined",
        marker_color=MODEL_COLORS["Combined"],
        marker_pattern_shape=MODEL_PATTERNS["Combined"],
        marker_pattern_fillmode="replace",
        marker_pattern_fgcolor="white",
    ))
    fig.update_layout(
        barmode="group",
        title="Rank Difference from Actual",
        xaxis_title="Rank Difference (model \u2212 actual)",
        yaxis=dict(autorange="reversed"),
        height=600,
        template=PLOTLY_TEMPLATE,
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    return fig


def fig_points_difference(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bar chart: points difference for baseline and combined models."""
    comp = comparison_df.sort_values("actual_rank")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=comp["team_name"], x=comp["baseline_points_diff"],
        orientation="h", name="Baseline",
        marker_color=MODEL_COLORS["Baseline"],
        marker_pattern_shape=MODEL_PATTERNS["Baseline"],
        marker_pattern_fillmode="replace",
        marker_pattern_fgcolor="white",
    ))
    fig.add_trace(go.Bar(
        y=comp["team_name"], x=comp["combined_points_diff"],
        orientation="h", name="Combined",
        marker_color=MODEL_COLORS["Combined"],
        marker_pattern_shape=MODEL_PATTERNS["Combined"],
        marker_pattern_fillmode="replace",
        marker_pattern_fgcolor="white",
    ))
    fig.update_layout(
        barmode="group",
        title="Points Difference from Actual",
        xaxis_title="Points Difference (model \u2212 actual)",
        yaxis=dict(autorange="reversed"),
        height=600,
        template=PLOTLY_TEMPLATE,
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    return fig
