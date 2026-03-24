"""xG Model Dashboard — Overview (home page)."""

import streamlit as st
from src.dashboard import init_page, load_metrics, fig_metric_bars, fig_metric_table

init_page("Overview")

st.header("Model Performance Overview")

# ── KPI cards ────────────────────────────────────────────────────────────
metrics = load_metrics()

cols = st.columns(4)
for col, (_, row) in zip(cols, metrics.iterrows()):
    with col:
        delta = (
            f"+{row['auc_gain']:.4f} vs Baseline"
            if row["model"] != "Baseline"
            else None
        )
        col.metric(
            label=row["model"],
            value=f"{row['roc_auc']:.4f}",
            delta=delta,
            help=f"Log loss: {row['log_loss']:.4f} | Brier: {row['brier_score']:.4f} | {int(row['n_shots']):,} shots",
        )

# ── Takeaway ─────────────────────────────────────────────────────────────
best_auc_row = metrics.loc[metrics["roc_auc"].idxmax()]
best_brier_row = metrics.loc[metrics["brier_score"].idxmin()]

st.markdown(
    f"The **{best_auc_row['model']}** model achieves the highest AUC "
    f"({best_auc_row['roc_auc']:.4f}) and lowest Brier score "
    f"({best_brier_row['brier_score']:.4f}), indicating modest but consistent "
    "improvement over the Baseline."
)

# ── Metric table ─────────────────────────────────────────────────────────
st.plotly_chart(fig_metric_table(metrics), use_container_width=True)

# ── Bar charts ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(fig_metric_bars(metrics, "roc_auc", "ROC AUC"), use_container_width=True)
    st.caption("Higher AUC indicates better discrimination between goals and non-goals.")

with col2:
    st.plotly_chart(fig_metric_bars(metrics, "log_loss", "Log Loss"), use_container_width=True)
    st.caption("Lower log loss indicates better calibrated probability estimates.")

with col3:
    st.plotly_chart(fig_metric_bars(metrics, "brier_score", "Brier Score"), use_container_width=True)
    st.caption("Lower Brier score indicates smaller average prediction errors.")

# ── Metric explainer ─────────────────────────────────────────────────────
with st.expander("About these metrics"):
    st.markdown(
        "- **ROC AUC**: Discrimination \u2014 higher is better.\n"
        "- **Average Precision**: Area under precision-recall curve \u2014 higher is better.\n"
        "- **Log Loss**: Calibration-sensitive \u2014 lower is better.\n"
        "- **Brier Score**: Mean squared error of probabilities \u2014 lower is better."
    )
