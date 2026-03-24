"""Page — Model Analysis: ROC curves, calibration, and feature importance."""

import streamlit as st
from src.dashboard import (
    init_page,
    load_metrics, load_shot_predictions, load_calibration_table, load_coefficients,
    fig_roc_curves, fig_calibration, fig_coefficient_importance,
)

init_page("Model Analysis")

st.header("Model Analysis")

st.markdown(
    "The ROC curve shows each model's ability to separate goals from non-goals "
    "across all decision thresholds. The calibration curve shows how well "
    "predicted probabilities match observed goal rates."
)

metrics = load_metrics()
shots = load_shot_predictions()
cal_table = load_calibration_table()
coeffs_df, coeffs_reason = load_coefficients()

if coeffs_reason:
    st.warning(coeffs_reason)

# ── Build tabs dynamically ───────────────────────────────────────────────
tab_names = ["ROC Curves", "Calibration"]
if coeffs_df is not None:
    tab_names.append("Feature Importance")

tabs = st.tabs(tab_names)

with tabs[0]:
    st.plotly_chart(fig_roc_curves(shots, metrics), use_container_width=True)
    st.caption("AUC values are from saved metrics. Curves are recomputed from shot-level predictions.")

with tabs[1]:
    st.plotly_chart(fig_calibration(cal_table), use_container_width=True)
    st.caption("Calibration bins are quantile-based (10 bins per model), loaded from the saved calibration table.")

if coeffs_df is not None:
    with tabs[2]:
        fig = fig_coefficient_importance(coeffs_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No combined-model coefficients available.")
        st.caption("Positive coefficients increase goal probability; negative decrease it. Only Combined model shown.")
