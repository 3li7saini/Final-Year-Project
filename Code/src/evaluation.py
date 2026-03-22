import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def metric_summary(y_true, y_prob, split_name: str) -> dict:
    """Return a metrics dict for one split.

    Keys: split, n_shots, goal_rate, roc_auc, average_precision,
          log_loss, brier_score, mean_pred_xg.
    Matches the exact dict structure produced inline in notebook 04.
    """
    return {
        "split":             split_name,
        "n_shots":           int(len(y_true)),
        "goal_rate":         float(np.mean(y_true)),
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "log_loss":          float(log_loss(y_true, y_prob)),
        "brier_score":       float(brier_score_loss(y_true, y_prob)),
        "mean_pred_xg":      float(np.mean(y_prob)),
    }


def build_calibration_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Build a decile calibration table using pd.qcut.

    Bins y_pred into n_bins quantiles (duplicate edges dropped).
    Returns a DataFrame with columns:
      xg_bin, n_shots, mean_pred_xg, observed_goal_rate

    Column naming matches the notebook's inline logic exactly:
    uses "is_goal" / "xg_pred" internally so aggregation labels are identical.
    """
    df = pd.DataFrame({"is_goal": y_true, "xg_pred": y_pred})
    df["xg_bin"] = pd.qcut(df["xg_pred"], q=n_bins, duplicates="drop")
    return (
        df.groupby("xg_bin")
        .agg(
            n_shots=("is_goal", "count"),
            mean_pred_xg=("xg_pred", "mean"),
            observed_goal_rate=("is_goal", "mean"),
        )
        .reset_index()
    )


def extract_model_coefficients(pipeline, feature_names: list) -> pd.DataFrame:
    """Return a DataFrame of logistic-regression coefficients from a fitted pipeline.

    Expects pipeline to have named step "logreg" with a coef_ attribute.
    Asserts len(feature_names) == len(coef) to catch silent feature-list mismatches.
    Returns DataFrame with columns: feature, coefficient, odds_ratio,
    sorted descending by coefficient (most positive first).
    """
    coef = pipeline.named_steps["logreg"].coef_[0]
    assert len(feature_names) == len(coef), (
        f"feature_names length ({len(feature_names)}) ≠ coef length ({len(coef)})"
    )
    return pd.DataFrame({
        "feature":     feature_names,
        "coefficient": coef,
        "odds_ratio":  np.exp(coef),
    }).sort_values("coefficient", ascending=False)
