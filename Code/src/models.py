import joblib
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_dir


def build_logistic_xg_pipeline(random_state: int = 42, C: float = 1.0) -> Pipeline:
    """Return an unfitted sklearn Pipeline: StandardScaler → LogisticRegression (L2, lbfgs).

    Hyperparameters are locked for the baseline model:
      penalty="l2", C=1.0, solver="lbfgs", max_iter=1000.
    random_state controls LogisticRegression initialisation for reproducibility.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )),
    ])


def train_logistic_xg(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Fit pipeline on training data and return the fitted pipeline."""
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_xg(pipeline: Pipeline, X) -> np.ndarray:
    """Return 1-D array of xG probabilities (predict_proba[:, 1])."""
    return pipeline.predict_proba(X)[:, 1]


def save_model(pipeline: Pipeline, path) -> None:
    """Persist a fitted pipeline to disk with joblib.dump.

    Creates parent directories if they do not already exist.
    path should end in .joblib by convention.
    """
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(pipeline, path)


def load_model(path) -> Pipeline:
    """Load a fitted pipeline from disk with joblib.load.

    Raises FileNotFoundError with a clear message if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
