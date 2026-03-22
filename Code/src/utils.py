import numpy as np
import random
from pathlib import Path


def ensure_dir(path):
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator, denominator, fill=0.0):
    """Return numerator / denominator, or fill when denominator is zero."""
    if denominator == 0:
        return fill
    return numerator / denominator


def assert_columns(df, cols, label="DataFrame"):
    """Raise AssertionError if any column in cols is missing from df."""
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"{label} is missing columns: {missing}"


def assert_binary(df, col):
    """Raise AssertionError if col contains values other than 0, 1, True, or False."""
    unique = set(df[col].dropna().unique())
    assert unique <= {0, 1, True, False}, (
        f"Column '{col}' contains non-binary values: {unique}"
    )


def set_random_seed(seed):
    """Set numpy and stdlib random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
