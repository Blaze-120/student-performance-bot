"""Data loading and synthetic data generation."""

import numpy as np
import pandas as pd

FEATURES = ["attendance", "assignments", "cats", "study_hours", "previous_grades"]
TARGET = "final_grade"


def generate_synthetic_data(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic student dataset for demo/testing purposes."""
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(40, 100, n)
    assignments = rng.uniform(0, 100, n)
    cats = rng.uniform(0, 100, n)
    study_hours = rng.uniform(0, 25, n)
    previous_grades = rng.uniform(20, 100, n)

    noise = rng.normal(0, 5, n)
    final_grade = (
        0.25 * attendance
        + 0.20 * assignments
        + 0.25 * cats
        + 0.10 * study_hours * 4
        + 0.20 * previous_grades
        + noise
    )
    final_grade = np.clip(final_grade, 0, 100)

    return pd.DataFrame({
        "attendance": attendance,
        "assignments": assignments,
        "cats": cats,
        "study_hours": study_hours,
        "previous_grades": previous_grades,
        "final_grade": final_grade,
    })


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load real data from csv_path if given, otherwise generate synthetic data.

    Expected CSV columns: attendance, assignments, cats, study_hours,
    previous_grades, final_grade
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = set(FEATURES + [TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        return df
    return generate_synthetic_data()
