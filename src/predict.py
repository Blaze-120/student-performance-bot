"""Load a trained model and make predictions."""

import joblib
import pandas as pd

FAIL_THRESHOLD = 40


def load_model(path: str = "models/best_model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_student(
    model,
    features: list[str],
    attendance: float,
    assignments: float,
    cats: float,
    study_hours: float,
    previous_grades: float,
    fail_threshold: float = FAIL_THRESHOLD,
):
    x = pd.DataFrame([{
        "attendance": attendance,
        "assignments": assignments,
        "cats": cats,
        "study_hours": study_hours,
        "previous_grades": previous_grades,
    }])[features]

    grade = float(model.predict(x)[0])
    grade = max(0.0, min(100.0, grade))
    at_risk = grade < fail_threshold
    return {"predicted_grade": round(grade, 1), "at_risk": at_risk}
