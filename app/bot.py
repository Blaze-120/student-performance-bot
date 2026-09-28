"""Interactive CLI bot for the Student Performance Predictor.

Run from project root:
    python -m app.bot
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import load_model, predict_student, FAIL_THRESHOLD

MODEL_PATH = "models/best_model.joblib"


def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number.")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}.")
        print("Train one first:  python -m src.train")
        return

    model, features = load_model(MODEL_PATH)

    print("=== Student Performance Predictor Bot ===")
    print(f"(Predicting final grade; risk = below {FAIL_THRESHOLD}%)\n")
    print("Type 'quit' at any prompt to exit.\n")

    while True:
        raw = input("Attendance (%) [or 'quit']: ")
        if raw.strip().lower() == "quit":
            break
        try:
            attendance = float(raw)
            assignments = ask_float("Avg assignment score (%): ")
            cats = ask_float("Avg CAT score (%): ")
            study_hours = ask_float("Study hours/week: ")
            previous_grades = ask_float("Previous grades avg (%): ")
        except ValueError:
            print("Please enter numbers.\n")
            continue

        result = predict_student(
            model, features, attendance, assignments, cats, study_hours, previous_grades
        )
        status = "AT RISK of failing" if result["at_risk"] else "On track"
        print(f"\nPredicted final grade: {result['predicted_grade']}%  ->  {status}\n")


if __name__ == "__main__":
    main()
