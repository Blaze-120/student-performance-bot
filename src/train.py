"""Train Linear Regression, Random Forest, and XGBoost models and pick the best."""

import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.data import load_data, FEATURES, TARGET

MODEL_OUT_PATH = "models/best_model.joblib"


def train_models(df, verbose: bool = True):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "mae": mae, "r2": r2}
        if verbose:
            print(f"{name:20s}  MAE: {mae:6.2f}  R2: {r2:.3f}")

    best_name = min(results, key=lambda k: results[k]["mae"])
    if verbose:
        print(f"\nBest model: {best_name}")
    return results, best_name


def main():
    parser = argparse.ArgumentParser(description="Train student performance models")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to CSV with columns: attendance, assignments, cats, "
             "study_hours, previous_grades, final_grade. Omit to use synthetic data."
    )
    parser.add_argument(
        "--out", type=str, default=MODEL_OUT_PATH,
        help="Where to save the trained best model"
    )
    args = parser.parse_args()

    df = load_data(args.data)
    results, best_name = train_models(df)
    best_model = results[best_name]["model"]

    joblib.dump({"model": best_model, "features": FEATURES, "name": best_name}, args.out)
    print(f"Saved best model ({best_name}) to {args.out}")


if __name__ == "__main__":
    main()
