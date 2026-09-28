"""FastAPI web app for the Student Performance Predictor.

Run from project root:
    uvicorn app.web:app --reload
Then open http://127.0.0.1:8000
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.predict import load_model, predict_student

MODEL_PATH = "models/best_model.joblib"

app = FastAPI(title="Student Performance Predictor")

_model = None
_features = None


class StudentInput(BaseModel):
    attendance: float
    assignments: float
    cats: float
    study_hours: float
    previous_grades: float


@app.on_event("startup")
def startup():
    global _model, _features
    if os.path.exists(MODEL_PATH):
        _model, _features = load_model(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Student Performance Predictor</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; }
            h1 { font-size: 1.4rem; }
            label { display: block; margin-top: 12px; font-size: 0.9rem; }
            input { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
            button { margin-top: 20px; padding: 10px 16px; width: 100%; font-size: 1rem; cursor: pointer; }
            #result { margin-top: 20px; padding: 12px; border-radius: 6px; display: none; }
            .ok { background: #e6f4ea; color: #1e4620; }
            .risk { background: #fdecea; color: #5f2120; }
        </style>
    </head>
    <body>
        <h1>Student Performance Predictor</h1>
        <form id="form">
            <label>Attendance (%)<input type="number" id="attendance" required></label>
            <label>Avg assignment score (%)<input type="number" id="assignments" required></label>
            <label>Avg CAT score (%)<input type="number" id="cats" required></label>
            <label>Study hours/week<input type="number" id="study_hours" required></label>
            <label>Previous grades avg (%)<input type="number" id="previous_grades" required></label>
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>
        <script>
            document.getElementById("form").addEventListener("submit", async (e) => {
                e.preventDefault();
                const payload = {
                    attendance: parseFloat(document.getElementById("attendance").value),
                    assignments: parseFloat(document.getElementById("assignments").value),
                    cats: parseFloat(document.getElementById("cats").value),
                    study_hours: parseFloat(document.getElementById("study_hours").value),
                    previous_grades: parseFloat(document.getElementById("previous_grades").value),
                };
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                const el = document.getElementById("result");
                el.style.display = "block";
                if (data.error) {
                    el.className = "risk";
                    el.innerText = data.error;
                    return;
                }
                el.className = data.at_risk ? "risk" : "ok";
                el.innerText = `Predicted final grade: ${data.predicted_grade}%  ->  ${data.at_risk ? "AT RISK of failing" : "On track"}`;
            });
        </script>
    </body>
    </html>
    """


@app.post("/predict")
def predict(student: StudentInput):
    if _model is None:
        return {"error": "No trained model found. Run: python -m src.train"}
    result = predict_student(
        _model, _features,
        student.attendance, student.assignments, student.cats,
        student.study_hours, student.previous_grades,
    )
    return result
