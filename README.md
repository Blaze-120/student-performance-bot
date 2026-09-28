# Student Performance Predictor

Predicts a student's final grade and failure risk from:
attendance, assignments, CATs, study hours, and previous grades.

Trains and compares three models (Linear Regression, Random Forest, XGBoost)
and keeps the best one by MAE.

## Project structure

```
student-performance-bot/
├── src/
│   ├── data.py       # data loading + synthetic data generator
│   ├── train.py      # trains all 3 models, saves the best one
│   └── predict.py    # loads model, makes predictions
├── app/
│   ├── bot.py         # interactive CLI bot
│   └── web.py         # FastAPI web version with a simple form UI
├── models/            # trained model saved here (best_model.joblib)
├── data/              # put your real CSV here if you have one
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model

Using built-in synthetic data (works out of the box, no dataset needed):

```bash
python -m src.train
```

Using your own real data — put a CSV in `data/` with these exact columns:
`attendance, assignments, cats, study_hours, previous_grades, final_grade`

```bash
python -m src.train --data data/students.csv
```

This prints MAE/R² for all 3 models and saves the best one to
`models/best_model.joblib`.

## 2. Run the bot

CLI version:

```bash
python -m app.bot
```

Web version:

```bash
uvicorn app.web:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

## Notes

- `FAIL_THRESHOLD` in `src/predict.py` (default 40) controls the risk cutoff.
- Retrain any time your dataset changes by rerunning `python -m src.train`.
