# Task 1 — Messy Dataset, Baseline Model

**MLSA SRM Technical Recruitment — AI/ML | Second Year**

A reasoning-first ML workflow for the deliberately messy `recruitment_engagement.csv` dataset. The goal is to understand the data, make defensible cleaning choices, compare two simple models, and inspect errors — not to build an overengineered system.

## Goal

Predict whether an applicant completed the recruitment task (`completed_task`).

## What was messy?

The raw data contains missing values, inconsistent categorical representations, impossible numeric values, duplicate records, and unusual numerical observations. Examples include mixed casing such as `Technical`/`technical`/`TECHNICAL`, year values such as `First`/`1`/`1st Year`, experience values such as `Yes`/`yes`/`Y`, negative preparation hours, negative days since signup, and unusually large preparation-hour values.

## Cleaning decisions

- Remove exact duplicate rows only.
- Normalize equivalent categorical labels instead of treating them as separate categories.
- Convert numeric fields explicitly with `pd.to_numeric(..., errors='coerce')`.
- Treat negative preparation hours and negative days since signup as missing because they are logically impossible in context.
- Treat quiz scores outside 0–100 as missing.
- Do not discard applicants simply because values are missing; impute inside the scikit-learn pipeline.
- Use IQR to flag numerical outliers, but retain unusual-but-plausible observations. An outlier is not automatically an error.
- Exclude `applicant_id` and `name` from model features because they identify applicants rather than describe engagement.

## Models

1. **Logistic Regression** — simple and interpretable linear baseline.
2. **Random Forest** — modest nonlinear baseline capable of capturing interactions.

Both use the same preprocessing and held-out split. The notebook reports accuracy, precision, recall, F1, confusion matrix, and misclassified rows.

## Second-year extension

`model_selection.py` performs 5-fold stratified cross-validation on the training data before evaluating the selected model once on the held-out test set. This is a fairer way to choose between the two approaches without using the test set for model selection.

### What I would ship

If Logistic Regression is close to Random Forest under cross-validation, I would ship Logistic Regression because it is simpler and easier to explain for this small dataset. If Random Forest is clearly better, I would choose it and document the trade-off.

## Files

- `messy_dataset_baseline.ipynb` — full audit, cleaning, EDA, model comparison, and error analysis.
- `model_selection.py` — second-year cross-validation extension.
- `requirements.txt` — dependencies.

## Run

Open the notebook in Google Colab/Jupyter and run it top to bottom. For the extension:

```bash
pip install -r requirements.txt
python model_selection.py
```

## Limitations

The dataset is small, so performance estimates can vary with the split. This is a baseline for learning and comparison, not a production recruitment model. A larger dataset would justify repeated cross-validation, calibration checks, and monitoring.

## Dataset source

MLSA-SRM starter repository: https://github.com/MLSA-SRM/recruit-task-messy-dataset
