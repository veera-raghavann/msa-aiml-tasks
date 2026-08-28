# Task 4 — Messy Dataset

A second-year-friendly ML workflow for the MLSA SRM technical recruitment dataset.

## Goal

Predict whether an applicant completed the recruitment task (`completed_task`). The exercise focuses on understanding messy data and making defensible preprocessing choices rather than building an overengineered model.

## Workflow

1. Inspect missing values, categories, duplicates, and numeric ranges.
2. Normalize inconsistent categorical representations.
3. Treat logically impossible numeric values as missing.
4. Remove exact duplicate rows only.
5. Flag unusual numeric values with IQR; retain unusual-but-plausible observations.
6. Split the data before fitting preprocessing steps.
7. Impute missing values and one-hot encode categorical features inside a scikit-learn pipeline.
8. Compare Logistic Regression and Random Forest.
9. Evaluate with accuracy, precision, recall, F1, confusion matrix, and misclassified rows.
10. Use 5-fold stratified cross-validation in `model_selection.py` for a fairer model comparison before looking at the held-out test set.

## Important choices

- `applicant_id` and `name` are identifiers, so they are not model features.
- Negative preparation hours and negative days since signup are impossible in context, so they become missing values.
- Quiz scores outside 0–100 become missing.
- Missing values are not used as a reason to discard applicants.
- Large preparation-hour values are investigated, not blindly deleted: unusual does not automatically mean invalid.
- The test set is kept separate from preprocessing and model selection.

## Files

- `messy_dataset_baseline.ipynb` — complete walkthrough with audit, cleaning, EDA, two models, evaluation, and error analysis.
- `model_selection.py` — small second-year extension using 5-fold stratified cross-validation.
- `requirements.txt` — packages needed to run the task.

## Dataset

Starter dataset: `MLSA-SRM/recruit-task-messy-dataset` — `recruitment_engagement.csv`.

The notebook loads the public CSV directly, so the dataset itself does not need to be committed to this repository.

## Running

Open the notebook in Google Colab or Jupyter and run it from top to bottom. For the cross-validation extension:

```bash
pip install -r requirements.txt
python model_selection.py
```

## Limitation

This is a small dataset, so performance estimates can vary with the split. The result is a baseline for learning and comparison, not a production recruitment model.
