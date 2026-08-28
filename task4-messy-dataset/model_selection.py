import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

DATA_URL = "https://raw.githubusercontent.com/MLSA-SRM/recruit-task-messy-dataset/main/recruitment_engagement.csv"


def clean_data(df):
    df = df.drop_duplicates().copy()
    df["domain"] = df["domain"].astype("string").str.strip().str.lower().str.title()
    df["subdomain"] = df["subdomain"].astype("string").str.strip().str.lower().replace({"ai/ml": "AI/ML", "web dev": "Web Dev"}).str.title()
    df["subdomain"] = df["subdomain"].replace({"Ai/Ml": "AI/ML", "Pr": "PR"})
    df["year"] = df["year"].astype("string").str.strip().str.lower().map({"first": "1st Year", "1": "1st Year", "1st year": "1st Year", "second": "2nd Year", "2": "2nd Year", "2nd year": "2nd Year"})
    df["signup_source"] = df["signup_source"].astype("string").str.strip().str.title()
    df["prior_experience"] = df["prior_experience"].astype("string").str.strip().str.lower().map({"yes": "Yes", "y": "Yes", "no": "No", "n": "No"})
    for col in ["prep_hours_last_week", "quiz_score", "days_since_signup"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["prep_hours_last_week"] < 0, "prep_hours_last_week"] = np.nan
    df.loc[df["days_since_signup"] < 0, "days_since_signup"] = np.nan
    df.loc[~df["quiz_score"].between(0, 100), "quiz_score"] = np.nan
    df["completed_task"] = df["completed_task"].astype("string").str.strip().str.title()
    return df


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


df = clean_data(pd.read_csv(DATA_URL))
X = df.drop(columns=["completed_task", "applicant_id", "name"])
y = df["completed_task"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "Logistic Regression": Pipeline([("preprocess", make_preprocessor(X_train)), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
    "Random Forest": Pipeline([("preprocess", make_preprocessor(X_train)), ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    rows.append({"Model": name, "CV F1 mean": scores.mean(), "CV F1 std": scores.std()})

comparison = pd.DataFrame(rows).sort_values("CV F1 mean", ascending=False)
print("5-fold CV on training data")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

selected_name = comparison.iloc[0]["Model"]
selected = models[selected_name]
selected.fit(X_train, y_train)
pred = selected.predict(X_test)
print(f"\nFinal held-out test evaluation: {selected_name}")
print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred, pos_label='Yes'):.3f}")
print(f"Recall   : {recall_score(y_test, pred, pos_label='Yes'):.3f}")
print(f"F1       : {f1_score(y_test, pred, pos_label='Yes'):.3f}")
print("\nClassification report:\n")
print(classification_report(y_test, pred))

errors = X_test.copy()
errors["actual"] = y_test
errors["predicted"] = pred
print(f"Misclassified test rows: {(errors.actual != errors.predicted).sum()}")
print(errors[errors.actual != errors.predicted].to_string(index=False))
