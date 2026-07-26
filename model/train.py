import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

FEATURES = [
    "total_commits",
    "commits_last_30d",
    "commits_last_90d",
    "days_since_last_change",
    "unique_authors",
    "top_owner_pct",
    "avg_lines_added",
    "avg_lines_deleted",
    "max_lines_changed",
    "file_age_days",
    "avg_nloc",
]

def train(input_path: str = "data/features.csv"):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=FEATURES + ["is_buggy"])

    X = df[FEATURES]
    y = df["is_buggy"]

    print(f"Total samples: {len(df)}")
    print(f"Buggy: {y.sum()} | Clean: {(y == 0).sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # 1. logistic regression baseline
    print("=" * 50)
    print("1. Logistic Regression (baseline)")
    print("=" * 50)
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga"
    )
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    print(classification_report(y_test, lr_preds))
    results["logistic_regression"] = lr

    # 2. random forest
    print("=" * 50)
    print("2. Random Forest")
    print("=" * 50)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_report = classification_report(y_test, rf_preds, output_dict=True)
    print(classification_report(y_test, rf_preds))
    results["random_forest"] = rf

    # feature importance
    print("Top 5 features (Random Forest):")
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    print(importances.sort_values(ascending=False).head(5).to_string())
    print()

    # 3. xgboost
    print("=" * 50)
    print("3. XGBoost")
    print("=" * 50)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_report = classification_report(y_test, xgb_preds, output_dict=True)
    print(classification_report(y_test, xgb_preds))

    # pick best model between RF and XGBoost
    rf_f1  = rf_report["weighted avg"]["f1-score"]
    xgb_f1 = xgb_report["weighted avg"]["f1-score"]

    if xgb_f1 >= rf_f1:
        best_model = xgb
        best_name  = "XGBoost"
    else:
        best_model = rf
        best_name  = "Random Forest"

    print(f"\nBest model: {best_name} (F1: {max(rf_f1, xgb_f1):.4f})")

    # save best model + scaler
    os.makedirs("model", exist_ok=True)
    with open("model/saved_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "features": FEATURES}, f)

    print(f"Saved → model/saved_model.pkl")


if __name__ == "__main__":
    train()