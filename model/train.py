import os
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

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


def temporal_split(df, date_col="last_commit_date", train_frac=0.70, val_frac=0.15):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    return df.index[:train_end], df.index[train_end:val_end], df.index[val_end:]


def make_objective(X_train, y_train, X_val, y_val, scale_pos_weight):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)
    return objective


def train(input_path: str = "data/features.csv"):
    print("=" * 60)
    print("Git Risk Analyzer — Model Training")
    print("=" * 60)

    df = pd.read_csv(input_path)
    df = df.dropna(subset=FEATURES + ["is_buggy"])

    if "last_commit_date" in df.columns:
        df["last_commit_date"] = pd.to_datetime(df["last_commit_date"])
        date_col = "last_commit_date"
    else:
        print("\n[WARNING] last_commit_date not found. Using random fallback.\n")
        np.random.seed(42)
        df["split_key"] = np.random.rand(len(df))
        date_col = "split_key"

    X = df[FEATURES]
    y = df["is_buggy"]

    print(f"\nTotal samples : {len(df):,}")
    print(f"Buggy         : {y.sum():,}")
    print(f"Clean         : {(y == 0).sum():,}")
    print(f"Bug rate      : {y.mean():.2%}")

    train_idx, val_idx, test_idx = temporal_split(df, date_col=date_col)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    print(f"\nTrain : {len(train_idx):,}")
    print(f"Val   : {len(val_idx):,}")
    print(f"Test  : {len(test_idx):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. Logistic Regression
    print("\n" + "=" * 60)
    print("1. Logistic Regression")
    print("=" * 60)
    lr = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_val_proba = lr.predict_proba(X_val_scaled)[:, 1]
    lr_val_pred = lr.predict(X_val_scaled)
    print(classification_report(y_val, lr_val_pred))
    print(f"ROC-AUC : {roc_auc_score(y_val, lr_val_proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_val, lr_val_proba):.4f}")
    print(f"Brier   : {brier_score_loss(y_val, lr_val_proba):.4f}")
    results["logistic_regression"] = {"model": lr, "uses_scaler": True, "validation_auc": roc_auc_score(y_val, lr_val_proba)}

    # 2. Random Forest
    print("\n" + "=" * 60)
    print("2. Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_val_proba)
    print(classification_report(y_val, rf_val_pred))
    print(f"ROC-AUC : {rf_auc:.4f}")
    print(f"PR-AUC  : {average_precision_score(y_val, rf_val_proba):.4f}")
    print(f"Brier   : {brier_score_loss(y_val, rf_val_proba):.4f}")
    print(f"F1      : {f1_score(y_val, rf_val_pred, average='weighted'):.4f}")
    print("\nTop 5 features:\n", pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(5).to_string())
    results["random_forest"] = {"model": rf, "uses_scaler": False, "validation_auc": rf_auc}

    # 3. XGBoost (Optuna)
    print("\n" + "=" * 60)
    print("3. XGBoost (Optuna Tuning)")
    print("=" * 60)
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    print(f"scale_pos_weight : {scale_pos_weight:.3f}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(X_train, y_train, X_val, y_val, scale_pos_weight), n_trials=30, show_progress_bar=True)

    print(f"\nBest trial : {study.best_trial.number}")
    print(f"Best ROC-AUC : {study.best_value:.4f}")
    print(f"Best params : {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({"scale_pos_weight": scale_pos_weight, "random_state": 42, "n_jobs": -1, "verbosity": 0, "eval_metric": "logloss"})
    xgb = XGBClassifier(**best_params)
    xgb.fit(X_train, y_train)

    xgb_val_pred = xgb.predict(X_val)
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_val_proba)
    print(classification_report(y_val, xgb_val_pred))
    print(f"ROC-AUC : {xgb_auc:.4f}")
    print(f"PR-AUC  : {average_precision_score(y_val, xgb_val_proba):.4f}")
    print(f"Brier   : {brier_score_loss(y_val, xgb_val_proba):.4f}")
    print(f"F1      : {f1_score(y_val, xgb_val_pred, average='weighted'):.4f}")
    results["xgboost"] = {"model": xgb, "uses_scaler": False, "validation_auc": xgb_auc}

    # Model Selection
    print("\n" + "=" * 60)
    print("Model Selection")
    print("=" * 60)
    best_name = max(results, key=lambda k: results[k]["validation_auc"])
    best_raw_model = results[best_name]["model"]
    uses_scaler = results[best_name]["uses_scaler"]
    print(f"Best model : {best_name} (val ROC-AUC: {results[best_name]['validation_auc']:.4f})")

    # Calibration — sigmoid to avoid isotonic saturation
    print("\nCalibrating probabilities (sigmoid)...")
    if uses_scaler:
        calibrator = CalibratedClassifierCV(estimator=best_raw_model, method="sigmoid", cv=5)
        calibrator.fit(X_train_scaled, y_train)
        raw_test_proba = best_raw_model.predict_proba(X_test_scaled)[:, 1]
        calibrated_test_proba = calibrator.predict_proba(X_test_scaled)[:, 1]
        test_pred = calibrator.predict(X_test_scaled)
    else:
        calibrator = CalibratedClassifierCV(estimator=best_raw_model, method="sigmoid", cv=5)
        calibrator.fit(X_train, y_train)
        raw_test_proba = best_raw_model.predict_proba(X_test)[:, 1]
        calibrated_test_proba = calibrator.predict_proba(X_test)[:, 1]
        test_pred = calibrator.predict(X_test)

    # Clip to prevent exact 0/1 saturation
    calibrated_test_proba = np.clip(calibrated_test_proba, 0.001, 0.999)

    # Diagnostics
    print(f"\nRaw proba range     : [{raw_test_proba.min():.4f}, {raw_test_proba.max():.4f}]")
    print(f"Calibrated range  : [{calibrated_test_proba.min():.4f}, {calibrated_test_proba.max():.4f}]")
    print(f"Mean calibrated     : {calibrated_test_proba.mean():.4f}")
    print(f"Exact 1.0 count     : {np.sum(calibrated_test_proba >= 0.999)}")

    # Final Evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    print(classification_report(y_test, test_pred))

    final_auc = roc_auc_score(y_test, calibrated_test_proba)
    final_pr = average_precision_score(y_test, calibrated_test_proba)
    final_brier = brier_score_loss(y_test, calibrated_test_proba)
    final_f1 = f1_score(y_test, test_pred, average="weighted")

    print(f"ROC-AUC : {final_auc:.4f}")
    print(f"PR-AUC  : {final_pr:.4f}")
    print(f"Brier   : {final_brier:.4f}")
    print(f"F1      : {final_f1:.4f}")

    # Save
    os.makedirs("model", exist_ok=True)
    bundle = {
        "model": calibrator,
        "raw_model": best_raw_model,
        "scaler": scaler if uses_scaler else None,
        "features": FEATURES,
        "best_model_name": best_name,
        "metrics": {"roc_auc": final_auc, "pr_auc": final_pr, "brier": final_brier, "weighted_f1": final_f1},
    }
    with open("model/saved_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved → model/saved_model.pkl")
    print(f"Model : {best_name} (sigmoid calibrated + clipped)")
    print("=" * 60)


if __name__ == "__main__":
    train()