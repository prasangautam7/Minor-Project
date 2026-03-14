"""
Train Random Forest, XGBoost, and SVM models for sleep apnea detection.
Patient-level train/val/test split to prevent data leakage.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

from utils import PATIENT_IDS, MODEL_DIR, OUTPUT_DIR
from preprocess import preprocess_all_patients
from windowing import create_all_windows
from features import extract_all_features, FEATURE_NAMES


def patient_level_split(X, y, patient_ids, train_pids, val_pids, test_pids):
    """Split data by patient ID to prevent leakage."""
    train_mask = np.isin(patient_ids, train_pids)
    val_mask = np.isin(patient_ids, val_pids)
    test_mask = np.isin(patient_ids, test_pids)

    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_val": X[val_mask], "y_val": y[val_mask],
        "X_test": X[test_mask], "y_test": y[test_mask],
        "pids_train": patient_ids[train_mask],
        "pids_val": patient_ids[val_mask],
        "pids_test": patient_ids[test_mask],
    }


def get_split_ids():
    """
    Patient-level split: 35 train, 5 val, 10 test.
    Fixed split for reproducibility.
    """
    np.random.seed(42)
    ids = np.array(PATIENT_IDS)
    np.random.shuffle(ids)
    train_ids = list(ids[:35])
    val_ids = list(ids[35:40])
    test_ids = list(ids[40:])
    return train_ids, val_ids, test_ids


def evaluate_model(model, X, y, dataset_name=""):
    """Compute all metrics for a model on given data."""
    y_pred = model.predict(X)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y, y_prob)
        metrics["auc_pr"] = average_precision_score(y, y_prob)

    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()

    if dataset_name:
        print(f"\n{'='*50}")
        print(f"Results on {dataset_name}:")
        print(f"{'='*50}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if "auc_roc" in metrics:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
        print(f"  Confusion Matrix:\n  {metrics['confusion_matrix']}")
        print(classification_report(y, y_pred, target_names=["Normal", "Apnea"]))

    return metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [15, 25, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    evaluate_model(best_model, X_val, y_val, "Validation (RF)")
    return best_model


def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos = neg_count / max(pos_count, 1)

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [5, 8, 12],
        "learning_rate": [0.05, 0.1],
        "scale_pos_weight": [scale_pos],
    }
    xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    evaluate_model(best_model, X_val, y_val, "Validation (XGBoost)")
    return best_model


def train_svm(X_train, y_train, X_val, y_val, scaler):
    print("\n" + "="*60)
    print("Training SVM...")
    print("="*60)

    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"],
    }
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train_s, y_train)

    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    evaluate_model(best_model, X_val_s, y_val, "Validation (SVM)")
    return best_model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load and preprocess
    print("Step 1: Loading and preprocessing all patients...")
    patients = preprocess_all_patients()

    # Step 2: Create windows
    print("\nStep 2: Creating windows...")
    windows = create_all_windows(patients)
    print(f"\nTotal windows: {len(windows)}")
    print(f"Apnea windows: {sum(1 for w in windows if w['label']==1)}")
    print(f"Normal windows: {sum(1 for w in windows if w['label']==0)}")

    # Step 3: Extract features
    print("\nStep 3: Extracting features...")
    X, y, pids = extract_all_features(windows)
    pids = np.array(pids)
    print(f"Feature matrix shape: {X.shape}")

    # Step 4: Patient-level split
    print("\nStep 4: Splitting data (patient-level)...")
    train_ids, val_ids, test_ids = get_split_ids()
    print(f"Train patients ({len(train_ids)}): {train_ids}")
    print(f"Val patients ({len(val_ids)}):   {val_ids}")
    print(f"Test patients ({len(test_ids)}):  {test_ids}")

    split = patient_level_split(X, y, pids, train_ids, val_ids, test_ids)

    print(f"\nTrain: {len(split['y_train'])} samples "
          f"({split['y_train'].sum()} apnea, {(split['y_train']==0).sum()} normal)")
    print(f"Val:   {len(split['y_val'])} samples "
          f"({split['y_val'].sum()} apnea, {(split['y_val']==0).sum()} normal)")
    print(f"Test:  {len(split['y_test'])} samples "
          f"({split['y_test'].sum()} apnea, {(split['y_test']==0).sum()} normal)")

    # Scaler for SVM
    scaler = StandardScaler()
    scaler.fit(split["X_train"])

    # Step 5: Train models
    print("\nStep 5: Training models...")
    rf_model = train_random_forest(split["X_train"], split["y_train"],
                                   split["X_val"], split["y_val"])
    xgb_model = train_xgboost(split["X_train"], split["y_train"],
                              split["X_val"], split["y_val"])
    svm_model = train_svm(split["X_train"], split["y_train"],
                          split["X_val"], split["y_val"], scaler)

    # Step 6: Evaluate on test set
    print("\n" + "#"*60)
    print("FINAL TEST SET EVALUATION")
    print("#"*60)

    results = {}

    results["Random Forest"] = evaluate_model(
        rf_model, split["X_test"], split["y_test"], "Test (Random Forest)")

    results["XGBoost"] = evaluate_model(
        xgb_model, split["X_test"], split["y_test"], "Test (XGBoost)")

    X_test_s = scaler.transform(split["X_test"])
    results["SVM"] = evaluate_model(
        svm_model, X_test_s, split["y_test"], "Test (SVM)")

    # Save models
    joblib.dump(rf_model, MODEL_DIR / "random_forest.joblib")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost.joblib")
    joblib.dump(svm_model, MODEL_DIR / "svm.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    print(f"\nModels saved to {MODEL_DIR}")

    # Save results
    serializable = {}
    for name, metrics in results.items():
        serializable[name] = {k: v for k, v in metrics.items()}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR / 'results.json'}")

    # Save split info
    split_info = {
        "train_patients": train_ids,
        "val_patients": val_ids,
        "test_patients": test_ids,
        "feature_names": FEATURE_NAMES,
    }
    with open(OUTPUT_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Feature importance (RF)
    importances = rf_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print("\nRandom Forest Feature Importance (Top 10):")
    for name, imp in feat_imp[:10]:
        print(f"  {name:25s} {imp:.4f}")

    return results


if __name__ == "__main__":
    main()
