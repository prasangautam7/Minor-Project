"""
Evaluation and visualization: generate plots and detailed metrics.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from pathlib import Path

from utils import MODEL_DIR, OUTPUT_DIR, PATIENT_IDS
from preprocess import preprocess_all_patients
from windowing import create_all_windows
from features import extract_all_features, FEATURE_NAMES
from train import patient_level_split, get_split_ids


def plot_roc_curves(models_dict, X_test, y_test, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, (model, needs_scaling, scaler) in models_dict.items():
        X = scaler.transform(X_test) if needs_scaling else X_test
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Sleep Apnea Detection", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_pr_curves(models_dict, X_test, y_test, save_path):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, (model, needs_scaling, scaler) in models_dict.items():
        X = scaler.transform(X_test) if needs_scaling else X_test
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.decision_function(X)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        auc_val = auc(rec, prec)
        plt.plot(rec, prec, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)

    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({baseline:.2f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves - Sleep Apnea Detection", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PR curves saved to {save_path}")


def plot_confusion_matrices(models_dict, X_test, y_test, save_path):
    """Plot confusion matrices for all models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, (model, needs_scaling, scaler)) in enumerate(models_dict.items()):
        X = scaler.transform(X_test) if needs_scaling else X_test
        y_pred = model.predict(X)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Apnea"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(f"{name}", fontsize=13)

    plt.suptitle("Confusion Matrices - Test Set", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_feature_importance(model, save_path):
    """Plot feature importance from Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(FEATURE_NAMES)),
             importances[indices[::-1]],
             align="center", color="steelblue")
    plt.yticks(range(len(FEATURE_NAMES)),
               [FEATURE_NAMES[i] for i in indices[::-1]], fontsize=10)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title("Random Forest - Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def plot_model_comparison(results, save_path):
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    model_names = list(results.keys())

    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        vals = [results[model_name].get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=model_name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison - Test Set Metrics", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Model comparison saved to {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    rf_model = joblib.load(MODEL_DIR / "random_forest.joblib")
    xgb_model = joblib.load(MODEL_DIR / "xgboost.joblib")
    svm_model = joblib.load(MODEL_DIR / "svm.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    # Rebuild test data
    print("Loading data for evaluation...")
    patients = preprocess_all_patients()
    windows = create_all_windows(patients)
    X, y, pids = extract_all_features(windows)
    pids = np.array(pids)

    train_ids, val_ids, test_ids = get_split_ids()
    split = patient_level_split(X, y, pids, train_ids, val_ids, test_ids)

    models_dict = {
        "Random Forest": (rf_model, False, None),
        "XGBoost": (xgb_model, False, None),
        "SVM": (svm_model, True, scaler),
    }

    # Load results
    with open(OUTPUT_DIR / "results.json") as f:
        results = json.load(f)

    # Generate plots
    plot_roc_curves(models_dict, split["X_test"], split["y_test"],
                    OUTPUT_DIR / "roc_curves.png")
    plot_pr_curves(models_dict, split["X_test"], split["y_test"],
                   OUTPUT_DIR / "pr_curves.png")
    plot_confusion_matrices(models_dict, split["X_test"], split["y_test"],
                            OUTPUT_DIR / "confusion_matrices.png")
    plot_feature_importance(rf_model, OUTPUT_DIR / "feature_importance.png")
    plot_model_comparison(results, OUTPUT_DIR / "model_comparison.png")

    print("\nAll evaluation plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
