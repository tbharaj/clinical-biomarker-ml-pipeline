import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)


def load_dataset():
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    # Original sklearn coding:
    # 0 = malignant, 1 = benign
    # Recode so the clinically important class is positive:
    # malignant = 1, benign = 0
    y = (y == 0).astype(int)

    df = X.copy()
    df["malignant"] = y
    df.to_csv(DATA_DIR / "breast_cancer_biomarker_dataset.csv", index=False)

    return X, y, dataset.feature_names


def build_models():
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_auc = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "cv_roc_auc_mean": float(np.mean(cv_auc)),
        "cv_roc_auc_sd": float(np.std(cv_auc)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
    }

    cm = confusion_matrix(y_test, y_pred)

    pd.DataFrame(
        cm,
        index=["Actual benign", "Actual malignant"],
        columns=["Predicted benign", "Predicted malignant"],
    ).to_csv(RESULTS_DIR / f"{name}_confusion_matrix.csv")

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC curve: {name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}_roc_curve.png", dpi=300)
    plt.close()

    joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    return metrics, model


def save_feature_importance(model, X_test, y_test, feature_names, model_name):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc",
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_sd": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(
        RESULTS_DIR / f"{model_name}_permutation_importance.csv",
        index=False,
    )

    top_features = importance_df.head(12).sort_values("importance_mean")

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features["importance_mean"])
    plt.xlabel("Mean decrease in ROC AUC")
    plt.title(f"Top biomarker features: {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_feature_importance.png", dpi=300)
    plt.close()


def main():
    X, y, feature_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    models = build_models()

    all_metrics = []
    fitted_models = {}

    for name, model in models.items():
        metrics, fitted_model = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        all_metrics.append(metrics)
        fitted_models[name] = fitted_model

    metrics_df = pd.DataFrame(all_metrics).sort_values(
        "test_roc_auc",
        ascending=False,
    )

    metrics_df.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    save_feature_importance(
        best_model,
        X_test,
        y_test,
        feature_names,
        best_model_name,
    )

    metadata = {
        "dataset": "sklearn breast cancer diagnostic dataset",
        "task": "binary classification of malignant vs benign tumour samples",
        "positive_class": "malignant",
        "best_model": best_model_name,
        "important_boundary": (
            "This is an educational biomarker ML pipeline using a public benchmark dataset. "
            "It is not a clinical diagnostic tool."
        ),
    }

    with open(RESULTS_DIR / "project_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nPipeline complete.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()