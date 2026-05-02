from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data

    # sklearn coding: 0 = malignant, 1 = benign.
    # Recode so malignant is the positive class.
    y = (dataset.target == 0).astype(int)

    return X, y


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


def calculate_threshold_metrics(y_true, y_probability, threshold):
    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": threshold,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

    rows = []

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        for threshold in thresholds:
            row = calculate_threshold_metrics(y_test, probabilities, threshold)
            row["model"] = model_name
            rows.append(row)

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)

    logistic_df = threshold_df[threshold_df["model"] == "logistic_regression"]

    plt.figure(figsize=(8, 6))
    plt.plot(
        logistic_df["threshold"],
        logistic_df["sensitivity_recall"],
        marker="o",
        label="Sensitivity / recall",
    )
    plt.plot(
        logistic_df["threshold"],
        logistic_df["specificity"],
        marker="o",
        label="Specificity",
    )
    plt.xlabel("Classification threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold trade-off: Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logistic_regression_threshold_tradeoff.png", dpi=300)
    plt.close()

    clinically_sensitive = logistic_df[logistic_df["false_negatives"] <= 2]

    if not clinically_sensitive.empty:
        selected = clinically_sensitive.sort_values(
            ["false_negatives", "false_positives"],
            ascending=[True, True],
        ).iloc[0]
    else:
        selected = logistic_df.sort_values(
            ["sensitivity_recall", "specificity"],
            ascending=[False, False],
        ).iloc[0]

    selected.to_frame().T.to_csv(
        RESULTS_DIR / "selected_threshold_summary.csv",
        index=False,
    )

    print("\nThreshold analysis complete.")
    print("Selected clinically cautious logistic regression threshold:")
    print(selected.to_string())


if __name__ == "__main__":
    main()
