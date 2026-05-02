from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
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

    calibrated_random_forest = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        method="sigmoid",
        cv=5,
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "calibrated_random_forest": calibrated_random_forest,
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

    rows = []

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        prob_true, prob_pred = calibration_curve(
            y_test,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(prob_pred, prob_true, marker="o", label=model_name)

        rows.append(
            {
                "model": model_name,
                "brier_score": brier_score_loss(y_test, probabilities),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Probability calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300)
    plt.close()

    calibration_df = pd.DataFrame(rows).sort_values("brier_score")
    calibration_df.to_csv(RESULTS_DIR / "calibration_summary.csv", index=False)

    print("\nCalibration analysis complete.")
    print(calibration_df.to_string(index=False))


if __name__ == "__main__":
    main()
