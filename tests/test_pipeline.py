import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_recoding_contains_binary_target():
    dataset = load_breast_cancer(as_frame=True)
    y = (dataset.target == 0).astype(int)

    assert set(y.unique()) == {0, 1}
    assert y.sum() == int((dataset.target == 0).sum())


def test_main_pipeline_runs_and_creates_expected_outputs():
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr + result.stdout

    expected_files = [
        "results/model_performance_summary.csv",
        "results/logistic_regression_confusion_matrix.csv",
        "results/random_forest_confusion_matrix.csv",
        "figures/logistic_regression_roc_curve.png",
        "figures/random_forest_roc_curve.png",
        "models/logistic_regression.joblib",
        "models/random_forest.joblib",
    ]

    for file_path in expected_files:
        assert (PROJECT_ROOT / file_path).exists(), f"Missing {file_path}"


def test_model_performance_summary_has_expected_columns():
    summary_path = PROJECT_ROOT / "results/model_performance_summary.csv"

    if not summary_path.exists():
        subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=PROJECT_ROOT,
            check=True,
            timeout=120,
        )

    df = pd.read_csv(summary_path)

    expected_columns = {
        "model",
        "cv_roc_auc_mean",
        "cv_roc_auc_sd",
        "test_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    }

    assert expected_columns.issubset(df.columns)
    assert df["test_roc_auc"].between(0, 1).all()
