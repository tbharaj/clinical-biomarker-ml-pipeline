"""
Generate SHAP feature-importance outputs for the breast cancer classification pipeline.

This script uses the Breast Cancer Wisconsin dataset loaded directly from
sklearn.datasets.load_breast_cancer(), fits a scaled logistic-regression model,
and saves:
- figures/shap_summary_plot.png
- figures/shap_top_features.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Fit a simple interpretable model and save SHAP outputs."""
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    print(f"Logistic regression ROC-AUC: {auc:.3f}")

    masker = shap.maskers.Independent(X_train_scaled)
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer(X_test_scaled)

    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=data.feature_names,
        show=False,
        max_display=12,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_plot.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = (
        pd.DataFrame({
            "feature": data.feature_names,
            "mean_abs_shap": mean_abs_shap,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
    )
    top_features.to_csv(figures_dir / "shap_top_features.csv", index=False)

    print("Saved figures/shap_summary_plot.png")
    print("Saved figures/shap_top_features.csv")


if __name__ == "__main__":
    main()
