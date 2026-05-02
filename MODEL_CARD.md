# Model Card: Clinical Biomarker ML Pipeline

## Model overview

This repository contains an educational biomedical machine-learning pipeline for classifying malignant versus benign tumour samples using the Breast Cancer Wisconsin Diagnostic dataset.

The project compares logistic regression and random forest classifiers, then extends the evaluation with threshold analysis and probability calibration.

## Intended use

This project is intended to demonstrate:

- biomedical machine-learning workflow design
- binary classification model evaluation
- ROC-AUC, precision, recall and F1 interpretation
- threshold trade-off analysis
- probability calibration
- cautious clinical interpretation
- reproducible Python project organisation

## Not intended use

This project is not intended for:

- clinical diagnosis
- treatment decision-making
- medical advice
- deployment in healthcare systems
- claims about real-world diagnostic safety

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset accessed through `sklearn.datasets.load_breast_cancer`.

The target variable is recoded so that:

- `1` = malignant
- `0` = benign

## Models

The project compares:

- logistic regression with standardisation and class balancing
- random forest classifier with class balancing
- calibrated random forest for probability calibration analysis

## Key evaluation metrics

The project reports:

- cross-validated ROC-AUC
- test ROC-AUC
- accuracy
- precision
- recall/sensitivity
- F1 score
- confusion matrices
- Brier score
- threshold-level false positives and false negatives

## Clinical interpretation caution

In clinical biomarker contexts, false negatives may be more harmful than false positives because missing malignant disease can delay diagnosis.

For that reason, threshold selection should not rely only on accuracy or ROC-AUC. Sensitivity, specificity, false negatives and calibration should also be considered.

## Limitations

This project uses a clean public benchmark dataset rather than raw hospital, clinical trial or external validation data.

The models have not been externally validated or prospectively tested.

Feature importance results are model-interpretation outputs and should not be treated as causal biological evidence.

High performance metrics partly reflect the structured nature of the benchmark dataset and should not be overgeneralised to real-world clinical data.
