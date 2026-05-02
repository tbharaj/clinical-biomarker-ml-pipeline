# Clinical Biomarker ML Pipeline

A reproducible Python machine-learning project for binary biomarker classification using a public breast cancer diagnostic dataset.

## Project summary

This project builds a complete machine-learning workflow for classifying malignant versus benign tumour samples using biomarker-style diagnostic features.

The pipeline includes data loading, train-test splitting, stratified cross-validation, model comparison, ROC-AUC evaluation, confusion matrices, feature-importance analysis and model export.

The project demonstrates practical Python, scikit-learn, biomedical data science and reproducible machine-learning workflow skills.

## Key result

Both models achieved strong held-out test performance on the public breast cancer diagnostic dataset.

| Model | CV ROC-AUC mean | Test ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random forest | 0.9866 | 0.9973 | 0.9650 | 0.9800 | 0.9245 | 0.9515 |
| Logistic regression | 0.9906 | 0.9950 | 0.9720 | 0.9804 | 0.9434 | 0.9615 |

Logistic regression achieved slightly higher accuracy, recall and F1 score, while random forest achieved slightly higher ROC-AUC.

Because this is a clean benchmark dataset, these results should be interpreted as evidence of a working machine-learning workflow rather than evidence of real-world clinical readiness.

## Main result figures

### Random forest ROC curve

![Random forest ROC curve](figures/random_forest_roc_curve.png)

### Logistic regression ROC curve

![Logistic regression ROC curve](figures/logistic_regression_roc_curve.png)

### Random forest feature importance

![Random forest feature importance](figures/random_forest_feature_importance.png)

## Important interpretation boundary

This is an educational machine-learning pipeline using a public benchmark dataset. It is not a clinical diagnostic tool and should not be interpreted as medical advice or validated clinical software.

The purpose is to demonstrate reproducible biomedical ML workflow design, model evaluation and cautious interpretation.

## Dataset

The project uses the breast cancer diagnostic dataset available through `sklearn.datasets.load_breast_cancer`.

The original dataset labels are recoded so that:

- `1` = malignant
- `0` = benign

This makes malignant disease the positive class for model evaluation.

## Models compared

Two classification models are compared:

- Logistic regression with standardisation and class balancing
- Random forest classifier with class balancing

## Key outputs

| Output | Location |
|---|---|
| Model performance summary | `results/model_performance_summary.csv` |
| Logistic regression confusion matrix | `results/logistic_regression_confusion_matrix.csv` |
| Random forest confusion matrix | `results/random_forest_confusion_matrix.csv` |
| Feature importance table | `results/random_forest_permutation_importance.csv` |
| ROC curve plots | `figures/` |
| Feature importance plot | `figures/random_forest_feature_importance.png` |
| Saved models | `models/` |

## Skills demonstrated

- Python programming
- Biomedical machine learning
- scikit-learn model building
- Train-test splitting
- Stratified cross-validation
- ROC-AUC evaluation
- Confusion matrix interpretation
- Precision, recall and F1 score interpretation
- Feature importance analysis
- Model export using `joblib`
- Reproducible project organisation
- Scientific caution around clinical ML claims

## Repository structure

```text
data/       Processed dataset export
figures/    ROC curves and feature-importance plots
models/     Saved trained models
results/    Model metrics, metadata and confusion matrices
scripts/    Python analysis pipeline
```

## Reproducibility

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python scripts/run_pipeline.py
```

## Limitations

This project uses a clean public benchmark dataset rather than raw hospital, clinical trial or external validation data.

The model has not been externally validated, prospectively tested or assessed for deployment in real clinical settings.

The feature importance results are useful for model interpretation, but they should not be treated as causal biological evidence.

The high performance metrics partly reflect the structured nature of the benchmark dataset and should not be overgeneralised to messy real-world clinical data.

## Conclusion

This project demonstrates a complete biomedical machine-learning workflow using Python and scikit-learn.

It shows practical ability to build, evaluate, interpret and organise a reproducible ML pipeline while maintaining appropriate caution around clinical claims.