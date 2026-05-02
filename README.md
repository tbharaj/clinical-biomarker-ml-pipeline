\# Clinical Biomarker ML Pipeline



A reproducible Python machine-learning project for binary biomarker classification using a public breast cancer diagnostic dataset.



\## Project summary



This project builds a complete machine-learning workflow for classifying malignant versus benign tumour samples using biomarker-style diagnostic features.



The pipeline includes data loading, train-test splitting, cross-validation, model comparison, ROC-AUC evaluation, confusion matrices, feature-importance analysis and model export.



The project is designed to demonstrate practical Python, scikit-learn, biomedical data science and reproducible machine-learning workflow skills.



\## Important interpretation boundary



This is an educational machine-learning pipeline using a public benchmark dataset. It is not a clinical diagnostic tool and should not be interpreted as medical advice or validated clinical software.



The purpose is to demonstrate reproducible biomedical ML workflow design, model evaluation and cautious interpretation.



\## Dataset



The project uses the breast cancer diagnostic dataset available through `sklearn.datasets.load\_breast\_cancer`.



The original dataset labels are recoded so that:



\- `1` = malignant

\- `0` = benign



This makes malignant disease the positive class for model evaluation.



\## Models compared



Two classification models are compared:



\- Logistic regression with standardisation and class balancing

\- Random forest classifier with class balancing



\## Results summary



| Model | CV ROC-AUC mean | Test ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 |

|---|---:|---:|---:|---:|---:|---:|

| Random forest | 0.9866 | 0.9973 | 0.9650 | 0.9800 | 0.9245 | 0.9515 |

| Logistic regression | 0.9906 | 0.9950 | 0.9720 | 0.9804 | 0.9434 | 0.9615 |



The models achieved strong classification performance on the held-out test set. Logistic regression had slightly higher test accuracy, recall and F1 score, while random forest had slightly higher ROC-AUC.



Because this is a clean benchmark dataset, these results should be interpreted as evidence of a working ML workflow rather than evidence of real-world clinical readiness.



\## Main outputs



| Output | Location |

|---|---|

| Model performance summary | `results/model\_performance\_summary.csv` |

| Logistic regression confusion matrix | `results/logistic\_regression\_confusion\_matrix.csv` |

| Random forest confusion matrix | `results/random\_forest\_confusion\_matrix.csv` |

| Feature importance table | `results/random\_forest\_permutation\_importance.csv` |

| ROC curve plots | `figures/` |

| Feature importance plot | `figures/random\_forest\_feature\_importance.png` |

| Saved models | `models/` |



\## Example figures



\### Random forest ROC curve



!\[Random forest ROC curve](figures/random\_forest\_roc\_curve.png)



\### Logistic regression ROC curve



!\[Logistic regression ROC curve](figures/logistic\_regression\_roc\_curve.png)



\### Feature importance



!\[Feature importance](figures/random\_forest\_feature\_importance.png)



\## Skills demonstrated



\- Python programming

\- Biomedical machine learning

\- scikit-learn pipelines

\- Train-test splitting

\- Stratified cross-validation

\- ROC-AUC evaluation

\- Confusion matrix interpretation

\- Precision, recall and F1 score interpretation

\- Feature importance analysis

\- Model export using `joblib`

\- Reproducible project organisation

\- Scientific caution around clinical ML claims



\## Repository structure



## Repository structure

```text
data/       Processed dataset export
figures/    ROC curves and feature-importance plots
models/     Saved trained models
results/    Model metrics, metadata and confusion matrices
scripts/    Python analysis pipeline