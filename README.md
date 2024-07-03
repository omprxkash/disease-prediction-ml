# Disease Prediction with Machine Learning

Heart disease risk prediction from clinical records using the Cleveland Heart Disease Dataset. Built as part of a data analysis course at VIT Chennai.

## The Problem

Heart disease doesn't come with a warning label. The goal here was to build a model that could flag high-risk patients from routine clinical measurements — before anything serious happens. The Cleveland dataset is small (303 records, 14 features) but well-studied, which makes it a good benchmark for comparing how different classifiers handle structured medical data.

## Dataset

**Cleveland Heart Disease Dataset**
- 303 patient records, 14 features, binary target (disease present / absent)
- Features include: age, sex, chest pain type (4 types), resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, max heart rate achieved, exercise-induced angina, ST depression (oldpeak), slope, major vessels coloured by fluoroscopy, thalassemia type
- Train/test split: 70/30 (212 training, 91 test samples)
- Preprocessing: outlier removal, missing value imputation, one-hot encoding for categorical features, StandardScaler for 5 numeric columns

## Approach

Ran five classifiers against each other to find what works best on this dataset:

| Model | Test Accuracy | AUC-ROC |
|---|---|---|
| **XGBoost** | **95%** | **0.99** |
| Logistic Regression | 86.81% | 0.9166 |
| Random Forest | ~79.5% CV | — |
| K-Nearest Neighbors (k=12) | — | — |
| Decision Tree | — | — |

XGBoost won clearly. The AUC of 0.99 means the model separates positive and negative cases almost perfectly — which matters more than raw accuracy on a medical classification task.

5-fold cross-validation mean score: 83%

## Repository Contents

| File | Description |
|---|---|
| `21BCE1950_21BCE5828_code.ipynb` | Full pipeline — EDA, preprocessing, model training, evaluation |
| `21BCE1950_21BCE5828_dataset.ipynb` | Dataset exploration and analysis |
| `21BCE1950_21BCE5828_DA1 (1).docx` | Data analysis assignment 1 report |
| `21BCE1950_21BCE5828_DA3_afterreview.pdf` | Final reviewed research report |

## Stack

Python · scikit-learn · XGBoost · pandas · matplotlib · Jupyter Notebook
