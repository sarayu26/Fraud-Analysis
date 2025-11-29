# Fraud Detection in Mobile Money Transactions

This project builds a machine learning pipeline to detect fraudulent mobile money transactions using real transaction data. It covers data understanding, feature engineering, handling class imbalance with SMOTE, training multiple models, and estimating the financial impact of deploying the best model.[1][2][3]

## Project Overview

- Goal: Accurately identify fraudulent transactions in near real time.
- Dataset: Mobile money transaction log with:
  - step (time in hours)
  - type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
  - amount (transaction value)
  - nameOrig, nameDest (sender/receiver IDs)
  - oldbalanceOrg, newbalanceOrig
  - oldbalanceDest, newbalanceDest
  - isFraud (0 = legitimate, 1 = fraud)
- Main file: fraud_detection_script.py (end‑to‑end pipeline).

## Features & Methods
- Data audit:
  - Checks info, missing values, duplicates, class and type distribution.
- Feature engineering:
  - orig_bal_diff = oldbalanceOrg − newbalanceOrig
  - dest_bal_diff = oldbalanceDest − newbalanceDest
  - orig_ratio = amount / (oldbalanceOrg + 1)
  - dest_ratio = amount / (oldbalanceDest + 1)
  - One‑hot encoding of transaction type.
- Scaling:
  - StandardScaler applied to numeric features for stable training of models like Logistic Regression.
- Class imbalance handling:
  - SMOTE used on the training set to balance fraud vs non‑fraud before model training.

## Models

Trained and evaluated on an 80/20 stratified train–test split:

- Logistic Regression (baseline)
- Random Forest (200 trees)
- Gradient Boosting (200 estimators)
- LightGBM (200 estimators)

For each model, the script computes:

- Classification report
- Confusion matrix
- Precision, Recall, F1‑score
- ROC‑AUC
- ROC curves
- Feature importance plots for tree‑based models

## Financial Impact Analysis

Using the best model (LightGBM), the script estimates:

- Total fraud amount correctly detected (true positives)
- Fraud amount missed (false negatives)
- Operational cost from false positives (can be mapped to investigation cost)
- Total amount “saved” by the model on the test set

This connects model performance to real business value.

## File Structure

- fraud_detection_script.py – main ML pipeline (data loading → preprocessing → modelling → evaluation → financial impact).
- Fraud_Analysis_Dataset.xlsx – source dataset used by the script.
- description.txt – data dictionary explaining each column and transaction type.
- processed_fraud_data.csv – processed/exported version of the dataset (optional artefact).
- Capstone-PPT.pptx / fraud_detection.ipynb (optional, if you add) – presentation and notebook version of the analysis.

## How to Run

1. Clone the repo:
   - git clone <your_repo_url>
   - cd <your_repo_name>
2. Install dependencies (example):
   - pip install -r requirements.txt
3. Place Fraud_Analysis_Dataset.xlsx in the expected data folder or update the path in fraud_detection_script.py.
4. Run:
   - python fraud_detection_script.py

The script will print metrics to the console and display ROC curves and feature importance charts.

## Possible Extensions

- Hyperparameter tuning (GridSearchCV/Optuna).
- Threshold tuning based on business cost of false positives vs false negatives.
- Adding time, geographic, or behavioural features for better fraud patterns.
- Packaging as an API for real‑time scoring in production.
