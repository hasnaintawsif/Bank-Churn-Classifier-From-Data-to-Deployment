# Bank-Churn-Classifier-From-Data-to-Deployment
This project aims to build a robust machine learning model to predict customer attrition (churn) for a bank based on historical customer data. The goal is to identify customers likely to leave the bank, allowing the organization to take proactive retention actions.

Problem Statement:
Customer churn prediction is a critical task in the banking industry. Early identification of customers likely to leave can help in formulating effective retention strategies. The goal of this project is to build a high-performing model to predict whether a customer will churn based on their demographic and transactional attributes.

Workflow Summary:
1. Data Ingestion
Dataset: BankChurners.csv

Unnecessary features such as CLIENTNUM and internal model outputs were removed.

Infrequent categorical levels (e.g., Platinum, Gold) were grouped under Others for simplification.

2. Data Preprocessing
Binary categorical columns (e.g., Gender, Attrition_Flag) encoded using LabelBinarizer.

Multiclass categorical columns encoded with one-hot encoding (get_dummies with drop_first=True).

Feature scaling applied using StandardScaler.

3. Feature Selection
Feature importance scores obtained using XGBoost.

Low-importance features (importance ≤ 5) were removed to reduce dimensionality and enhance model performance.

4. Model Development
Multiple classifiers were trained and evaluated, including:

Logistic Regression

Support Vector Machine

K-Nearest Neighbors

Decision Tree

Random Forest

Extra Trees

Bagging

AdaBoost

Gradient Boosting

XGBoost

Performance metrics were calculated using both test sets and 5-fold cross-validation:

Accuracy

Precision

Recall

F1 Score

5. Ensemble Learning
Hard and soft voting classifiers were constructed using an ensemble of all base models to leverage collective performance. Hard voting produced slightly higher accuracy and stability.

6. Hyperparameter Tuning
RandomizedSearchCV was used to tune the final XGBoost model for:

n_estimators

max_depth

Scoring metric: accuracy

5-fold cross-validation was used for model validation.

7. Final Model Evaluation
The tuned XGBoost model delivered the following cross-validated performance:

Accuracy: ~94.4%

Precision: ~96.0%

Recall: ~97.4%

F1 Score: ~96.7%

These results confirm strong predictive capability for both churned and retained customers.

8. Model Serialization
The best estimator was saved using joblib for reuse in deployment scenarios.

The fitted StandardScaler was also saved to maintain consistency in preprocessing.

Output directory: models/

models/xgboost_final.joblib

models/scaler.joblib
