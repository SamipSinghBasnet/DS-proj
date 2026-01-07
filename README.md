Duolingo-Style User Behavior Prediction
End-to-End Machine Learning Pipeline

Overview
This project implements a complete machine learning pipeline to classify mobile user behavior based on device usage patterns. The workflow includes data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and model persistence. The goal is to build a reproducible and interpretable system suitable for academic research and real-world analytics applications.

Dataset
The dataset contains 700 user records with 11 features:

User ID

Device Model

Operating System

App Usage Time (min/day)

Screen On Time (hours/day)

Battery Drain (mAh/day)

Number of Apps Installed

Data Usage (MB/day)

Age

Gender

User Behavior Class (target variable: classes 1–5)

The class distribution is balanced across all five behavior categories.

Exploratory Data Analysis
Key findings from EDA:

Usage-related features such as App Usage Time, Screen On Time, Battery Drain, Number of Apps Installed, and Data Usage show strong positive correlations with each other and with the target class.

Age shows minimal correlation with user behavior.

Histograms indicate varied distributions across usage metrics, with most users falling into moderate usage ranges.

Preprocessing
The preprocessing pipeline includes:

One-hot encoding of categorical variables (Device Model, Operating System, Gender)

Standard scaling of numerical features

Stratified train-test split (80/20) to preserve class proportions

Final feature count after encoding: 13

Model
A Random Forest classifier was selected due to its robustness, interpretability, and ability to handle mixed feature types.

Model configuration:

n_estimators = 300

max_depth = 10

random_state = 42

The model was trained on scaled training data.

Evaluation
The model achieved perfect performance on the test set:

Accuracy: 1.00

Precision, Recall, F1-score: 1.00 for all five classes

Confusion matrix: Zero misclassifications across all classes

This indicates that the feature set is highly predictive for the target behavior categories.

Feature Importance
The top predictive features identified by the Random Forest model include:

Data Usage (MB/day)

Number of Apps Installed

App Usage Time (min/day)

Battery Drain (mAh/day)

Screen On Time (hours/day)

These features align with expected behavioral patterns in mobile device usage.

Model Saving
The trained model and scaler are saved using Joblib:

user_behavior_model.pkl

scaler.pkl

These artifacts can be loaded for downstream inference or integration into production systems.

#Results Visual

<img width="3798" height="1938" alt="image" src="https://github.com/user-attachments/assets/f8ff5b8c-02e9-41b2-a702-3b84ecfe6a8b" />
<img width="3263" height="1825" alt="image" src="https://github.com/user-attachments/assets/3c2e90f3-a370-4619-a477-81a507e3ab1a" />
<img width="1406" height="1072" alt="image" src="https://github.com/user-attachments/assets/88b7b71b-30d1-4156-8aca-f44ea44d344d" />
<img width="2371" height="1465" alt="image" src="https://github.com/user-attachments/assets/b78892a8-c1a7-439a-887d-41652fae2469" />

Future Work
Evaluate additional models such as XGBoost, SVM, and Gradient Boosting

Incorporate temporal usage patterns for sequential modeling

Apply SHAP or LIME for deeper interpretability

Deploy as an API for real-time behavior prediction


