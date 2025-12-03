
#  Duolingo-Style User Behavior Prediction Project
#  FULL END-TO-END PIPELINE IN ONE FILE


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1️ LOAD DATA


DATA_PATH = "user_behavior_dataset.csv"   # <-- change this

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.info())

# 2️⃣ INITIAL OVERVIEW


print("\n=== Dataset Summary ===")
print(df.describe())
print("\nClass Distribution:")
print(df['User Behavior Class'].value_counts())

# 3️ EDA 


plt.figure(figsize=(12, 10))
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 4️ PREPROCESSING


print("\nEncoding categorical variables...")
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("User Behavior Class", axis=1)
y = df_encoded["User Behavior Class"]

print(f"\nTotal Features After Encoding: {X.shape[1]}")


# 5️ TRAIN/TEST SPLIT
# 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print("\nTrain/Test split complete.")


# 6️ SCALING


print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7️⃣ TRAIN RANDOM FOREST MODEL


print("\nTraining Random Forest Model...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf.fit(X_train_scaled, y_train)


# 8️⃣ EVALUATION


y_pred = rf.predict(X_test_scaled)

print("\n=== MODEL EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 9️⃣ FEATURE IMPORTANCE


importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 15 Feature Importances")
plt.show()


# 🔟 SAVE MODEL + SCALER


print("\nSaving model and scaler...")
joblib.dump(rf, "user_behavior_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n=== DONE! ===")
print("Saved model: user_behavior_model.pkl")
print("Saved scaler: scaler.pkl")
