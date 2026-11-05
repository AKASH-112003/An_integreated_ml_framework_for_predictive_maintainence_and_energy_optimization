# ---------------- IMPORT LIBRARIES ----------------
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier


# ---------------- DATASET FILE ----------------
DATASET_FILE = "motor_dataset.csv"


# ---------------- CREATE DATASET ----------------
def create_dataset(file_path, n_samples=10000):

    data = []

    for _ in range(n_samples):

        # ---- SENSOR VALUES ----
        voltage = round(np.random.uniform(130, 270), 2)
        current = round(np.random.uniform(0, 20), 2)
        temperature = round(np.random.uniform(20, 60), 2)

        # ---- FAULT LOGIC ----
        fault = 0

        if voltage >= 250:
            fault = 1
        elif voltage <= 150:
            fault = 1
        elif temperature >= 40:
            fault = 1

        data.append([
            voltage,
            current,
            temperature,
            fault
        ])

    df = pd.DataFrame(data, columns=[
        "Voltage",
        "Current",
        "Temperature",
        "Fault"
    ])

    df.to_csv(file_path, index=False)
    return df


# ---------------- LOAD OR CREATE DATA ----------------
if os.path.exists(DATASET_FILE):
    print("📂 Loading dataset...")
    df = pd.read_csv(DATASET_FILE)
else:
    print("⚡ Creating dataset...")
    df = create_dataset(DATASET_FILE)


# ---------------- FEATURES & LABEL ----------------
X = df[["Voltage", "Current", "Temperature"]]
y = df["Fault"]


# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ---------------- MODEL TRAINING ----------------
print("\n🚀 Training XGBoost Model...")

model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)


# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)


# ---------------- EVALUATION ----------------
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", accuracy)


# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost Motor Fault")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ---------------- FEATURE IMPORTANCE ----------------
importance = model.feature_importances_

plt.figure(figsize=(6,4))
plt.barh(X.columns, importance)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Score")
plt.show()


# ---------------- SAVE MODEL ----------------
MODEL_FILE = "motor_fault_model.pkl"
joblib.dump(model, MODEL_FILE)

print(f"\n💾 Model saved as {MODEL_FILE}")