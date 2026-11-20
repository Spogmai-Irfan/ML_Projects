
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
data = pd.read_csv(
    r"D:\ML Projects\ML_Projects\Credit Card Fraud Detection\creditcard.csv"
)
print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
print(data.info())

print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())
class_counts = data["Class"].value_counts()

print("Transaction Counts:")
print(class_counts)

print("\nTransaction Percentages:")
print(data["Class"].value_counts(normalize=True) * 100)

print("\n0 = Legitimate Transactions")
print("1 = Fraudulent Transactions")

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="Class")

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.xticks([0, 1], ["Legitimate", "Fraud"])

plt.show()

print("\n========== TRANSACTION ANALYSIS ==========\n")

print("Average Transaction Amount by Class:")

print(data.groupby("Class")["Amount"].mean())

print("\nTransaction Amount Statistics:")

print(data.groupby("Class")["Amount"].describe())


plt.figure(figsize=(10, 5))

sns.boxplot(data=data, x="Class", y="Amount")


plt.title("Transaction Amount: Fraud vs Legitimate")

plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")

plt.show()
print("\n========== PREPARING DATA ==========\n")

X = data.drop("Class", axis=1)

y = data["Class"]

print("Features Shape:", X.shape)

print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)

print("Testing Data:", X_test.shape)

print("\nTraining Class Distribution:")

print(y_train.value_counts())

print("\nTesting Class Distribution:")

print(y_test.value_counts())

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Baseline Model Training Completed!")

y_pred = model.predict(X_test)

print("\n========== BASELINE MODEL EVALUATION ==========\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred, zero_division=0))

print("Recall:", recall_score(y_test, y_pred, zero_division=0))

print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Baseline Model - Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.show()
print("\n========== BALANCED RANDOM FOREST ==========\n")

balanced_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

balanced_model.fit(X_train, y_train)

print("Balanced Model Training Completed!")

balanced_pred = balanced_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, balanced_pred))

print("Precision:", precision_score(y_test, balanced_pred, zero_division=0))

print("Recall:", recall_score(y_test, balanced_pred, zero_division=0))

print("F1 Score:", f1_score(y_test, balanced_pred, zero_division=0))

print("\nClassification Report:")

print(classification_report(
    y_test,
    balanced_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

balanced_cm = confusion_matrix(y_test, balanced_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    balanced_cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Balanced Model - Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.show()
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],

    "Baseline Model": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0)
    ],

    "Balanced Model": [
        accuracy_score(y_test, balanced_pred),
        precision_score(y_test, balanced_pred, zero_division=0),
        recall_score(y_test, balanced_pred, zero_division=0),
        f1_score(y_test, balanced_pred, zero_division=0)
    ]
})

print(comparison.to_string(index=False))

results = X_test.copy()

results["Actual"] = y_test

results["Predicted"] = balanced_pred

# False Positives

false_positives = results[
    (results["Actual"] == 0) &
    (results["Predicted"] == 1)
]

# False Negatives

false_negatives = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 0)
]

print("Number of False Positives:", len(false_positives))

print("Number of False Negatives:", len(false_negatives))

print("\nFalse Positive Examples:")

print(false_positives.head())

print("\nFalse Negative Examples:")

print(false_negatives.head())
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": balanced_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.show()
print("CREDIT CARD FRAUD DETECTION COMPLETED!")
print("Baseline F1 Score:",
      f1_score(y_test, y_pred, zero_division=0))

print("Balanced F1 Score:",
      f1_score(y_test, balanced_pred, zero_division=0))
