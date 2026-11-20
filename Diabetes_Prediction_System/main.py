

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


print("\nLoading Diabetes Dataset...")

url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

data = pd.read_csv(url)

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nDataset Shape:")
print(data.shape)

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDiabetes Outcome Counts:")
print(data["Outcome"].value_counts())



columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[columns_with_zero] = data[columns_with_zero].replace(0, np.nan)

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())


plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="Outcome")

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Patients")

plt.show()



plt.figure(figsize=(10, 7))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


X = data.drop("Outcome", axis=1)

y = data["Outcome"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])


parameters = {
    "knn__n_neighbors": range(1, 22, 2)
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="recall"
)

grid_search.fit(X_train, y_train)

print("\nBest K Value:")
print(grid_search.best_params_)

print("\nBest Cross Validation Recall:")
print(grid_search.best_score_)


model = grid_search.best_estimator_

print("\nKNN Model Trained Successfully!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall:    {recall * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

k_values = range(1, 22, 2)

accuracies = []
recalls = []

for k in k_values:

    test_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    test_pipeline.fit(X_train, y_train)

    predictions = test_pipeline.predict(X_test)

    accuracies.append(
        accuracy_score(y_test, predictions)
    )

    recalls.append(
        recall_score(y_test, predictions, zero_division=0)
    )

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    accuracies,
    marker="o",
    label="Accuracy"
)

plt.plot(
    list(k_values),
    recalls,
    marker="s",
    label="Recall"
)

plt.xlabel("K Value")
plt.ylabel("Score")
plt.title("KNN Performance for Different K Values")

plt.legend()
plt.grid(True)

plt.show()


print("\n========== NEW PATIENT PREDICTION ==========")

# Order:
# Pregnancies, Glucose, BloodPressure,
# SkinThickness, Insulin, BMI,
# DiabetesPedigreeFunction, Age

new_patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}])

prediction = model.predict(new_patient)

probabilities = model.predict_proba(new_patient)

print("\nPatient Information:")
print(new_patient)

if prediction[0] == 1:

    print("\nModel Prediction: Diabetes-positive class")

else:

    print("\nModel Prediction: Diabetes-negative class")

print(
    f"\nModel estimated probability for positive class: "
    f"{probabilities[0][1] * 100:.2f}%"
)


