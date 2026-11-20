
# ==========================================
# PROJECT 3: TITANIC SURVIVAL PREDICTION
# ==========================================

# 1. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = pd.read_csv(
    r"D:\ML Projects\ML_Projects\Titanic _Survival_Prediction\train.csv"
)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())


# 3. Explore the dataset

print("\nSurvival count:")
print(data["Survived"].value_counts())

sns.countplot(data=data, x="Survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


# 4. Select useful features

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = data[features]
y = data["Survived"]


# 5. Separate numerical and categorical columns

numerical_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# 6. Handle missing values and prepare data

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 7. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 8. Create and train the model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("\nModel training completed!")


# 9. Make predictions

predictions = model.predict(X_test)

print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nActual answers:")
print(y_test.iloc[:10].values)


# 10. Evaluate model accuracy

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# 11. Confusion Matrix

cm = confusion_matrix(y_test, predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.title("Titanic Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 12. Test with a new passenger

new_passenger = pd.DataFrame([{
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80,
    "Embarked": "S"
}])

result = model.predict(new_passenger)

print("\nNew Passenger Prediction:")

if result[0] == 1:
    print("Passenger Survived!")
else:
    print("Passenger Did Not Survive!")

print("\nSurvival Probability:",
      model.predict_proba(new_passenger)[0][1] * 100, "%")
