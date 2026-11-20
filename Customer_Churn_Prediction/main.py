
# ==========================================
# CUSTOMER CHURN PREDICTION
# Using Decision Tree Classification
# ==========================================

# STEP 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# STEP 2: Load Dataset
# ==========================================

import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    folder,
    "Telco-Customer-Churn.csv"
)

print("Looking for dataset:", file_path)

data = pd.read_csv(file_path)

print(data.head())
print("Dataset loaded successfully!")
print("\nFIRST FIVE RECORDS")
print(data.head())

print("\nDATASET SHAPE")
print(data.shape)

print("\nCOLUMN NAMES")
print(data.columns.tolist())

print("\nDATA TYPES")
print(data.dtypes)

print("\nMISSING VALUES")
print(data.isnull().sum())

# ==========================================
# STEP 3: Data Cleaning
# ==========================================

# Convert TotalCharges into numeric values
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
data.dropna(inplace=True)

# Remove customerID because it is not useful
data.drop("customerID", axis=1, inplace=True)

print("\nDATA AFTER CLEANING")
print(data.shape)

# ==========================================
# STEP 4: Separate Features and Target
# ==========================================

X = data.drop("Churn", axis=1)

y = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(y.head())

# ==========================================
# STEP 5: Identify Numerical and
# Categorical Columns
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNUMERICAL FEATURES")
print(numeric_features)

print("\nCATEGORICAL FEATURES")
print(categorical_features)

# ==========================================
# STEP 6: Encode Categorical Data
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),

        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# ==========================================
# STEP 7: Split Training and Testing Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# ==========================================
# STEP 8: Create Decision Tree Model
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=20,
        random_state=42
    ))
])

# ==========================================
# STEP 9: Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY!")

# ==========================================
# STEP 10: Make Predictions
# ==========================================

y_pred = model.predict(X_test)

print("\nFIRST 10 PREDICTIONS")
print(y_pred[:10])

print("\nACTUAL VALUES")
print(y_test.iloc[:10].values)

# ==========================================
# STEP 11: Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(f"{accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

# ==========================================
# STEP 12: Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"],
    cmap="Blues"
)

plt.title("Customer Churn Confusion Matrix")
plt.show()

# ==========================================
# STEP 13: Visualize Decision Tree
# ==========================================

trained_tree = model.named_steps["classifier"]

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(24, 12))

plot_tree(
    trained_tree,
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Customer Churn Decision Tree")
plt.show()

# ==========================================
# STEP 14: Feature Importance
# ==========================================

importance = trained_tree.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.head(15))

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

# ==========================================
# STEP 15: Predict for a New Customer
# ==========================================

# Use one unseen testing customer as an example
new_customer = X_test.iloc[[0]]

prediction = model.predict(new_customer)

probability = model.predict_proba(new_customer)[0]

print("\nNEW CUSTOMER DETAILS")
print(new_customer.to_string(index=False))

print("\nCUSTOMER CHURN PREDICTION")

if prediction[0] == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer is likely to STAY.")

print(f"\nChurn Probability: {probability[1] * 100:.2f}%")
print(f"Stay Probability: {probability[0] * 100:.2f}%")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
