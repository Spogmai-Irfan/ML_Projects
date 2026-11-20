

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


print("\nDownloading House Price Dataset...")

try:
    housing = fetch_openml(
        name="house_prices",
        version=1,
        as_frame=True
    )

    X = housing.data
    y = housing.target

    print("Dataset loaded successfully!")

except Exception as e:
    print("Dataset could not be downloaded.")
    print("Check your internet connection.")
    print("Error:", e)
    raise SystemExit



features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "Neighborhood"
]

target = "SalePrice"

X = X[features].copy()
y = pd.to_numeric(y, errors="coerce")

valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("\nSelected Dataset Columns:")
print(X.head())

print("\nDataset Information:")
print(X.info())



print("\nMissing Values:")
print(X.isnull().sum())

print("\nTarget Price Information:")
print(y.describe())


numerical_features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath"
]

categorical_features = [
    "Neighborhood"
]



numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))



model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


print("\nTraining Machine Learning Model...")

model.fit(X_train, y_train)

print("Model trained successfully!")



predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")

results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": predictions[:10]
})

print(results.round(2))


mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: ${mae:,.2f}")

print(f"Mean Squared Error: {mse:,.2f}")

print(f"Root Mean Squared Error: ${rmse:,.2f}")

print(f"R2 Score: {r2:.4f}")


joblib.dump(model, "house_price_model.pkl")

print("\nTrained model saved successfully!")



try:

    area = float(input("\nEnter house size (square feet): "))

    bedrooms = int(input("Enter number of bedrooms: "))

    bathrooms = int(input("Enter number of full bathrooms: "))

    neighborhoods = sorted(
        X["Neighborhood"].dropna().unique()
    )

    print("\nAvailable Neighborhoods:")

    print(", ".join(neighborhoods))

    location = input("\nEnter neighborhood exactly as shown: ").strip()


    new_house = pd.DataFrame({
        "GrLivArea": [area],
        "BedroomAbvGr": [bedrooms],
        "FullBath": [bathrooms],
        "Neighborhood": [location]
    })


    predicted_price = model.predict(new_house)

    

    print(f"House Size: {area:,.0f} sq ft")

    print(f"Bedrooms: {bedrooms}")

    print(f"Bathrooms: {bathrooms}")

    print(f"Neighborhood: {location}")

    print("\nEstimated House Price:")

    print(f"${predicted_price[0]:,.2f}")

    print("\nPrediction completed successfully!")

except ValueError:
    print("\nPlease enter valid numerical values.")

except Exception as e:
    print("\nSomething went wrong:", e)
