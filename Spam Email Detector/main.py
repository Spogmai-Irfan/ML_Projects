
import pandas as pd
import urllib.request
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

zip_file = "spam_dataset.zip"

if not os.path.exists("SMSSpamCollection"):
    print("Downloading dataset...")

    urllib.request.urlretrieve(url, zip_file)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Dataset downloaded successfully!")


data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("\nDataset loaded successfully!")

print(data.head())

print("\nDataset shape:", data.shape)


data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

print("\nAfter cleaning:")

print(data.shape)

# Convert labels into numbers
# ham = 0 (Not Spam)
# spam = 1 (Spam)

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# =====================================
# 4. SPLIT DATA INTO FEATURES AND LABELS
# =====================================

X = data["message"]

y = data["label"]


# =====================================
# 5. SPLIT DATASET INTO TRAINING AND TESTING
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))



model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])



print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model trained successfully!")


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Not Spam", "Spam"]
))



while True:

    message = input(
        "\nEnter your message (or type exit): "
    )

    if message.lower() == "exit":
        print("Thank you for using Spam Detector!")
        break

    prediction = model.predict([message])

    if prediction[0] == 1:

        print("\nResult: SPAM MESSAGE!")

    else:

        print("\nResult: NOT SPAM!")

