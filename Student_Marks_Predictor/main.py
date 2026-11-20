
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_folder, "student_data.csv")

data = pd.read_csv(csv_path)

print(data.head())

X = data[["Hours"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

predicted_marks = model.predict(X)

plt.scatter(X, y, label="Actual Marks")

plt.plot(X, predicted_marks, label="Prediction Line")

plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Student Marks Prediction")

plt.legend()
plt.show()

hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

print("Predicted Marks:", round(prediction[0], 2))
