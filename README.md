# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data.

2.Split Dataset into Training and Testing Sets.

3.Train the Model Using Stochastic Gradient Descent (SGD).

4.Make Predictions and Evaluate Accuracy.

5.Generate Confusion Matrix.

## Program:
```
# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load Dataset
data = pd.read_csv("Placement_Data.csv")

# 3️⃣ Separate Features and Target
X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]

# 4️⃣ One-Hot Encoding for Categorical Columns
X = pd.get_dummies(X, drop_first=True)

# 5️⃣ Save Feature Names
feature_names = X.columns

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Feature Scaling (keep as DataFrame)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

# 8️⃣ Create SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# 9️⃣ Train Model
model.fit(X_train_scaled, y_train)

# 🔟 Predict on Test Data
y_pred = model.predict(X_test_scaled)

# 1️⃣1️⃣ Accuracy & Reports
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================
# 1️⃣2️⃣ Correct New Student Prediction
# ============================

# Create empty student with all features = 0
new_student_dict = dict.fromkeys(feature_names, 0)

# Fill ONLY numerical features (example values)
new_student_dict['ssc_p'] = 67
new_student_dict['hsc_p'] = 91
new_student_dict['degree_p'] = 58
new_student_dict['etest_p'] = 88
new_student_dict['mba_p'] = 67

# Convert to DataFrame with same columns
new_student_df = pd.DataFrame([new_student_dict])

# Scale using same scaler
new_student_scaled = pd.DataFrame(
    scaler.transform(new_student_df),
    columns=feature_names
)

# Predict
pred = model.predict(new_student_scaled)
print("\nPredicted Status:", pred[0])
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SRIRAM S
RegisterNumber:  212225240155
*/
```

## Output:
<img width="647" height="403" alt="Screenshot 2026-02-03 093614" src="https://github.com/user-attachments/assets/0bba5c72-194e-44c6-aae0-b5062b64f9f8" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
