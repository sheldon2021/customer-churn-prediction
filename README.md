# 📊 Customer Churn Prediction with Python

This project is a **machine learning-based churn prediction system** built using Python. It predicts whether a customer is likely to churn (i.e., stop using a service) based on historical data. It uses preprocessing, logistic regression, and model evaluation techniques.

---

## 🚀 Features

- Data cleaning and preprocessing
- One-hot encoding of categorical variables
- Logistic Regression model
- Accuracy, confusion matrix, and classification report
- Confusion matrix heatmap using Seaborn
- Train-test split using scikit-learn
- Feature scaling with `StandardScaler`
- Ready to extend with:
  - Cross-validation
  - GridSearchCV
  - Random Forests
  - Streamlit/Flask deployment

---

## 📁 Dataset

- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Source: Kaggle (Telco Customer Churn)
- Columns include:
  - `customerID`, `gender`, `InternetService`, `Contract`, `TotalCharges`, `Churn`, etc.

---

## 🧹 Data Preprocessing

- Convert `TotalCharges` to numeric
- Drop `customerID` (identifier)
- Drop rows with missing values
- One-hot encode all categorical features (`drop_first=True`)
- Scale features using `StandardScaler`

---

## 🧠 Model Used

- Logistic Regression from `sklearn.linear_model`
- Split: 80% train / 20% test
- Evaluation metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 📈 Results Example

Accuracy: 0.80
Confusion Matrix:
[[900 100]
[140 260]]
Classification Report:
precision recall f1-score support

markdown
Copy
Edit
       0       0.87      0.90      0.88      1000
       1       0.72      0.65      0.68       400
yaml
Copy
Edit

---

## 📊 Visualization

- Seaborn Heatmap for confusion matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, cmap="Blues")
