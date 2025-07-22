# Customer Churn Prediction Project
This is a machine learning project I worked on to predict customer churn — basically figuring out which customers are likely to leave a company based on some past data. It's a common use case for companies like telecom, internet services, etc.

---

## Project Overview
The goal here was to build a classification model that takes in customer info and predicts whether or not they’ll churn (i.e., leave). The dataset included things like contract type, charges, internet usage, and more. I cleaned the data, explored it, trained a few models, and evaluated their performance.

---

## Files in This Repo
- `churn_prediction.ipynb`: Main Jupyter notebook with all the code.
- `churn_data.csv`: The dataset I used (if it's not here, it's probably too big for GitHub).
- `README.md`: This file you're reading right now.

---

## What I Did (Step-by-Step)
1. **Loaded and cleaned the data** – removed missing values, fixed data types.
2. **Explored the data** – used plots and charts to understand churn trends.
3. **Did some feature engineering** – used one-hot encoding for categorical stuff.
4. **Split data** into training and testing sets.
5. **Trained a Logistic Regression model** (basic to start with).
6. **Evaluated it** using accuracy score, confusion matrix, and ROC/AUC.
7. **Saved the model** using `joblib` so it can be reused later.

---

## Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn & Matplotlib


---

## What’s Next / Improvements
- Try better models like Decision Trees or Random Forest.
- Use GridSearchCV for hyperparameter tuning.
- Maybe turn this into a small app using Streamlit.
- Deploy it somewhere like Hugging Face or Render.

---
