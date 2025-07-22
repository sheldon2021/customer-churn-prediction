import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import joblib



# 🔹 Load and Inspect the Data

df = pd.read_csv("tele_churn.csv")
print(df.head())

print(df.info())
print(df.isnull().sum())

# Clean TOtal charges since its non numeric we convert it to numeric column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Drop NA values
df.dropna(inplace=True)

#Dropping customer id is because its an identifier and not useful for prediction 
df.drop('customerID', axis=1, inplace=True)

#Print Cleaned data for debugging and visuality
print(df.head())
print("\n🔹 Data Types After Cleaning:")
print(df.dtypes)    

print("\n🔹 Any Missing Values After Cleaning?")
print(df.isnull().sum())

# Preview again
print("\n🔹 Cleaned Data Preview:")
print(df.head())

# Select the categorical columns and convert them to a list and remove the churn column since it has to be predicted
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Churn')  # We'll handle Churn separately



 # select columns with cat cols and drop the first column and convert the categorical columns in one-hot encoding method
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Map the target column seperately
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


#Provide the input (features) and target to the variables
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']               # Target

#Train - test split the data into training and testing , set random state so that we get same random set everytime and improve consistency in this way
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


#Feature scaling is done for the input features only fit is applied only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Applying a model to the training and testing data to carry out the process and check accuracy
model = LogisticRegression()
model.fit(X_train, y_train)

#Calculating the value of y for the model based on testing data of input features = X
y_pred = model.predict(X_test)


#Printing the data acquired for debugging if need & visualization
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




# Uses Seaborn to draw a heatmap of the confusion matrix , plot the x and y lables and also title the graph 


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))



# Probabilities (not labels) take the 1st column of all rows which is the probability of churn
y_probs = rf_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# Plot the roc curve to check the AUC score for the model and see if it ranks the tpr > fpr
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

joblib.dump(rf_model, 'churn_model.pkl')
print("✅ Model saved as churn_model.pkl")

joblib.dump(model, 'logistic_model.pkl')
print("✅ Logistic Regression model saved as logistic_model.pkl")