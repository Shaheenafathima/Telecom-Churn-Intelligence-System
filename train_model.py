import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading Telecom Dataset...")

df = pd.read_csv("dataset/Churn.csv")

print(df.head())
print(df.columns)
print(df.columns.tolist())

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert Churn
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Features
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "gender"
]

X = df[features]
y = df["Churn"]

# One-hot encoding
X = pd.get_dummies(X)

model_columns = X.columns.tolist()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# XGBoost Model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Accuracy
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("Model Saved Successfully")