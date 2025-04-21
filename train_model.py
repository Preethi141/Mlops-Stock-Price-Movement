import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("stock_sentiment_extended.csv")
print(df)
# One-hot encode ticker
df = pd.get_dummies(df, columns=["ticker"])

# Store the final column order for prediction later
feature_columns = [col for col in df.columns if col not in ["date", "movement"]]
X = df[feature_columns]
y = df["movement"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and column order
joblib.dump(model, "model.pkl")
joblib.dump(feature_columns, "model_columns.pkl")

# Report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
