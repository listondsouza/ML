
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create dataset
df = pd.DataFrame({
    'income': [20, 25, 30, 35, 40, 50, 60, 70, 80, 90],
    'credit': [450, 500, 520, 550, 600, 650, 700, 720, 750, 800],
    'loan': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

# Step 2: Select input and output
X = df[['income', 'credit']]
y = df['loan']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Create Bagging model
model = BaggingClassifier(
    n_estimators=10,
    random_state=42
)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate model
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Predict for a new customer
new_customer = pd.DataFrame({
    'income': [55],
    'credit': [680]
})

print("Loan Approval Prediction:", model.predict(new_customer))
