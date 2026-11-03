
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create dataset
df = pd.DataFrame({
    'links': [0, 1, 0, 1, 2, 5, 6, 7, 8, 9],
    'caps': [2, 3, 1, 4, 5, 20, 25, 30, 28, 35],
    'spam': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

# Step 2: Select input and output
X = df[['links', 'caps']]
y = df['spam']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Create AdaBoost model
model = AdaBoostClassifier(
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

# Step 8: Predict for a new email
new_email = pd.DataFrame({
    'links': [4],
    'caps': [22]
})

print("Spam Prediction:", model.predict(new_email))
