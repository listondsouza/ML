
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create dataset
df = pd.DataFrame({
    'age': [22, 25, 28, 30, 35, 40, 45, 48, 52, 55],
    'salary': [15, 18, 20, 25, 40, 55, 60, 70, 80, 90],
    'buy': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

# Step 2: Select input and output
X = df[['age', 'salary']]
y = df['buy']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Create SVM model
model = SVC(kernel='linear')

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
    'age': [38],
    'salary': [50]
})

print("Purchase Prediction:", model.predict(new_customer))
