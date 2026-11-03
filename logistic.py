
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
df = pd.DataFrame({
    'area': [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
    'rooms': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    'price': [26, 38, 48, 63, 72, 83, 101, 112, 128, 137]
})

# Input and output
X = df[['area', 'rooms']]
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Display results
print("Predicted Prices:", y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict for new house
new_house = pd.DataFrame({
    'area': [1600],
    'rooms': [3]
})

print("Predicted Price:", model.predict(new_house))
