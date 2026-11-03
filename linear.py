
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
df = pd.DataFrame({
    'area': [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
    'price': [25, 36, 49, 60, 74, 85, 99, 110, 124, 135]
})

# Separate input and output
X = df[['area']]
y = df['price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display results
print("Predicted:", y_pred)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict price for a new house
new_house = pd.DataFrame({'area': [1800]})
print("Price for 1800 sqft:", model.predict(new_house))
