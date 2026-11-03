import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create the dataset
df = pd.DataFrame({
    'area': [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
    'rooms': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    'price': [26, 38, 48, 63, 72, 83, 101, 112, 128, 137]
})

# Step 2: Select input and output
X = df[['area', 'rooms']]
y = df['price']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create the model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict prices
y_pred = model.predict(X_test)

# Step 7: Display results
print("Predicted Prices:", y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 8: Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Predict price for a new house
new_house = pd.DataFrame({
    'area': [1600],
    'rooms': [3]
})

print("Predicted Price for new house:", model.predict(new_house))