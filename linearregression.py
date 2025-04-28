# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('Housing.csv')  # replace with your actual file name

# Step 2: Check columns
print("Columns available:\n", data.columns.tolist())

# Step 3: Preprocessing
# Selecting features and target
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]  # You can add/remove columns
y = data['price']

# Step 4: Handle categorical variables if you want to use them (optional)
# (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
# Uncomment below lines if needed
# data = pd.get_dummies(data, drop_first=True)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Step 9: Interpreting the model
print("\nModel Coefficients:")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

print("\nIntercept (b0):", model.intercept_)

# Step 10: Plotting (only possible if using single feature like 'area')
# Optional: Only if X = [['area']]
# plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
# plt.plot(X_test['area'], y_pred, color='red', linewidth=2, label='Predicted Price')
# plt.title('Area vs Price')
# plt.xlabel('Area')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
