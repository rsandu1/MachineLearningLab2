# CISC 5800 Homework 2 Question 2
# Robert Sandu

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Inputs: u (input vector), y (output vector)
u = np.array([...])  # Replace with actual data
y = np.array([...])  # Replace with actual data

# Step 1: Split data into training and test sets
u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.5, random_state=42)

# Step 2: Fit models for d = 1, 2, ..., 10
best_mse = float('inf')
best_d = None
for d in range(1, 11):
    # Create design matrix for order d
    X_train = np.array([np.exp(-j * u_train / d) for j in range(d + 1)]).T
    X_test = np.array([np.exp(-j * u_test / d) for j in range(d + 1)]).T

    # Fit linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Compute mean squared error on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Update best model
    if mse < best_mse:
        best_mse = mse
        best_d = d

print(f"Best model order: {best_d}, MSE: {best_mse}")
