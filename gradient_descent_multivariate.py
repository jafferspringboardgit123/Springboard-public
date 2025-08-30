import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Step 1: Prepare sample data
# Features: [Size (sqft), Number of bedrooms]
X = np.array([
    [2104, 3],
    [1600, 3],
    [2400, 3],
    [1416, 2],
    [3000, 4]
], dtype=float)

# Target: Price
y = np.array([399900, 329900, 369000, 232000, 539900], dtype=float)

# Step 2: Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Step 3: Add bias term (column of ones for theta0)
m = len(y)
X_aug = np.hstack([np.ones((m, 1)), X_norm])  # Shape: (m, n+1)

# Step 4: Initialize parameters
n = X_aug.shape[1]  # Number of features + 1
theta = np.zeros(n)
alpha = 0.01
epochs = 1000

# Step 5: Gradient Descent Loop
cost_history = []

for i in range(epochs):
    predictions = X_aug @ theta  # Matrix multiplication
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    cost_history.append(cost)

    gradients = (1 / m) * (X_aug.T @ errors)
    theta -= alpha * gradients

    if i % 100 == 0:
        print(f"Iteration {i}: Cost={cost:.2f}, Thetas={theta}")

# Step 6: Final Parameters
print("\nFinal theta values:", theta)

# Step 7: Predict a new house (e.g., 1650 sqft, 3 be
# drooms)
new_house = np.array([1650, 3], dtype=float)
new_house_norm = (new_house - X_mean) / X_std
new_house_aug = np.hstack([1, new_house_norm])
predicted_price = new_house_aug @ theta
print(f"\nPredicted price for house {new_house}: ${predicted_price:.2f}")

# Step 8: Plot cost history
plt.plot(range(epochs), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost vs. Iterations")
plt.grid(True)
plt.show()
st.pyplot(plt)