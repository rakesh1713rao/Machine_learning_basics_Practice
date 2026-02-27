import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Square footage (X) and House Price (y)
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train) # This is the "Learning" phase


# new_house = np.array([[1800]])
# prediction = model.predict(new_house)
# print(f"Predicted Price: ${prediction[0]}")


# 1. Get predictions for our test data
y_pred = model.predict(X_test)

# 2. Compare them to the actual prices (y_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")