# This script performs a comprehensive K-Nearest Neighbors (KNN) regression analysis on a dataset containing advertising data. 

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Read the file 'Advertising.csv' into a Pandas dataset
df = pd.read_csv("Advertising.csv")

# Check for missing values and basic statistics
if df.isnull().sum().any():
    print("Warning: Dataset contains missing values. Please handle them appropriately.")
else:
    print("Dataset contains no missing values.")

# Take a quick look at the data
df.head()

# Set the 'TV' column as predictor variable and reshape into 2D
x = df["TV"].values.reshape(-1, 1)

# Set the 'Sales' column as response variable
y = df["Sales"].values

# Split the dataset in training and testing with 60% training set and
# 40% testing set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, random_state=66
)

# Choosing k range from 1 to 70
k_value_min = 1
k_value_max = 70

# Create a list of integer k values between k_value_min and
# k_value_max using linspace
k_list = np.linspace(k_value_min, k_value_max, num=70, dtype=int)

# Setup a grid for plotting the data and predictions
fig, ax = plt.subplots(figsize=(10, 6))

# Create a dictionary to store the k value against MSE fit {k: MSE@k}
knn_dict = {}

# Variable used for altering the linewidth of values kNN models
j = 0

# Loop over all k values
for k_value in k_list:

    # Create a KNN Regression model for the current k
    model = KNeighborsRegressor(n_neighbors=int(k_value))

    # Fit the model on the train data
    model.fit(x_train, y_train)

    # Use the trained model to predict on the test data
    y_pred = model.predict(x_test)

    # Calculate the MSE of the test data predictions
    MSE = mean_squared_error(y_test, y_pred)

    # Store the MSE values of each k value in the dictionary
    knn_dict[k_value] = MSE

    # Helper code to plot the data and various kNN model predictions
    colors = ["grey", "r", "b"]
    if k_value in [1, 10, 70]:
        xvals = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        ypreds = model.predict(xvals)
        ax.plot(
            xvals,
            ypreds,
            "-",
            label=f"k = {int(k_value)}",
            linewidth=j + 2,
            color=colors[j],
        )
        j += 1

ax.legend(loc="lower right", fontsize=20)
ax.plot(x_train, y_train, "x", label="test", color="k")
ax.set_xlabel("TV budget in $1000", fontsize=20)
ax.set_ylabel("Sales in $1000", fontsize=20)
plt.tight_layout()


# Plot a graph which depicts the relation between the k values and MSE
plt.figure(figsize=(8, 6))
plt.plot(list(knn_dict.keys()), list(knn_dict.values()), "k.-", alpha=0.5, linewidth=2)

# Set the title and axis labels
plt.xlabel("k", fontsize=20)
plt.ylabel("MSE", fontsize=20)
plt.title("Test $MSE$ values for different k values - KNN regression", fontsize=20)
plt.tight_layout()

# Find the lowest MSE among all the kNN models
min_mse = min(knn_dict.values())

# Use list comprehensions to find the k value associated with the lowest MSE
best_model = [key for (key, value) in knn_dict.items() if value == min_mse]

# Print the best k-value
print("The best k value is ", best_model[0], "with a MSE of ", min_mse)

# Compute the R2_score of your best model
model = KNeighborsRegressor(n_neighbors=best_model[0])
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)

# Calculate baseline MSE
baseline_mse = mean_squared_error(y_test, [y_test.mean()] * len(y_test))
print("Baseline MSE:", baseline_mse)

