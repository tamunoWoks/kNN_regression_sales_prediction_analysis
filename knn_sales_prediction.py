# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Read the data from the file "Advertising.csv"
filename = "Advertising.csv"
df_adv = pd.read_csv(filename)

# Take a quick look of the dataset
df_adv.head()

# Get a subset of the data i.e. rows 5 to 13
# Use the TV column as the predictor
x_true = df_adv.TV.iloc[5:13]

# Use the Sales column as the response
y_true = df_adv.Sales.iloc[5:13]

# Sort the data to get indices ordered from lowest to highest TV values
idx = np.argsort(x_true).values

# Get the predictor data in the order given by idx above
x_true = x_true.iloc[idx].values

# Get the response data in the order given by idx above
y_true = y_true.iloc[idx].values

### edTest(test_findnearest) ###
# Define a function that finds the index of the nearest neighbor
# and returns the value of the nearest neighbor.
# Note that this is just for k = 1 where the distance function is
# simply the absolute value.


def find_nearest(array, value):

    # Hint: To find idx, use .idxmin() function on the series
    idx = pd.Series(np.abs(array - value)).idxmin()

    # Return the nearest neighbor index and value
    return idx, array[idx]


# Create some synthetic x-values (might not be in the actual dataset)
x = np.linspace(np.min(x_true), np.max(x_true), 100)

# Initialize the y-values for the length of the synthetic x-values to zero
y = np.zeros(len(x))

# Apply the KNN algorithm to predict the y-value for the given x value
for i, xi in enumerate(x):

    # Get the Sales values closest to the given x value
    nearest_idx, _ = find_nearest(x_true, xi)
    y[i] = y_true[nearest_idx]

# Plot the synthetic data along with the predictions
plt.plot(x, y, "-.", label="Predicted Sales")

# Plot the original data using black x's.
plt.plot(x_true, y_true, "kx", label="Original data")

# Set the title and axis labels
plt.title("TV vs Sales")
plt.xlabel("TV budget in $1000")
plt.ylabel("Sales in $1000")

# Add a legend
plt.legend()

# Display the plot
plt.show()
