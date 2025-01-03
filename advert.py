# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


data_filename = 'Advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)

# Get a quick look at the data
df.describe()

# Create a new dataframe by selecting the first 7 rows of the current dataframe
df_new = df.head(7)
print(df_new)

# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(data['TV'], data['Sales'])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel('TV budget')
plt.ylabel('Sales')

# Add plot title 
plt.title('TV Budget vs Sales')
plt.show()
