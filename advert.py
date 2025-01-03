# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


data_filename = 'Advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)
df.head()

# Create a new dataframe by selecting the first 7 rows of the current dataframe
df_new = df.head(7)
print(df_new)
