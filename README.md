# Project: TV Advertising Sales Prediction using k-Nearest Neighbors (kNN)  

## Overview
This project aims to explore the relationship between TV advertising budgets and sales using the k-Nearest Neighbors (kNN) regression technique. By applying kNN, we can understand how varying values of TV advertising budgets impact sales, visualize the model predictions, and analyze the accuracy of the kNN approach in capturing this relationship.

## Purpose
The primary objective of this project is to apply kNN regression to the Advertising.csv dataset to predict sales based on TV advertising budgets. We aim to demonstrate the effect of different values of k (the number of nearest neighbors) on the model’s predictions and performance, and also find the best value for k in this model.

## Dataset
The [dataset](https://github.com/tamunoWoks/advert_analysis/blob/main/Advertising.csv) used in this project is from the Advertising.csv file. It contains information on TV advertising budgets and corresponding sales data collected from a set of observations.
Here is a brief look of the data.
- ![Data Head (first 10 rows)](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/data_head_10.png)
- ![Data Description](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/data_description.png)

## Columns in the dataset:
- TV: TV advertising budget (in thousands of dollars)
- Radio: Radio advertising budget (in thousands of dollars)
- Newspaper: Newspaper advertising budget (in thousands of dollars)
- Sales: Sales (in thousands of dollars)

## Scripts and Analysis
### 1. Script 1: kNN Regression for k >= 1, with TV as Predictor
- Objective: This [script](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/kNN_regression_analysis.ipynb) applies kNN regression to predict sales based on TV advertising budgets. It demonstrates how different values of k (from 1 to 70) impact the model’s predictions and visualizes these results.

- Key Components:

    - Reading the data from Advertising.csv.
    - Splitting the data into training and testing sets.
    - Training kNN models with varying k values.
    - Plotting the regression results, showing how k affects model performance.
  
- Results:

    - The script provides insights into how increasing or decreasing k influences the accuracy of predictions.
    - A visual comparison of the predicted values with actual sales data is generated, helping to identify the optimal k.
    - ![Visualization for kNN Regression where k = 1, 10 and 70, with TV as Predictor](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/k_for_several_values.png)
  
### 2. Script 2: Find the best value of k
- Objective: This [script](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/find_best_k_in_kNN_regression.ipynb) implements a function to find the nearest neighbor for each given value of TV advertising budget and predicts sales accordingly.

- Key Components:

    - Import necessary libraries.
    - Reading the data from Advertising.csv.
    - Splitting the data into training and testing sets.
    - Choosing k range from 1 to 70
    - Plotting the regression results, showing how k affects model performance
    - Plot a graph which depicts the relation between the k values and MSE
    - Find the lowest MSE among all the kNN models, find the k value associated with the lowest MSE.
    - compute the R2 score
    
- Results:

    - The nearest neighbor approach provides a basic understanding of the relationship between TV budgets and sales.
    - The best k value is  9 with a MSE of  13.046766975308643
    - The R2 score for your model is 0.5492457002030715
    - ![Visualization of MSE value test for different values of k](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/test_MSE_for_different_values_of_k.png)
  

## 3. Script 3: Simple Scatter Plot of TV vs Sales
- Objective: This [script](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/tv_vs_sales_scatter_plot.ipynb) focuses on visualizing the relationship between TV advertising budgets and sales using a simple scatter plot.

- Key Components:

    - Reading the data from Advertising.csv and displaying the first 7 rows.
    - Creating a scatter plot to visualize the correlation between TV budgets and sales.
    - Adding axis labels and titles to enhance readability.
  
- Results:

    - A clear visual representation of the TV advertising budget vs sales data.
    - Helps to understand the general trend of how TV budgets impact sales.
    - ![Visualization of scatter plot for first ten values of TV vs Sales](https://github.com/tamunoWoks/kNN_regression_sales_prediction_analysis/blob/main/scatter%20plot%20tv%20vs%20sales.png)

## Installation and Prerequisites
### 1. Environment Setup:

 - Python 3.x installed on your machine.
 - Required libraries: numpy, pandas, matplotlib, sklearn.

### 2. Data File:

- The [Advertising.csv](https://github.com/tamunoWoks/advert_analysis/blob/main/Advertising.csv) file is located in the same directory as the scripts.

## Conclusion

This project provides a clear understanding of how k-Nearest Neighbors regression can be applied to predict sales based on TV advertising budgets. By experimenting with different values of k, we explored the impact of model complexity and accuracy, thereby predicting the best value of k. The visualizations and analysis contribute to a better grasp of the relationship between TV advertising and sales, helping to make informed decisions in business scenarios.
