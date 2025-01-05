# Project: TV Advertising Sales Prediction using k-Nearest Neighbors (kNN)  

## Overview
This project aims to explore the relationship between TV advertising budgets and sales using the k-Nearest Neighbors (kNN) regression technique. By applying kNN, we can understand how varying values of TV advertising budgets impact sales, visualize the model predictions, and analyze the accuracy of the kNN approach in capturing this relationship.

## Purpose
The primary objective of this project is to apply kNN regression to the Advertising.csv dataset to predict sales based on TV advertising budgets. We aim to demonstrate the effect of different values of k (the number of nearest neighbors) on the model’s predictions and performance.

## Dataset
The dataset used in this project is from the Advertising.csv file. It contains information on TV advertising budgets and corresponding sales data collected from a set of observations.

## Columns in the dataset:
- TV: TV advertising budget (in thousands of dollars)
- Radio: Radio advertising budget (in thousands of dollars)
- Newspaper: Newspaper advertising budget (in thousands of dollars)
- Sales: Sales (in thousands of dollars)

## Scripts and Analysis
### 1. Script 1: kNN Regression with TV as Predictor
- Objective: This script applies kNN regression to predict sales based on TV advertising budgets. It demonstrates how different values of k (from 1 to 70) impact the model’s predictions and visualizes these results.

- Key Components:

    - Reading the data from Advertising.csv.
    - Splitting the data into training and testing sets.
    - Training kNN models with varying k values.
    - Plotting the regression results, showing how k affects model performance.
  
- Results:

    - The script provides insights into how increasing or decreasing k influences the accuracy of predictions.
    - A visual comparison of the predicted values with actual sales data is generated, helping to identify the optimal k.
  
### 2. Script 2: Find Nearest Neighbor Function
- Objective: This script implements a function to find the nearest neighbor for each given value of TV advertising budget and predicts sales accordingly.

- Key Components:

    - Defining a find_nearest function to locate the nearest neighbor in the training set.
    - Using this function to predict sales based on the nearest TV advertising budget.
    - Visualizing the results with a scatter plot of the original and predicted sales data.

- Results:

    - The nearest neighbor approach provides a basic understanding of the relationship between TV budgets and sales.
    - A comparison between actual and predicted values is plotted, highlighting how closely the predictions align with the data.

## 3. Script 3: Simple Scatter Plot of TV vs Sales
- Objective: This script focuses on visualizing the relationship between TV advertising budgets and sales using a simple scatter plot.

- Key Components:

    - Reading the data from Advertising.csv and displaying the first 7 rows.
    - Creating a scatter plot to visualize the correlation between TV budgets and sales.
    - Adding axis labels and titles to enhance readability.
  
- Results:

    - A clear visual representation of the TV advertising budget vs sales data.
    - Helps to understand the general trend of how TV budgets impact sales.

## Installation and Prerequisites
### 1. Environment Setup:

    - Python 3.x installed on your machine.
    - Required libraries: numpy, pandas, matplotlib, sklearn.

### 2. Data File:

- The Advertising.csv file is located in the same directory as the scripts.
