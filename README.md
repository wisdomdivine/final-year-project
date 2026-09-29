# final-year-project

Pore Pressure Prediction in Well 15/9-F-12 Using Machine Learning

## Overview

This repository contains the dataset documentation, preprocessing workflows, and machine learning models for predicting pore pressure (PP) in well 15/9-F-12 (Volve Field, Norwegian North Sea).

The project evaluates four machine learning algorithms across an independently reconstructed well-log dataset comprising 1,811 observations spanning the depth interval 3127.71 m to 3403.55 m.

## Modeling Framework

The study benchmarks four regression models representing different algorithmic families:

1. Multiple Linear Regression (MLR): Linear baseline
2. Random Forest Regression (RF): Bagged decision tree ensemble
3. CatBoost Regression: Gradient boosted decision trees
4. Multilayer Perceptron (MLP): Artificial neural network

Performance evaluation is conducted using four standard metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R-squared)

## Data Partitioning Strategy

To account for vertical spatial autocorrelation in well logs, the data is partitioned into contiguous depth blocks rather than random shuffling:

- Training Set (70%): Shallower interval (~1,268 observations)
- Validation Set (15%): Middle interval (~272 observations)
- Testing Set (15%): Deepest interval (~271 observations)

All data preprocessing parameters (such as feature scaling) are fitted strictly on the training set and applied forward to validation and test sets to eliminate data leakage.

## Repository Contents

- `README.md`: Project summary and framework outline
- `DATA_SUMMARY.md`: Predictor dataset verification, feature mappings, and split partitions

Further implementation scripts and modeling code will be added as workflows progress.
