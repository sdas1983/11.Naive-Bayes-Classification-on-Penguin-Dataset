# Naive-Bayes-Classification-on-Penguin-Dataset

## Overview
This project demonstrates the application of the Naive Bayes classification algorithm on the Palmer Penguins dataset. The dataset includes physical measurements of penguins such as culmen length, culmen depth, flipper length, and body mass, along with categorical variables like species, island, and sex. The goal of this project is to classify penguin species based on these attributes.

## Dataset
The dataset used in this project is the Palmer Penguins dataset. It contains the following features:

island: Island name (Biscoe, Dream, or Torgersen)
culmen_length_mm: Culmen length (mm)
culmen_depth_mm: Culmen depth (mm)
flipper_length_mm: Flipper length (mm)
body_mass_g: Body mass (g)
sex: Gender (male or female)
species: Penguin species (Adelie, Chinstrap, or Gentoo)

## Data Preprocessing
Handled missing values by removing rows with null entries.
Encoded categorical variables using ordinary encoding.
Standardized the numerical features using StandardScaler.

## Exploratory Data Analysis
Several visualizations were created to explore the data:

Scatter Plots: Visualize relationships between features with species as the hue.
Pair Plots: Provide a grid of scatter plots for all feature pairs.
Box Plots: Visualize the distribution of culmen length across species and sex.

## Model Training
The Naive Bayes classifier was trained using the preprocessed data. The dataset was split into training and testing sets with a 65-35 split.

### Steps Involved:
1. Data Splitting: Split the data into training and testing sets.
2. Model Training: Train a Gaussian Naive Bayes classifier on the training data.
3. Predictions: Generate predictions on the test data.
4. Evaluation: Evaluate the model's performance using accuracy, confusion matrix, and ROC curves.

## Model Evaluation
The model's performance was evaluated using the following metrics:

- Accuracy Score: Measures the overall accuracy of the classifier.
- Confusion Matrix: Provides insights into the classification performance across all classes.
- ROC Curve: Used to evaluate the model's performance in distinguishing between classes.

## Hyperparameter Tuning
Hyperparameters were tuned using GridSearchCV to find the best model parameters, and the results were evaluated using a multiclass ROC curve.

## Saving and Loading the Model
The trained Naive Bayes model was saved using the pickle library, allowing for easy reuse and deployment.

## Conclusion
This project demonstrates how to preprocess data, train a Naive Bayes classifier, evaluate its performance, and save the model for future use. It also shows how to create visualizations to better understand the data and model performance.
