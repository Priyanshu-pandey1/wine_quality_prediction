# 🍷 Wine Quality Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a machine learning project that classifies red wine as **"Good Quality"** or **"Bad Quality"** based on its chemical properties. A `RandomForestClassifier` is trained on the *Red Wine Quality* dataset from the UCI Machine Learning Repository to perform this classification.

---

## 🎯 Problem Statement

The original dataset rates wine quality on a scale from 1 to 10. To frame this as a classification problem (which is what this project does), we perform **Label Binarization**.

The 'quality' column is transformed into a binary target variable `Y`:
* **Good Quality (1):** Wine with a 'quality' score of 7 or higher.
* **Bad Quality (0):** Wine with a 'quality' score below 7.

The goal is to train a model that can accurately predict this binary label for a new wine sample.

---

## 📋 Project Workflow

1.  **Import Dependencies:** Load necessary libraries (`numpy`, `pandas`, `seaborn`, `sklearn`).
2.  **Data Collection:** Load the `winequality-red.csv` dataset.
3.  **Data Analysis & Visualization:**
    * Explore dataset statistics using `describe()`.
    * Check for missing values.
    * Visualize the distribution of the 'quality' variable.
    * Analyze the relationship between features (like 'volatile acidity', 'citric acid') and 'quality'.
    * Create a correlation heatmap to understand feature relationships.
4.  **Data Preprocessing:**
    * Separate features (`X`) from the target variable ('quality').
    * Apply **Label Binarization** to create the new target `Y` (Good/Bad).
5.  **Train & Test Split:** Split the data into training (80%) and testing (20%) sets.
6.  **Model Training:** Train a `RandomForestClassifier` on the training data.
7.  **Model Evaluation:** Evaluate the model's performance on the unseen test data using the **Accuracy Score**.
8.  **Predictive System:** Build a simple function to take new data and predict its quality.

---

## 🛠️ Technologies Used

* **Python**
* **Scikit-learn:** For model training (`RandomForestClassifier`), splitting (`train_test_split`), and evaluation (`accuracy_score`).
* **Pandas:** For data loading and manipulation.
* **NumPy:** For numerical operations and data reshaping.
* **Matplotlib & Seaborn:** For data visualization and creating plots (bar plots, heatmaps).

---

## 📊 Dataset

The model is trained on the **Red Wine Quality** dataset.

### Features (X)
1.  `fixed acidity`
2.  `volatile acidity`
3.  `citric acid`
4.  `residual sugar`
5.  `chlorides`
6.  `free sulfur dioxide`
7.  `total sulfur dioxide`
8.  `density`
9.  `pH`
10. `sulphates`
11. `alcohol`

### Target Variable (Y)
* `quality`: (Binarized)
    * **1** = Good Quality (Score 7, 8, 9, 10)
    * **0** = Bad Quality (Score 1, 2, 3, 4, 5, 6)

---

## 🤖 Model Performance

A **Random Forest Classifier** was used for this classification task.

* **Metric:** Accuracy Score
* **Test Data Accuracy:** **[INSERT YOUR ACCURACY SCORE HERE, e.g., 91.5%]**

This accuracy score indicates the percentage of wines in the test set that the model correctly classified as "Good Quality" or "Bad Quality".

---

## 🚀 How to Use the Predictive System

You can use the trained model to predict the quality of a new, unseen wine sample.

1.  Define your input data as a tuple or list containing the 11 features in the correct order:
    ```python
    # (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
    
    input_data = (7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4)
    ```

2.  The script will process this data, make a prediction, and print the result:
    ```
    Prediction: [1]
    Good Quality Wine
    ```


