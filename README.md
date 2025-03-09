# Student Performance Predictor

## Overview

This project aims to predict student performance based on various factors such as **Gender**, **Ethnicity**, **Parental Level of Education**, **Lunch**, and **Test Preparation Course**. Using a regression model, the project will predict the **average score** of a student by analyzing these features. The dataset is sourced from Kaggle and contains information about students' scores in different subjects.

**Dataset**: The dataset used to predict student performance is collected from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977).

--

## Data Understanding

The dataset contains information related to student performance, including test scores and other features. The data is structured as follows:

- **Samples**: 1000
- **Features**: 8 columns
- **Target Column**: 1 column (Average score)

### Features:

1. **Gender**: Student's gender (Male/Female)  
   Type: `str`

2. **Race/Ethnicity**: Ethnicity of students (Group A, B, C, D, E)  
   Type: `str`

3. **Parental Level of Education**: The highest education level of the student's parents (e.g., Bachelor's degree, Some college, Master's degree, Associate's degree, High school)  
   Type: `str`

4. **Lunch**: Whether the student had a standard lunch or a free/reduced lunch before the test  
   Type: `str`

5. **Test Preparation Course**: Whether the student completed the test preparation course (Completed/Not Completed)  
   Type: `str`

6. **Math Score**: Score achieved in the math exam  
   Type: `int`

7. **Reading Score**: Score achieved in the reading exam  
   Type: `int`

8. **Writing Score**: Score achieved in the writing exam  
   Type: `int`

### Target Column:

- **Average**: The average score across all subjects (Math, Reading, and Writing)  
  Type: `int`

--

## Objective

This project is focused on predicting a student’s **average exam score** based on the above features. The task is modeled as a **regression problem** where the objective is to predict a continuous value (average score). The prediction will be done using machine learning algorithms such as Linear Regression, Random Forest, etc.

--

## Type of Machine Learning Task

The problem is a **regression** task, where given a set of features, we predict the **average score** of a student across different subjects.

--

## Performance Metrics

Since this is a **regression problem**, the following performance metrics will be used to evaluate the model's accuracy:

- **R² Score**: A statistical measure of how well the regression predictions approximate the real data points.
- **Root Mean Squared Error (RMSE)**: A measure of the differences between predicted and observed values, representing how far off the predictions are.
- **Mean Squared Error (MSE)**: The average of the squared differences between the predicted and actual values, another measure of prediction accuracy.

--

## Libraries and Tools

The following libraries and tools are used in this project:

- **Python**: Programming language for data processing and model building.
- **Seaborn**: Data visualization library for statistical plotting.
- **Pandas**: Data manipulation library to handle datasets.
- **NumPy**: Numerical operations library for handling arrays and matrices.
- **scikit-learn**: Machine learning library for building, training, and evaluating models.

--

## Installation

To set up the environment for this project, you can follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Pranav-Code-007/student-performance-predictor.git
    cd student-performance-predictor
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

   The `requirements.txt` should include the following dependencies:
    ```bash
    pandas
    numpy
    seaborn
    matplotlib
    scipy
    scikit-learn
    ```
