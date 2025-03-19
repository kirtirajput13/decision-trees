# Decision Tree Classification on Iris Dataset

## Overview
This project demonstrates the implementation of a Decision Tree classifier using the popular Iris dataset. The goal is to classify iris flowers into one of three species (Setosa, Versicolor, or Virginica) based on four features: sepal length, sepal width, petal length, and petal width. The project covers the entire data science workflow, including data loading, exploratory data analysis (EDA), model training, evaluation, and visualization.

## Dataset
The Iris dataset is a classic dataset in machine learning and statistics. It contains 150 samples of iris flowers, with 50 samples from each of the three species. Each sample has four features:
- Sepal length (in cm)
- Sepal width (in cm)
- Petal length (in cm)
- Petal width (in cm)

The target variable is the species of the iris flower, which is one of the following:
- Setosa
- Versicolor
- Virginica

## Project Steps
1. **Data Loading and Preprocessing**:
   - The dataset is loaded using the `load_iris()` function from the `sklearn.datasets` module.
   - The data is converted into a pandas DataFrame for easier manipulation and visualization.

2. **Exploratory Data Analysis (EDA)**:
   - A pairplot is created to visualize the relationships between the features and how they differ across the three species.
   - This step helps in understanding the data and identifying patterns or trends.

3. **Train-Test Split**:
   - The dataset is split into training and testing sets using a 70-30 split. This ensures that the model is trained on one portion of the data and evaluated on another, unseen portion.

4. **Model Training**:
   - A Decision Tree classifier is initialized with the `criterion='entropy'` parameter, which maximizes information gain during splits.
   - The model is trained on the training data using the `fit()` method.

5. **Model Evaluation**:
   - The model's accuracy is calculated using the `accuracy_score()` function.
   - A classification report is generated, which includes precision, recall, and F1-score for each class.
   - A confusion matrix is plotted to visualize the true vs. predicted labels.

6. **Decision Tree Visualization**:
   - The decision tree is visualized using the `plot_tree()` function from the `sklearn.tree` module. This provides a graphical representation of the decision-making process of the model.

7. **Feature Importance**:
   - The importance of each feature in the decision tree is calculated and visualized using a bar plot. This helps in understanding which features contribute the most to the model's predictions.

## Results
- **Accuracy**: The model achieves an accuracy of approximately 0.96, meaning it correctly classifies 96% of the test samples.
- **Classification Report**: The report shows high precision, recall, and F1-scores for all three species, indicating that the model performs well across all classes.
- **Confusion Matrix**: The confusion matrix shows that most samples are correctly classified, with only a few misclassifications.
- **Decision Tree Visualization**: The decision tree is visualized, showing the splits and decisions made by the model.
- **Feature Importance**: The bar plot indicates that petal length and petal width are the most important features for classification.

## How to Run the Code
1. Open the Google Colab notebook provided in this repository.
2. Run the code cell by cell to see the outputs.
3. The notebook includes comments and explanations for each step, making it easy to follow along.

## Dependencies
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Conclusion
This project provides a comprehensive example of how to build and evaluate a Decision Tree classifier using the Iris dataset. It covers all the essential steps in a data science workflow, from data loading and preprocessing to model training, evaluation, and visualization. The results demonstrate the effectiveness of the Decision Tree algorithm for classification tasks and provide insights into the importance of different features in the dataset.

link for google colab file: https://colab.research.google.com/drive/1qt_qtYh-O7-wDkvRSybZfOV454pIB895?usp=sharing
