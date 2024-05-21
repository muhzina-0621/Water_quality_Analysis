# Water Quality Analysis using Python

This program predicts and classifies the potability of water.

## Model Descriptions

### 1. Support Vector Machine (SVM):

#### svm_model.py

**Steps:**

- **Importing Libraries:** 
  - pandas, numpy, matplotlib, and sklearn.
- **Loading and Preprocessing Data:**
  - Load the dataset and fill missing values with column means.
- **Preparing Features and Target Variable:**
  - Extract features (x) and target variable (y).
- **Splitting Data:**
  - Split the data into training and testing sets (80-20 split).
- **Training the SVM Model:**
  - Initialize and train an SVM classifier with a linear kernel.
- **Making Predictions and Evaluating the Model:**
  - Predict the target variable for the test set.
  - Compute and display the confusion matrix.
  - Print the classification report and accuracy score.

### 2. Gaussian Naive Bayes:

#### naive_bayes_model.py

**Steps:**

- **Importing Libraries:** 
  - pandas, numpy, matplotlib, and sklearn.
- **Loading and Preprocessing Data:**
  - Load the dataset and fill missing values with column means.
- **Preparing Features and Target Variable:**
  - Extract features (x) and target variable (y).
- **Splitting Data:**
  - Split the data into training and testing sets (80-20 split).
- **Training the Gaussian Naive Bayes Model:**
  - Initialize and train a Gaussian Naive Bayes classifier.
- **Making Predictions and Evaluating the Model:**
  - Predict the target variable for the test set.
  - Compute and display the confusion matrix.
  - Print the classification report and accuracy score.

## Results

The performance of each model is evaluated using accuracy, confusion matrix, and classification report. The results are displayed in the console and as plots.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss any changes or improvements.

