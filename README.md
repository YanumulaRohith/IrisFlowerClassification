### Iris Flower Classification

---

## Project Overview

This project involves the classification of Iris flowers into three species—Setosa, Versicolor, and Virginica—based on their sepal and petal measurements using machine learning. The dataset consists of 150 entries with 5 columns: sepal length, sepal width, petal length, petal width, and species.

The objective is to build a machine learning model capable of accurately predicting the species of Iris flowers based on the given measurements. The model is trained using the popular Iris dataset, which is a commonly used benchmark in machine learning classification tasks.

---

## Dataset

The dataset contains the following columns:

- **sepal_length**: Length of the sepal (in cm).
- **sepal_width**: Width of the sepal (in cm).
- **petal_length**: Length of the petal (in cm).
- **petal_width**: Width of the petal (in cm).
- **species**: The species of the flower (Iris-setosa, Iris-versicolor, Iris-virginica).

### Data Summary

The dataset has 150 entries with 5 columns. All values are numerical, except for the `species` column, which contains categorical labels (Iris-setosa, Iris-versicolor, Iris-virginica).

- **Total entries**: 150
- **Columns**: 5
  - 4 numerical columns (sepal length, sepal width, petal length, petal width)
  - 1 categorical column (`species`)

No missing values were found in the dataset, as confirmed by the following data checks:

```plaintext
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
```

---

## Approach

### 1. Data Preprocessing

- **Loading the Data**: The dataset was loaded using Pandas `read_csv` function.
- **Feature Selection**: The features used for classification are sepal length, sepal width, petal length, and petal width.
- **Label Encoding**: The categorical target variable `species` was converted into numeric labels (0, 1, 2).
- **Scaling**: The numerical features were scaled using StandardScaler to standardize the values and make the machine learning model more efficient.

### 2. Model Selection

Several machine learning models were evaluated to classify the Iris species:
1. **Random Forest Classifier**: Chosen for its accuracy and ability to handle classification tasks with multiple classes.
2. **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, and **Decision Trees** were considered for comparison but Random Forest yielded the best results.

### 3. Model Evaluation

The Random Forest model was trained on the data and evaluated using the following metrics:
- **Accuracy**: The model's accuracy on the test dataset was 100%.
- **Confusion Matrix**: The confusion matrix showed that the model predicted every class correctly.
- **Classification Report**: The model achieved perfect precision, recall, and f1-score for each species.

The output evaluation metrics for the model are as follows:

```plaintext
Accuracy: 1.0
Confusion Matrix:
[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

---

## Requirements

- **Python**: Version 3.6 or higher
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` (for plotting, if necessary)

To install the necessary dependencies, run:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## How to Run the Code

1. Clone this repository:

```bash
git clone https://github.com/yourusername/iris-flower-classification.git
```

2. Navigate to the project directory:

```bash
cd iris-flower-classification
```

3. Run the Python script to train and evaluate the model:

```bash
python train_model.py
```

4. Alternatively, you can open the Jupyter notebook `iris_classification.ipynb` to interactively run the code.

---

## Project Structure

```plaintext
Iris-Flower-Classification/
├── data/
│   └── iris.csv               # Dataset file
├── notebooks/
│   └── iris_classification.ipynb   # Jupyter notebook with interactive analysis
├── scripts/
│   └── train_model.py         # Python script to train and evaluate the model
└── README.md                  # Project documentation
```

---

## Conclusion

This project demonstrates the application of machine learning to classify Iris flowers into three species based on their sepal and petal measurements. The Random Forest model achieved 100% accuracy on the test set. The dataset was clean, and the preprocessing steps ensured the data was in a format suitable for classification tasks. This task provided an introduction to classification, data preprocessing, and model evaluation in machine learning.

