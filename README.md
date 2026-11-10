# Fraud Detection with PySpark

## Project Overview

This project uses PySpark and Spark MLlib to prepare transaction data and build a machine learning model for fraud detection.

The workflow covers data quality checks, exploratory analysis, feature transformation, and model evaluation. It demonstrates distributed data processing and highlights the challenges of detecting fraud in an imbalanced dataset.

## Business Objectives

- Identify patterns associated with fraudulent transactions.
- Build a reproducible data preparation workflow.
- Train a classification model to distinguish fraudulent and legitimate transactions.
- Assess false positives and missed fraud cases to understand operational trade-offs.

## Dataset

The dataset contains transaction characteristics such as:

| Feature | Description |
|---------|-------------|
| distance_from_home | Distance between the transaction location and the customer's home |
| distance_from_last_transaction | Distance from the previous transaction |
| ratio_to_median_purchase_price | Transaction amount relative to the median purchase amount |
| repeat_retailer | Whether the retailer was previously used |
| used_chip | Whether the transaction used a chip |
| used_pin_number | Whether the transaction used a PIN |
| online_order | Whether the transaction was made online |
| fraud | Target label: fraudulent or legitimate transaction |

Approximately 8.7% of the transactions were labelled as fraudulent, creating a class imbalance that required attention during evaluation.

## Tools and Technologies

- Python
- PySpark
- Spark SQL
- Spark MLlib
- Google Colab
- Data visualisation libraries

## Project Workflow

### 1. Data Cleaning and Validation

- Inspected the dataset structure and column types.
- Checked for missing values.
- Handled incomplete records.
- Verified numerical and binary variables before modelling.

### 2. Exploratory Data Analysis

- Examined the distribution of transaction characteristics.
- Compared means, medians, and standard deviations.
- Assessed the proportion of fraudulent transactions.
- Explored patterns in transaction distances and relative purchase amounts.

### 3. Feature Engineering and Preprocessing

- Assembled numerical variables into feature vectors.
- Applied StandardScaler to distance-related features.
- Applied scaling transformations to the purchase-price ratio.
- Combined transformed and binary variables into a final feature vector.
- Split the data into training and test sets using an 80/20 split and a fixed random seed.

### 4. Model Development

Trained a Random Forest classifier with Spark MLlib to predict the fraud label from the prepared transaction features.

### 5. Model Evaluation

Evaluated predictions using a confusion matrix, classification metrics, and an ROC curve.

## Results

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraudulent |
|--------------------|-----------:|-----------:|
| Legitimate | 94,037 | 4,099 |
| Fraudulent | 2,310 | 6,914 |

### Fraud-Class Metrics

Calculated from the confusion matrix above:

| Metric | Result |
|--------|-------:|
| Overall accuracy | 94.03% |
| Fraud precision | 62.78% |
| Fraud recall | 74.96% |
| Fraud F1-score | 68.33% |

The model detected approximately 75% of the fraudulent transactions in the test set. However, it also flagged 4,099 legitimate transactions and missed 2,310 fraudulent transactions.

These results show why overall accuracy alone is insufficient for evaluating fraud detection.

## Key Insights

- Fraud represented a minority of transactions, making class-specific evaluation essential.
- Transaction distances and purchase-price ratios showed substantial variation and required careful preprocessing.
- The model identified many fraudulent transactions, but false alerts and missed fraud remained material.
- PySpark supported data preparation, feature transformation, and model training within the same processing environment.

## Business Recommendations

- Use model predictions to support investigation and prioritisation of suspicious transactions.
- Adjust the classification threshold according to the cost of missed fraud and the team's capacity to review alerts.
- Investigate class weighting and alternative models to improve fraud detection.
- Monitor precision, recall, and changes in transaction patterns over time.
- Validate the model on new, time-separated data before considering operational deployment.

## Limitations and Future Improvements

- The project was developed in Google Colab; processing larger datasets on a production cluster was not benchmarked.
- A random train/test split does not establish performance on future transactions.
- Scaling is not required for Random Forest models, although these transformations can support comparisons with other algorithms.
- Further work could include hyperparameter tuning, threshold optimisation, and an integrated Spark ML pipeline.
- Production use would require additional validation, monitoring, and deployment work.

## Author

Aram Samb
