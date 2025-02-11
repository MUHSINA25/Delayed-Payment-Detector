# Delayed Payment Detector

## Overview

Delayed Payment Detector is a machine learning-based solution designed to predict whether an invoice payment will be delayed. This project leverages various classification models, preprocessing techniques, and oversampling methods to handle imbalanced data effectively.

## Features

- Predicts delayed payments based on historical invoice data
- Supports multiple machine learning models (Random Forest, SVM, Gradient Boosting, etc.)
- Uses **SMOTE** to handle class imbalance
- Automated data preprocessing and feature engineering
- Hyperparameter tuning with **GridSearchCV**
- Saves trained models for future predictions

## Dataset

The model uses invoice-related data, including:

- **Receipt Date** and **Due Date** (processed into year, month, and day features)
- **Invoice details** such as amount and customer-related features
- **Target Variable**: `Delayed_Payment` (1 if payment is delayed, 0 otherwise)


## Usage

### Training the Model

```python
from payment_pipeline import PaymentPredictionPipeline

pipeline = PaymentPredictionPipeline("dataset.csv")
pipeline.load_data()
pipeline.split_data()
pipeline.build_pipeline(model_name="Random Forest")
pipeline.train_and_evaluate()
```

### Predicting on Unseen Data

```python
predictions = pipeline.predict_unseen_data("unclassify.csv")
print(predictions)
```

## Model Performance

- The classification report provides precision, recall, and F1-score
- The best model is saved as `payment_prediction_pipeline.pkl`

## Future Improvements

- Integration with a real-time invoice processing system
- Enhanced feature selection and engineering
- Experimentation with deep learning models

