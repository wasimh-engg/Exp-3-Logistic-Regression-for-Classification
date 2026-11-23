# Experiment 3: Logistic Regression for Classification

## Aim

To classify data using Logistic Regression and evaluate the model using
accuracy, confusion matrix, and classification report.

## Algorithm

1.  **Start.**

2.  Load the dataset and extract features `X` and target `y`.

3.  Split the data into training and test sets (e.g., 80:20).

4.  Standardize features using `StandardScaler`.

5.  Train `LogisticRegression` on the training data.

    The logistic (sigmoid) function is:

    σ(z) = 1 / (1 + e^(−z))

    where

    z = β₀ + β₁x₁ + ⋯ + βₚxₚ

6.  Predict labels on the test data.

7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.

8.  Print the evaluation metrics.

9.  **Stop.**

## Program

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset and extract features X and target y
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print("Dataset shape:", X.shape)
print("Classes:", {n: int(c) for n, c in zip(data.target_names, np.bincount(y))})

# Step 3: Split into training and test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", X_train.shape[0], "| Test samples:", X_test.shape[0])

# Step 4: Standardize features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Step 5: Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

# Step 6: Predict labels on test data
y_pred = model.predict(X_test_s)

# Steps 7-8: Evaluate and print metrics
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))
```

## Output

```
Dataset shape: (569, 30)
Classes: {'malignant': 212, 'benign': 357}
Training samples: 455 | Test samples: 114

Accuracy: 0.9825

Confusion Matrix:
 [[41  1]
 [ 1 71]]

Classification Report:
               precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

## Result

The program was executed successfully. A Logistic Regression model was
trained on the standardized training data (455 samples) and tested on 114
unseen samples.

-   The model achieved an **accuracy of 98.25%**.
-   The confusion matrix shows 41 malignant and 71 benign cases were
    classified correctly, with only 1 misclassification in each class.
-   Precision, recall and F1-score are about 0.98 to 0.99 for both
    classes, so the model performs well on both.

Thus, Logistic Regression can be used effectively for binary
classification.
