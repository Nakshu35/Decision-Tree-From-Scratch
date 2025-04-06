# 🌳 Decision Tree from Scratch

This repository contains an implementation of a **Decision Tree algorithm** from scratch in Python. It supports both **classification** and **regression** tasks, using **entropy** and **information gain** for classification, and **variance** and **mean squared error (MSE)** for regression.

## 📁 File Structure

- `Decision_Tree.py` – Main implementation of the Decision Tree algorithm.
  - Supports both classification and regression.
  - Uses entropy and information gain for classification.
  - Uses variance and MSE for regression.
- `Implementation_DT.ipynb` – Example usage of Decision_Tree.py.

## ✅ Features

- [x] No external ML libraries like scikit-learn.
- [x] Custom implementation of entropy and information gain.
- [x] Custom splitting logic for regression using variance and MSE.
- [x] Handles both classification and regression.
- [x] Simple and modular code for easy understanding and extension.

## ⚙️ How It Works

### Classification
- Calculates **entropy** for each feature split.
- Chooses the feature with the highest **information gain**.
- Recursively splits the data until stopping criteria are met.

### Regression
- Evaluates splits based on both **variance reduction** and **mean squared error (MSE)**.
- Selects the best split by minimizing the overall error.
- Recursively builds the tree to fit continuous target variables.

## 🧪 Example Usage

```python
from Decision_Tree import DecisionTree_Classifier, DecisionTree_Regressor

# Example for classification
clf = DecisionTree_Classifier(max_depth=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Example for regression
reg = DecisionTree_Regressor(max_depth=5)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
