# 🏠 House Prices Prediction – Step-by-Step Code Breakdown

This project walks through the process of preparing a dataset, training a model, and making predictions — all while explaining how and why each step is done. Let's break down the code into meaningful sections.


---

## 📊 Training Set vs Test Set: What's the Difference?

| Term        | Description |
|-------------|-------------|
| **Training Set** | The data that we use to teach the model. It contains both the **features** (inputs) and the **target** (`SalePrice`) that we want to predict. |
| **Test Set**     | Data the model has never seen before. It has all the features, but it **does NOT contain the target**. We use it to evaluate how well the model can generalize to unseen data. |

Think of it like studying for an exam:
- **Training data** is your study material (with answers).
- **Test data** is the exam (no answers until grading).

---

## 📦 Part 1: Import Libraries and Load Data
upload the data, search'kaggle/input/house-prices-advanced-regression-techniques/train.csv' and then add in your notebook. 

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline
```

### 💡 Explanation:
We import all the essential Python libraries for data handling, visualization, and model building.

---

## 📥 Part 2: Load and Prepare the Dataset

```python
# Load training and test datasets
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Combine train and test sets for unified preprocessing
full_data = pd.concat([train_df.drop(columns=['SalePrice']), test_df], ignore_index=True)

# Drop 'Id' column
for df in [train_df, test_df, full_data]:
    df.drop(['Id'], axis=1, inplace=True)

# Drop columns with too many missing values
drop_cols = ['LotFrontage']
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)
full_data.drop(columns=drop_cols, inplace=True)
```

### 💡 Explanation:
- We load the datasets.
- Combine them into `full_data` to apply the same cleaning steps to both.
- Drop unnecessary columns like `Id` and `LotFrontage`.

---

## 🧹 Part 3: Handle Missing Values

```python
# Fill missing numeric values with column mean
def fill_numeric_with_mean(df, columns):
    for col in columns:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

# Fill missing categorical values with most frequent value
def fill_categorical_with_mode(df, columns):
    for col in columns:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Get column types
numeric_cols = full_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = full_data.select_dtypes(include='object').columns

# Apply filling
fill_numeric_with_mean(full_data, numeric_cols)
fill_categorical_with_mode(full_data, categorical_cols)

# Update train and test datasets from full_data
train_df.update(full_data.loc[:train_df.shape[0]-1])
test_df.update(full_data.loc[train_df.shape[0]:].reset_index(drop=True))

# Confirm all missing values are handled
print("Remaining nulls in train:", train_df.isnull().sum().sum())
print("Remaining nulls in test:", test_df.isnull().sum().sum())
```

### 💡 Explanation:
- We create functions to handle missing data.
- Missing numeric values are filled with the mean.
- Missing categorical values are filled with the mode.
- After processing, we update the original train and test data.

---

## 🛠️ Part 4: Feature Engineering

```python
# Log-transform the target for better modeling
train_df["LogPrice"] = np.log(train_df["SalePrice"])

# Function to add new features
def add_features(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Bathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['Total_porch_sf'] = (
        df['OpenPorchSF'] + df['3SsnPorch'] +
        df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    )
    df['haspool'] = (df['PoolArea'] > 0).astype(int)
    df['has2ndfloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['hasgarage'] = (df['GarageArea'] > 0).astype(int)
    df['hasbsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['hasfireplace'] = (df['Fireplaces'] > 0).astype(int)
    return df

# Apply feature engineering
train_df = add_features(train_df)
test_df = add_features(test_df)

# Detect outliers (optional)
outliers = train_df[train_df["SalePrice"] > 600000]
```

### 💡 Explanation:
- We **log-transform** the sale price to reduce skew.
- We **engineer new features** that might help the model predict better.
- Outliers are optionally checked for further analysis.

---

## 🧠 Part 5: Encode Categorical Variables

```python
# Function to label encode categorical columns
def label_encode(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category').cat.codes
    return df

# Get categorical columns
categorical_values_train = train_df.select_dtypes(include='object').columns.tolist()
categorical_values_test = test_df.select_dtypes(include='object').columns.tolist()

# Apply encoding
train_df = label_encode(train_df, categorical_values_train)
test_df = label_encode(test_df, categorical_values_test)
```

### 💡 Explanation:
- Machine learning models require numeric data, so we convert text labels into numbers using label encoding.

---

## 🤖 Part 6: Train a Decision Tree Model

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Prepare feature matrix and target
X = train_df.drop(["SalePrice", "LogPrice"], axis=1)
y = train_df["LogPrice"]

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Check and manually fix any nulls
print("Nulls in training set:\n", X_train.isnull().sum()[X_train.isnull().sum() > 0])
X_train.loc[1379, "Electrical"] = 5  # Example fix

# Define model parameters for Grid Search
params = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "splitter": ["best", "random"],
    "min_samples_split": [2, 3, 5, 10],
    "max_features": ["auto", "log2"]
}

# Grid search for best model
model = DecisionTreeRegressor(random_state=100)
grid = GridSearchCV(model, params, verbose=1, scoring="r2")
grid.fit(X_train, y_train)

# Output best model and score
print("Best Model:", grid.best_estimator_)
print("Best Score:", grid.best_score_)
```

### 💡 Explanation:
- We split the data again into **training** and **validation** sets to check model performance.
- A **Decision Tree Regressor** is trained using **GridSearchCV** to find the best combination of parameters.

---

## 📤 Part 7: Make Predictions and Submit

```python
# Align test dataset columns with training data
test_df = test_df[X_train.columns]

# Predict on test data
predictions = grid.predict(test_df)

# Create submission file
test_ids = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')['Id']
output = pd.DataFrame({'Id': test_ids, 'SalePrice': np.exp(predictions)})
output.to_csv('/kaggle/working/submission.csv', index=False)

output.head()
```

### 💡 Explanation:
- We ensure the test dataset has the same columns as the training set.
- Predict house prices using the trained model.
- Save the predictions in a `.csv` file formatted for Kaggle submission.

---

## ✅ Summary

- We cleaned and preprocessed the data.
- We explored the difference between **training** and **test** sets.
- We engineered features, handled missing values, and encoded categorical data.
- We trained and optimized a **Decision Tree model**.
- Finally, we made predictions and created a submission file.

