# Improves feature engineering. Validation R² should around 0.9. Final score = 0.13336
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

# Load data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Save target and log transform
y = np.log1p(train_df['SalePrice'])

# Drop ID and target from train
train_df.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test_ids = test_df['Id']
test_df.drop(['Id'], axis=1, inplace=True)

# Combine datasets for consistent processing
full_data = pd.concat([train_df, test_df], axis=0)

# === Feature Engineering ===
def add_features(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Bathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    return df

full_data = add_features(full_data)

# Drop some less useful or highly missing columns
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage']
full_data.drop(columns=drop_cols, inplace=True)

# Fill NA values
# Numeric: fill with median
numeric_cols = full_data.select_dtypes(include=['number']).columns
for col in numeric_cols:
    full_data[col].fillna(full_data[col].median(), inplace=True)

# Categorical: fill with mode
categorical_cols = full_data.select_dtypes(include='object').columns
for col in categorical_cols:
    full_data[col].fillna(full_data[col].mode()[0], inplace=True)

# One-Hot Encoding
full_data = pd.get_dummies(full_data, drop_first=True)

# Split back into train/test
X = full_data.iloc[:len(y), :]
X_test = full_data.iloc[len(y):, :]

# Split train into train/validation
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit XGBoost
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import r2_score
val_preds = model.predict(X_valid)
print("Validation R²:", r2_score(y_valid, val_preds))

# Predict on test set
final_preds = model.predict(X_test)
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': np.expm1(final_preds)  # Reverse log1p
})
submission.to_csv('submission.csv', index=False)
Validation R²: 0.9027962152645228
