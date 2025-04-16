# 🏠 House Prices Prediction with Machine Learning

This project predicts house prices using a Decision Tree Regressor trained on the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset. It includes data cleaning, feature engineering, model training, and submission creation — all explained step-by-step for beginners.

---

## 📌 What You’ll Learn

- How to handle missing values
- The difference between a **training set** and a **test set**
- How to engineer new features
- How to encode categorical data
- How to train and tune a Decision Tree model
- How to generate predictions and submit to Kaggle

---

## 📁 Dataset Info

The dataset contains housing data from Ames, Iowa, including 79 explanatory variables describing (almost) every aspect of residential homes.

- `train.csv` contains the data with known sale prices.
- `test.csv` contains the data you will predict for.
- The target variable is `SalePrice`.

---

## 🧠 What's the Difference Between the Training Set and Test Set?

- **Training Set**: This is the data you use to train your model. It includes the features (`X`) and the target variable (`SalePrice`).
- **Test Set**: This is the data without the target variable. You use your trained model to predict `SalePrice` on this set.

The key idea is:  
👉 Train your model on the **training set** and evaluate how well it generalizes to **unseen data**, i.e., the **test set**.

---

## 🛠️ Project Structure

```bash
.
├── data/
│   ├── train.csv
│   └── test.csv
├── house_price_prediction.ipynb
├── submission.csv
└── README.md
