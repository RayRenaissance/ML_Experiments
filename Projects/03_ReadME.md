# Project 03: Ensemble Learning and Model Comparison

This project focuses on predicting customer churn using the "Telco-Customer-Churn.csv" dataset. The primary goal is to train and compare the performance of multiple advanced ensemble models, especially after addressing the dataset's significant class imbalance.

## 🚀 Project Workflow

1.  **Data Loading & Inspection:**
    * The `Telco-Customer-Churn.csv` dataset is loaded into a pandas DataFrame.
    * Initial analysis reveals 7043 entries and 21 columns.
    * The target variable, `Churn`, is imbalanced: **No (5174)** vs. **Yes (1869)**.
    * The `TotalCharges` column is identified as an `object` type with potential missing values (blanks).

2.  **Data Preprocessing:**
    * **Missing Values:** `TotalCharges` is converted to a numeric type, with errors coerced to `NaN`. These `NaN` values are then imputed using the median of the column.
    * **Encoding:**
        * `LabelEncoder` is used to transform all categorical columns (e.g., `gender`, `Partner`, `PaymentMethod`) into numerical values.
        * The target variable `Churn` is also encoded (`No`=0, `Yes`=1).
    * **Feature Scaling:** `StandardScaler` is applied to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to normalize their range.

3.  **Handling Class Imbalance (SMOTE):**
    * The data is split into training and testing sets (80/20 split).
    * The initial training data reflects the imbalance (Class 0: 4138, Class 1: 1496).
    * **SMOTE (Synthetic Minority Over-sampling Technique)** is applied *only to the training set* to create a balanced dataset for model training (Class 0: 4138, Class 1: 4138).

4.  **Ensemble Model Training:**
    * Four different ensemble models are trained on the **resampled (balanced)** training data:
        * **RandomForestClassifier** (Bagging)
        * **XGBClassifier** (Boosting)
        * **LGBMClassifier** (Boosting)
        * **CatBoostClassifier** (Boosting)

5.  **Model Evaluation & Comparison:**
    * All models are evaluated on the original, *unseen* `X_test` and `y_test` data to get a realistic performance benchmark.
    * Performance is measured using the `classification_report` (precision, recall, f1-score) and the **ROC-AUC score**.

## 📊 Results

The models were compared based on their overall accuracy and their ability to discriminate between classes (ROC-AUC).

| Model | Accuracy | F1-Score (Class 1) | ROC-AUC Score |
| :--- | :---: | :---: | :---: |
| Random Forest | 0.78 | 0.59 | 0.723 |
| XGBoost | 0.79 | 0.61 | 0.838 |
| **LightGBM** | **0.80** | **0.62** | **0.844** |
| CatBoost | 0.79 | 0.61 | 0.843 |

### Conclusion

**LightGBM (LGBM) performed the best overall**. It achieved the highest accuracy (0.80) and the highest ROC-AUC score (0.844), indicating it was the most effective model at both correctly classifying customers and distinguishing between churners and non-churners.

## 🛠️ Requirements

The project relies on the following Python libraries:
* `pandas`
* `scikit-learn` (sklearn)
* `imbalanced-learn` (imblearn)
* `xgboost`
* `lightgbm`
* `catboost`