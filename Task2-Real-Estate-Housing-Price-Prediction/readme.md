# Real Estate Housing Price Prediction 🏡

A Machine Learning regression project that predicts median house values in California using housing and demographic data. This project compares Ordinary Least Squares (OLS), Ridge Regression (L2), and Lasso Regression (L1) to evaluate their predictive performance.

## 📌 Project Overview

The objective is to build and compare regression models that estimate median house values based on features such as median income, housing age, number of rooms, population, geographical coordinates, and ocean proximity.

The project demonstrates data preprocessing, feature engineering, regularization, model evaluation, and visualization using Python.

## 🎯 Objectives

- Load and explore the California Housing dataset.
- Identify missing values and analyze feature correlations.
- Handle missing numerical and categorical values using imputation.
- Engineer meaningful features and pairwise interactions.
- Encode categorical variables using One-Hot Encoding.
- Standardize numerical features.
- Compare OLS, Ridge, and Lasso regression models.
- Tune Ridge and Lasso hyperparameters using GridSearchCV.
- Evaluate performance using RMSE, MAE, and R².
- Visualize actual versus predicted values and residual distributions.

## 📊 Dataset

**Source:** [California Housing Dataset — GitHub](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)

The dataset contains 20,640 observations and 10 columns, including the target variable `median_house_value`.

### Features

- `longitude` — Geographic longitude
- `latitude` — Geographic latitude
- `housing_median_age` — Median age of housing units
- `total_rooms` — Total rooms in the block group
- `total_bedrooms` — Total bedrooms in the block group
- `population` — Population of the block group
- `households` — Number of households
- `median_income` — Median income in the block group
- `ocean_proximity` — Categorical indicator of proximity to the ocean

**Target:** `median_house_value`

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Statsmodels
- PyCharm

## ⚙️ Methodology

### 1. Data Exploration
- Inspect dataset structure, descriptive statistics, missing values, and duplicates.
- Generate a correlation heatmap to examine relationships between numerical features.

### 2. Data Preprocessing
- Split the dataset into training and testing sets.
- Impute missing numerical values using the median.
- Impute missing categorical values using the most frequent category.
- Encode categorical features using One-Hot Encoding.
- Standardize numerical features using StandardScaler.

### 3. Feature Engineering
Create domain-informed ratio features:

- Rooms per household
- Bedrooms per room
- Population per household

Generate pairwise numerical interaction features using PolynomialFeatures.

### 4. Model Training

**Ordinary Least Squares (OLS):** Establishes a baseline using linear regression.

**Ridge Regression (L2):** Applies L2 regularization to reduce the impact of large coefficients and help manage multicollinearity.

**Lasso Regression (L1):** Applies L1 regularization, which can shrink less useful feature coefficients to zero.

Ridge and Lasso hyperparameters are tuned using GridSearchCV with three-fold cross-validation in the initial configuration.

### 5. Model Evaluation

Models are evaluated using:

- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of Determination

Lower RMSE and MAE indicate smaller prediction errors, while higher R² indicates that the model explains more variation in the target.

### 6. Visualization
- Correlation heatmap
- Actual versus predicted plots
- Residual distribution plots
- Model performance comparison

## 📁 Project Structure

```text
California-Housing-Regression/
│
├── main.py
├── README.md
├── correlation_heatmap.png
├── model_evaluation.png
├── model_comparison.csv
└── predictions.csv
```

The image and CSV files are generated when the Python script runs successfully.

## 🚀 Installation and Usage

### Prerequisites
- Python 3.11 or newer
- PyCharm or another Python IDE
- Internet connection for downloading the dataset

### Step 1: Clone the repository

```bash
git clone https://github.com/FarazMemon004/Datheon_Machine_Learning_-_Data_Science_Internship/tree/main/Task2-Real-Estate-Housing-Price-Prediction
cd California-Housing-Regression
```

Alternatively, download the repository and open the project folder in PyCharm.

### Step 2: Install dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Step 3: Run the project

```bash
python main.py
```

### Step 4: Review the results

The program generates:

- `correlation_heatmap.png`
- `model_evaluation.png`
- `model_comparison.csv`
- `predictions.csv`

The evaluation metrics and selected hyperparameters are printed in the console.

## 📈 Results

The project compares OLS, Ridge, and Lasso using a held-out test set.

## 📈 Model Evaluation Results

The models were evaluated on a held-out test dataset using RMSE, MAE, and R².

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Lasso Regression | 66,504.97 | 47,484.97 | 0.66 |
| Ridge Regression | 66,506.90 | 47,496.77 | 0.66 |
| OLS Regression | 179,952.04 | 69,788.74 | -1.47 |

### Key Findings

- **Lasso Regression** achieved the lowest RMSE and MAE among the three models.
- **Ridge Regression** produced nearly identical performance to Lasso.
- **OLS Regression** showed substantially worse test performance, with a negative R² score, indicating that its predictions performed worse than a mean-target baseline.
- Regularization helped produce substantially better results in this experiment.

### Conclusion

Based on the current test results, Lasso Regression is the best-performing model among the three evaluated models. However, the unusually poor OLS result should be investigated to confirm that all models use consistent preprocessing and that the interaction features are not introducing numerical instability.

The model with the lowest test RMSE is identified as the best-performing model under this evaluation setup.

## 💡 Key Learning Outcomes

- Building end-to-end regression workflows
- Data cleaning and imputation
- Feature engineering and interaction terms
- Categorical encoding and numerical scaling
- Understanding L1 and L2 regularization
- Hyperparameter tuning with GridSearchCV
- Evaluating regression models using multiple metrics
- Visualizing prediction errors and model performance

## ⚠️ Limitations

- The dataset represents historical housing data, not current market prices.
- Linear regression models may not capture every nonlinear relationship.
- Interaction features can increase model complexity and overfitting risk.
- The target variable has an upper cap, which can limit predictions for expensive properties.
- The initial three-fold cross-validation setup can be increased to five folds for more robust model selection.

## 👨‍💻 Author

**Muhammad Faraz**

Software Engineering | App Developer | AI |
Machine Learning & Data Science | Python Developer


<p>
  <a href="https://github.com/FarazMemon004">
    <img src="https://img.shields.io/badge/GitHub-View%20Profile-181717?logo=github&logoColor=white" alt="GitHub Profile">
  </a>
  <a href="https://www.linkedin.com/in/muhammad-faraz-memon-92582a43a/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn Profile">
  </a>
</p>
---

If you find this project useful, consider giving the repository a ⭐ on GitHub.

#MachineLearning #Python #Regression #DataScience #ScikitLearn
