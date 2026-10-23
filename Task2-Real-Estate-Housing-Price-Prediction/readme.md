# 🏡 Real Estate Housing Price Prediction

## 📌 Project Overview

This project was completed as **Task 2** of the **Datheon Machine Learning & Data Science Internship — 2026**.

The objective of this project is to predict median house values using the California Housing dataset and compare different regression models to evaluate their predictive performance.

The complete workflow was developed in a **Jupyter Notebook / Google Colab**, covering data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and visualization.

## 🎯 Objectives

- Explore and understand the housing dataset.
- Handle missing values using appropriate imputation techniques.
- Perform feature engineering and create interaction features.
- Encode categorical variables and scale numerical features.
- Investigate feature correlations and multicollinearity.
- Train and compare OLS, Ridge, and Lasso regression models.
- Optimize model hyperparameters using GridSearchCV.
- Evaluate model performance using RMSE, MAE, and R².
- Visualize actual vs. predicted values and residual distributions.

## 📊 Dataset

**Dataset:** California Housing Dataset

**Source:** [California Housing Dataset (CSV)](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)

Dataset details:

- **Total Records:** 20,640
- **Total Columns:** 10
- **Target Variable:** `median_house_value`
- **Problem Type:** Supervised Machine Learning — Regression

The dataset includes housing-related features such as median income, housing age, total rooms, bedrooms, population, households, geographic coordinates, and ocean proximity.

## 🛠️ Technologies Used

- Python
- Google Colab
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## ⚙️ Project Workflow

### 1. Data Loading and Exploration
- Loaded the California Housing dataset.
- Examined dataset dimensions, data types, and statistical summaries.
- Identified missing values and explored feature relationships.

### 2. Data Preprocessing
- Imputed missing numerical values.
- Handled categorical features through encoding.
- Prepared numerical features for model training.

### 3. Feature Engineering
Created additional features to capture useful relationships:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

Interaction features were also generated to help the models capture relationships between input variables.

### 4. Model Training
Implemented and compared the following regression algorithms:

- **OLS Regression:** Ordinary Least Squares baseline.
- **Ridge Regression:** L2 regularization to reduce the impact of large coefficients.
- **Lasso Regression:** L1 regularization to encourage sparse coefficients.

### 5. Hyperparameter Tuning
Used **GridSearchCV** to search for suitable regularization parameters for Ridge and Lasso regression.

### 6. Model Evaluation
Evaluated model performance using:

- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R² Score:** Coefficient of Determination

## 📈 Model Performance

The following are the recorded test-set results from the current experiment.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Lasso Regression | 66,504.97 | 47,484.97 | 0.66 |
| Ridge Regression | 66,506.90 | 47,496.77 | 0.66 |
| OLS Regression | 179,952.04 | 69,788.74 | -1.47 |

**Best recorded model:** Lasso Regression

Lasso achieved the lowest RMSE and MAE among the three models, although its performance was very similar to Ridge Regression.

**Note:** The negative R² score for OLS indicates that it performed worse than a simple test-set mean baseline. This result should be investigated for numerical instability, feature construction, and preprocessing issues before drawing a final conclusion about OLS.

## 📉 Visualizations

The notebook includes visual analysis such as:

- Feature correlation heatmap
- Actual vs. predicted house values
- Residual distribution plots
- Model performance comparison

These visualizations help assess feature relationships, prediction quality, and model errors.

## 🚀 How to Run the Project

### Option 1: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Download or clone this repository.
3. Upload the project `.ipynb` notebook to Colab.
4. Ensure internet access is available so the dataset can be loaded from its source URL.
5. Select **Runtime → Run all**.
6. Review the outputs, evaluation metrics, and visualizations.

### Option 2: Jupyter Notebook

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and execute the cells in order.

## 📁 Project Structure

```text
Task2-Real-Estate-Housing-Price-Prediction/
│
├── README.md
├── California_Housing_Price_Prediction.ipynb
├──outputs/
   ├── correlation_heatmap.png
   ├── model_evaluation.png
   ├── model_comparison.csv
   └── predictions.csv
```

*This is the suggested structure. Keep only files that actually exist in your repository, and update the notebook filename if yours is different.*

## 💡 Key Learnings

- Data exploration and preprocessing
- Missing-value imputation
- Feature engineering and interaction terms
- Regression algorithms and regularization
- Hyperparameter optimization with GridSearchCV
- Model evaluation using regression metrics
- Data visualization and residual analysis
- Working with notebook-based machine learning projects


## 🏢 Internship

**Organization:** Datheon  
**Program:** Machine Learning & Data Science Internship  
**Year:** 2026


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
