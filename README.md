# 🧠 Datheon — Machine Learning & Data Science Internship

![Organization](https://img.shields.io/badge/Organization-Datheon-blue)
![Domain](https://img.shields.io/badge/Domain-Machine%20Learning-purple)
![Language](https://img.shields.io/badge/Language-Python-yellow)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

Welcome to my **Machine Learning & Data Science Internship** repository at **Datheon**.

This repository contains the projects, assignments, experiments, and practical implementations I complete throughout my internship. Each task is organized into a separate folder containing source code, documentation, results, and visualizations where applicable.

My goal is to strengthen my practical skills in Python, machine learning, data analysis, and artificial intelligence by working on real-world-inspired projects.

## 🎯 Internship Objectives

- Develop practical skills in Machine Learning and Data Science.
- Understand data preprocessing, feature engineering, and exploratory data analysis.
- Implement supervised and unsupervised learning algorithms.
- Evaluate machine learning models using appropriate performance metrics.
- Visualize and interpret datasets and model predictions.
- Understand regularization techniques and multicollinearity.
- Write clean, reusable, and well-documented Python code.
- Build a portfolio of practical machine learning projects.

## 📂 Repository Structure

```text
Datheon_Machine_Learning_-_Data_Science_Internship/
│
├── README.md
│
├── Task1-Iris-Classification/
│   ├── README.md
│   ├── iris_classification.py
│   ├── requirements.txt
│   └── outputs/
│       ├── pairplot.png
│       ├── species_distribution.png
│       ├── logistic_regression_confusion_matrix.png
│       ├── decision_tree_confusion_matrix.png
│       ├── model_comparison.png
│       ├── logistic_regression_decision_boundary.png
│       ├── decision_tree_decision_boundary.png
│       └── model_comparison.csv
│
└── Task2-California-Housing-Price-Prediction/
    ├── README.md
    ├── main.py
    ├── requirements.txt
    ├── correlation_heatmap.png
    ├── model_evaluation.png
    ├── model_comparison.csv
    └── predictions.csv
```

*The structure illustrates the organization of the repository. Actual files may vary depending on the implementation of each task.*

## 🚀 Internship Tasks

### Task 1 — Iris Flower Classification

**Folder:** [`Task1-Iris-Classification/`](https://github.com/FarazMemon004/Datheon_Machine_Learning_-_Data_Science_Internship/tree/main/Task1-Iris-Classification)

**Objective:** Implement supervised classification models using the Iris flower dataset and compare their performance.

**Key Features:**

- Load and explore the Iris dataset using Pandas.
- Perform exploratory data analysis using Seaborn.
- Split data into 80% training and 20% testing sets.
- Standardize features using `StandardScaler`.
- Train Logistic Regression and Decision Tree classifiers.
- Evaluate accuracy, precision, recall, and F1-score.
- Visualize confusion matrices and decision boundaries.
- Compare model performance and save results.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

**Dataset:** [Iris Flower Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)

For implementation details, instructions, and evaluation results, see the [Task 1 README](https://github.com/FarazMemon004/Datheon_Machine_Learning_-_Data_Science_Internship/blob/main/Task1-Iris-Classification/README.md).

### Task 2 — California Housing Price Prediction

**Folder:** [`Task2-California-Housing-Price-Prediction/`](https://github.com/FarazMemon004/Datheon_Machine_Learning_-_Data_Science_Internship/tree/main/Task2-California-Housing-Price-Prediction)

**Objective:** Predict median house values in California using regression algorithms, feature engineering, and regularization techniques.

**Key Features:**

- Load and explore the California Housing dataset.
- Analyze missing values, descriptive statistics, and feature correlations.
- Impute missing numerical and categorical values.
- Engineer meaningful features, including rooms per household, bedrooms per room, and population per household.
- Generate pairwise numerical interaction features.
- Encode categorical attributes using One-Hot Encoding.
- Standardize numerical features using `StandardScaler`.
- Implement Ordinary Least Squares (OLS), Ridge Regression (L2), and Lasso Regression (L1).
- Tune Ridge and Lasso hyperparameters using `GridSearchCV`.
- Evaluate models using RMSE, MAE, and R².
- Visualize actual versus predicted values and residual distributions.
- Export model comparison metrics and predictions to CSV files.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Statsmodels, and PyCharm.

**Dataset:** [California Housing Dataset](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)

#### 📊 Model Evaluation Results

The following results were obtained from the current test run:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Lasso Regression | 66,504.97 | 47,484.97 | 0.66 |
| Ridge Regression | 66,506.90 | 47,496.77 | 0.66 |
| OLS Regression | 179,952.04 | 69,788.74 | -1.47 |

**Key Findings:**

- Lasso Regression achieved the lowest RMSE and MAE in this experiment.
- Ridge Regression delivered nearly identical performance to Lasso.
- OLS produced substantially worse test results and a negative R² score, indicating that it performed worse than a baseline predicting the mean target value.
- Further investigation of the OLS result is warranted to understand the numerical behavior of the engineered features.

*These results represent the current run and may change if preprocessing, model settings, or hyperparameters are modified.*

## 🛠️ Technologies and Tools

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning algorithms and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Statsmodels | Statistical analysis and VIF calculations |
| Git & GitHub | Version control and project documentation |
| PyCharm | Python development environment |

The technologies used may differ across individual tasks.

## 💻 Getting Started

### Clone the repository

```bash
git clone https://github.com/FarazMemon004/Datheon_Machine_Learning_-_Data_Science_Internship.git

cd Datheon_Machine_Learning_-_Data_Science_Internship
```

### Set up the Python environment

On macOS, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies for the task you want to run.

### Run Task 1

```bash
cd Task1-Iris-Classification
pip install -r requirements.txt
python iris_classification.py
```

### Run Task 2

```bash
cd ../Task2-California-Housing-Price-Prediction
pip install -r requirements.txt
python main.py
```

An internet connection is required to download the datasets from their respective sources.

## 📊 Results and Documentation

Each completed task may include:

- Model performance metrics.
- Generated charts and visualizations.
- Confusion matrices and classification reports.
- Regression evaluation plots.
- Model comparison tables and CSV files.
- Source code and dependency requirements.
- Instructions for reproducing the results.

All reported metrics are based on the outputs generated by the corresponding project code.

## 📚 Learning Journey

This repository documents my ongoing learning journey in Machine Learning and Data Science, from understanding fundamental algorithms to implementing and evaluating practical solutions.

Through these projects, I am developing practical experience in data analysis, feature engineering, supervised learning, regression, classification, regularization, and model evaluation.

My goal is to continue expanding this repository with well-documented, reproducible projects while strengthening my skills as a Python Developer and Machine Learning Engineer.

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

## 🙏 Acknowledgment

I would like to thank **Datheon** for providing this internship opportunity to develop practical skills and gain hands-on experience in Machine Learning and Data Science.

---

⭐ Thank you for visiting my internship repository!
