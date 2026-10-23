# 🌸 Iris Flower Classification | Datheon Internship

<p align="center">
  <img src="https://img.shields.io/badge/Internship-Datheon-blue" alt="Datheon Internship">
  <img src="https://img.shields.io/badge/Domain-Machine%20Learning-purple" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Language-Python-yellow" alt="Python">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Project Status">
</p>

## 📌 About the Project

This project is part of my **Machine Learning & Data Science Internship at Datheon**.

The objective is to build and evaluate supervised machine learning classification models using the Iris flower dataset. Two algorithms, Logistic Regression and Decision Tree, are trained and compared using standard classification metrics.

The project demonstrates data preprocessing, feature scaling, model training, evaluation, and visualization.

## 🎯 Objectives

- Load and explore the Iris dataset using Pandas.
- Visualize feature relationships using Seaborn pairplots.
- Split the dataset into 80% training and 20% testing data.
- Apply feature scaling using `StandardScaler`.
- Train Logistic Regression and Decision Tree classifiers.
- Evaluate the models using Accuracy, Precision, Recall, and F1-score.
- Generate confusion matrices and decision boundary plots.
- Compare the performance of both models.

## 📊 Dataset Information

**Dataset:** Iris Flower Dataset

**Source:** [Seaborn Iris Dataset (CSV)](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)

| Property | Description |
|---|---|
| Total samples | 150 |
| Input features | 4 |
| Target classes | 3 |
| Training data | 80% (120 samples) |
| Testing data | 20% (30 samples) |

### Input Features

- Sepal length
- Sepal width
- Petal length
- Petal width

### Target Classes

- Setosa
- Versicolor
- Virginica

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, training, and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots and heatmaps |

## 🤖 Machine Learning Models

### 1. Logistic Regression

A supervised classification algorithm used to predict the flower species based on its measurements. Features are standardized using `StandardScaler`.

### 2. Decision Tree Classifier

A tree-based supervised learning algorithm that learns decision rules from the training data to classify flower species.

Both models use the same train/test split for a fair comparison.

## 📈 Model Evaluation Results

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Correct positive predictions relative to all positive predictions.
- **Recall:** Correctly identified samples relative to all actual samples.
- **F1-score:** Harmonic mean of precision and recall.

### Performance Comparison

Replace the placeholders below with the actual values printed by `iris_classification.py`.

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|---|---:|---:|---:|---:|
| Logistic Regression | Pending | Pending | Pending | Pending |
| Decision Tree | Pending | Pending | Pending | Pending |

**Best-performing model:** To be determined from the actual test results.

> Note: Metrics are intentionally left as pending until the script has been executed. This ensures the reported results are accurate and reproducible.

## 📉 Visualizations

The script generates the following visualizations in the `outputs/` directory.

| Visualization                                  | File                                               |
|------------------------------------------------|----------------------------------------------------|
| Feature relationships and species distribution | `outputs/pairplot.png`                             |
| Species distribution                           | `outputs/species_distribution.png`                 |
| Logistic Regression confusion matrix           | `outputs/logistic_regression_confusion_matrix.png` |
| Decision Tree confusion matrix                 | `outputs/decision_tree_confusion_matrix.png`               |
| Model performance comparison                   | `outputs/model_comparison.png`                             |
| Logistic Regression decision boundary          | `outputs/logistic_regression_decision_boundary.png`        |
| Decision Tree decision boundary                | `outputs/decision_tree_decision_boundary.png`              |

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each flower species.

### Decision Boundary

Decision boundaries visualize how each classifier separates the three flower species using petal length and petal width.

## ⚙️ Installation and Execution

### 1. Clone the repository

```bash
git https://github.com/FarazMemon004/Datheon_Machine_Learning_-_Data_Science_Internship/tree/main/Task1-Iris-Classification
cd Task1-Iris-Classification
```

Replace the URL and folder name with your actual GitHub repository details.

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python iris_classification.py
```

The script downloads the dataset from the provided URL, trains both models, prints evaluation results, and saves the generated plots and comparison CSV.

An active internet connection is required to download the dataset.

## 📂 Project Structure

```text
Datheon-MLDS-Task1-Iris-Classification/
│
├── iris_classification.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── pairplot.png
    ├── species_distribution.png
    ├── logistic_regression_confusion_matrix.png
    ├── decision_tree_confusion_matrix.png
    ├── model_comparison.png
    ├── logistic_regression_decision_boundary.png
    ├── decision_tree_decision_boundary.png
    └── model_comparison.csv
```

## 🎓 Learning Outcomes

Through this project, I gained practical experience with:

- Exploratory data analysis and visualization.
- Data preprocessing and feature standardization.
- Training supervised classification models.
- Evaluating machine learning model performance.
- Interpreting confusion matrices and classification metrics.
- Comparing machine learning algorithms using test data.

## 👨‍💻 Author

**Muhammad Faraz**

Software Engineering | App Developer | AI |
Machine Learning & Data Science | Python Developer

<p>
  <a href="https://www.linkedin.com/in/muhammad-faraz-memon-92582a43a/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white" alt="Connect on LinkedIn">
  </a>
  <a href="https://github.com/FarazMemon004">
    <img src="https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=white" alt="GitHub Profile">
  </a>
</p>

## 🙏 Acknowledgment

Thanks to **Datheon** for providing this internship opportunity to develop practical skills in Machine Learning and Data Science.

---

⭐ If you find this project useful, consider giving the repository a star.
