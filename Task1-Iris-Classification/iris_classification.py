
# ============================================================
# Task 1: Iris Flower Classification
# Models: Logistic Regression and Decision Tree
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/iris.csv"
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")

# ------------------------------------------------------------
# 2. Load the dataset using Pandas
# ------------------------------------------------------------

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 60)

df = pd.read_csv(DATA_URL)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nSpecies distribution:")
print(df["species"].value_counts())

# Handle missing values if any are present.
df = df.dropna().copy()

# Features and target
X = df.drop(columns=["species"])
y = df["species"]

feature_names = X.columns.tolist()
class_names = sorted(y.unique())

print("\nFeatures:", feature_names)
print("Classes:", class_names)

# ------------------------------------------------------------
# 3. Exploratory Data Analysis: Pairplot
# ------------------------------------------------------------

pair_plot = sns.pairplot(
    df,
    hue="species",
    diag_kind="hist",
    height=2.2
)

pair_plot.figure.suptitle(
    "Iris Dataset - Feature Pairplot",
    y=1.02
)

pair_plot.savefig(
    os.path.join(OUTPUT_DIR, "pairplot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(pair_plot.figure)

# Class distribution chart
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species")
plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "species_distribution.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 4. Train/Test Split: 80% Training, 20% Testing
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 5. Define models with feature scaling
# ------------------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ))
    ])
}

results = []
trained_models = {}

# ------------------------------------------------------------
# 6. Train and evaluate both models
# ------------------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predict test samples
    y_pred = model.predict(X_test)

    trained_models[model_name] = model

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=class_names,
        zero_division=0
    ))

    # --------------------------------------------------------
    # 7. Confusion Matrix Heatmap
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_names
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )
    plt.close()

# ------------------------------------------------------------
# 8. Compare model performance
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

# Metric comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

plot_data = results_df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=plot_data,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.ylim(0, 1.05)
plt.title("Logistic Regression vs Decision Tree")
plt.ylabel("Macro-Average Score")
plt.xlabel("Evaluation Metric")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 9. Decision Boundary Visualization
# Use two features for a 2D visualization.
# ------------------------------------------------------------

boundary_features = ["petal_length", "petal_width"]

X_2d = df[boundary_features]
y_all = df["species"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d,
    y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all
)

# Fit each model using only two features.
boundary_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ))
    ])
}

# Grid for plotting predicted regions
x_min, x_max = (
    X2_train.iloc[:, 0].min() - 0.5,
    X2_train.iloc[:, 0].max() + 0.5
)
y_min, y_max = (
    X2_train.iloc[:, 1].min() - 0.5,
    X2_train.iloc[:, 1].max() + 0.5
)

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = pd.DataFrame(
    np.c_[xx.ravel(), yy.ravel()],
    columns=boundary_features
)

for model_name, model in boundary_models.items():

    model.fit(X2_train, y2_train)

    # Convert predicted class names into numeric plot labels.
    grid_predictions = model.predict(grid)
    class_to_number = {
        name: index for index, name in enumerate(class_names)
    }

    Z = np.array([
        class_to_number[name] for name in grid_predictions
    ]).reshape(xx.shape)

    plt.figure(figsize=(9, 6))

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.25,
        levels=np.arange(len(class_names) + 1) - 0.5,
        cmap="viridis"
    )

    sns.scatterplot(
        x=X2_train["petal_length"],
        y=X2_train["petal_width"],
        hue=y2_train,
        hue_order=class_names,
        palette="viridis",
        edgecolor="black",
        s=55
    )

    plt.title(f"{model_name} - Decision Boundaries")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(title="Actual Species")
    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_decision_boundary.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )
    plt.close()

# ------------------------------------------------------------
# 10. Final result
# ------------------------------------------------------------

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Best model by test accuracy:", best_model["Model"])
print(f"Test accuracy: {best_model['Accuracy']:.4f}")

print(f"\nAll plots and results saved in: {OUTPUT_DIR}/")
print("Classification task completed successfully!")
