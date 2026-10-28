"""
Heart Attack Prediction with Logistic Regression
Machine Learning and Smart Systems - BA, SS 2026

How to run:
    pip install pandas scikit-learn matplotlib
    python main.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "heart_attack_sample.csv"
FIGURES_DIR = PROJECT_DIR / "figures"
TARGET = "heart_attack_risk"

def main():
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Missing values:\n", df.isna().sum())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nPerformance metrics")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:", round(recall_score(y_test, y_pred), 3))
    print("F1-score:", round(f1_score(y_test, y_pred), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    coefficients = pd.Series(
        model.named_steps["logreg"].coef_[0], index=X.columns
    ).sort_values(key=abs, ascending=False)
    print("\nMost important coefficients:")
    print(coefficients.head(8))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No risk", "Risk"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "confusion_matrix_from_code.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
