import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = "credit_scoring_preprocessing/credit_scoring_preprocessed.csv"
EXPERIMENT_NAME = "Credit Scoring CI Experiment"

mlflow.set_experiment(EXPERIMENT_NAME)


def get_target_column(df):
    possible_targets = [
        "Risk", "risk", "target", "Target", "label", "Label",
        "class", "Class", "default", "Default"
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    return df.columns[-1]


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset berhasil dimuat")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    target_col = get_target_column(df)
    print("Target column:", target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # Dalam mlflow run, MLflow sudah membuat run utama.
    # start_run() ini akan memakai run tersebut jika MLFLOW_RUN_ID tersedia.
    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("target_column", target_col)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5)
        )

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


if __name__ == "__main__":
    main()