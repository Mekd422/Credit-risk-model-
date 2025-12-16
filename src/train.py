import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from src.data_processing import build_preprocessing_pipeline
import joblib

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]
    return X, y

def train_models(X_train, y_train, preprocessor):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(random_state=42)
    }

    param_grids = {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1, 10]  # note the pipeline step prefix
        },
        "random_forest": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 5, 10]
        }
    }

    best_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        grid = GridSearchCV(pipeline, param_grids[name], cv=3, scoring="roc_auc")
        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_
        print(f"Best {name}: {grid.best_params_}")

        # MLflow logging
        with mlflow.start_run(run_name=name):
            mlflow.sklearn.log_model(grid.best_estimator_, "model")
            y_pred = grid.predict(X_train)
            mlflow.log_metric("accuracy", accuracy_score(y_train, y_pred))
            mlflow.log_metric("precision", precision_score(y_train, y_pred))
            mlflow.log_metric("recall", recall_score(y_train, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_train, y_pred))
            y_proba = grid.predict_proba(X_train)[:, 1]
            mlflow.log_metric("roc_auc", roc_auc_score(y_train, y_proba))

    return best_models

def main():
    data_path = "./data/processed/customer_features.csv"
    X, y = load_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build preprocessing pipeline
    preprocessor, _, _ = build_preprocessing_pipeline(X_train)

    # Train models
    best_models = train_models(X_train, y_train, preprocessor)

    # Save the preprocessing pipeline
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Preprocessing pipeline saved to models/preprocessor.pkl")

if __name__ == "__main__":
    main()
