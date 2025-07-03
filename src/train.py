import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if hasattr(model, "predict_proba") else 0.5
    }

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("../../data/processed/final_labeled_data.csv")

    # Separate features and target
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]

    # Feature selection using RandomForest importance
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="median"
    )
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"Selected {len(selected_features)} features:")
    print(selected_features.tolist())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    # Train and evaluate models
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, model_name)
            
            # Print results
            print(f"\n{model_name.upper()} Results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")