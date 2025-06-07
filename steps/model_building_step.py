import logging
from typing import Annotated
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib
import mlflow
from zenml import step
from zenml.client import Client
from zenml import Model
from src.model_building import ModelBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)


# Get the active experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker

# ZenML model metadata
model_metadata = Model(
    name="Customer Feedback Model",
    description="Sentiment classifier for customer feedback"
)

@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
    model=model_metadata,
)
def model_building_step(
    X_train: Annotated[csr_matrix, "Training features"],
    y_train: Annotated[pd.Series, "Training labels"],
    X_test: Annotated[csr_matrix, "Test features"],
    y_test: Annotated[pd.Series, "Test labels"],
    model_name: str = "logistic_regression"
) -> ClassifierMixin:
    """
    Train a classification model and log it with MLflow.
    """
    try:
        logging.info(f"Initializing training for model: {model_name}")

        # Start an MLflow run
        if not mlflow.active_run():
            mlflow.start_run(run_name=f"{model_name}_run")

        try:
            mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False, log_datasets=False)
            logging.info("MLflow autologging enabled.")
        except Exception as e:
            logging.warning(f"MLflow autologging could not be enabled: {e}")

        # Initialize model builder
        try:
            builder = ModelBuilder()
            model = builder.get_model(model_name, X_train, y_train)
            logging.info(f"{model_name} model successfully trained.")
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model saved locally at {model_path}")
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

        # Perform prediction
        try:
            y_pred = model.predict(X_test)
            logging.info("Prediction completed.")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

        # Calculate evaluation metrics
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            logging.info(f"Evaluation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise

        # Log to MLflow manually in case autolog missed anything
        try:
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, artifact_path="model")
            logging.info("Model and metrics logged to MLflow.")
        except Exception as e:
            logging.warning(f"Failed to log to MLflow manually: {e}")

        return model

    except Exception as final_error:
        logging.critical(f"Model building step failed: {final_error}", exc_info=True)
        raise final_error
