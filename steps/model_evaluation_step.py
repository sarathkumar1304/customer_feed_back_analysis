from src.model_evaluation import ModelEvaluation
from zenml import step

from typing import Any
from sklearn.base import BaseEstimator
@step
def model_evaluation_step(model: BaseEstimator, X_test: Any, y_test: Any) -> dict:
    model_evaluator = ModelEvaluation()
    return model_evaluator.evaluate(model, X_test, y_test)