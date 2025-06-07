from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import logging
import numpy as np


class ModelEvaluation:

    def evaluate(self, model,X_test,y_test):
        y_pred = model.predict(X_test)
        accuracy= accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        logging.info(f"Model Evaluation Results:\n"
                     f"Accuracy: {accuracy:.4f}\n"
                     f"Precision: {precision:.4f}\n"
                     f"Recall: {recall:.4f}\n"
                     f"F1 Score: {f1:.4f}")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }