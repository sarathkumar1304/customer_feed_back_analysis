import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import logging

class ModelBuilder:

    def __init__(self):
        self.trial = None

    def tune_model(self, model_name: str, X_train, y_train, X_val, y_val, n_trials=20):
        def objective(trial):
            self.trial = trial  # Save the trial for use in get_model
            model = self.get_model(model_name, X_train, y_train, tune=True)
            preds = model.predict(X_val)
            return 1.0 - accuracy_score(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        logging.info(f"Best trial for {model_name}: {study.best_params}")
        return self.get_model(model_name, X_train, y_train, tune=True)

    def logistic_regression(self, X_train, y_train):
        if self.trial:
            params = {
                "penalty": self.trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
                "C": self.trial.suggest_float("C", 0.01, 10.0, log=True),
                "solver": self.trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"]),
                "max_iter": self.trial.suggest_int("max_iter", 100, 1000)
            }
            model = LogisticRegression(**params)
        else:
            model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return model

    def decision_tree(self, X_train, y_train):
        if self.trial:
            params = {
                "criterion": self.trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": self.trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": self.trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": self.trial.suggest_int("min_samples_leaf", 1, 4)
            }
            model = DecisionTreeClassifier(**params, random_state=42)
        else:
            model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    def random_forest(self, X_train, y_train):
        if self.trial:
            params = {
                "n_estimators": self.trial.suggest_int("n_estimators", 100, 500),
                "max_depth": self.trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": self.trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": self.trial.suggest_int("min_samples_leaf", 1, 4),
                "max_features": self.trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
            }
            model = RandomForestClassifier(**params, random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    def svm(self, X_train, y_train):
        if self.trial:
            params = {
                "C": self.trial.suggest_float("C", 0.1, 10.0, log=True),
                "kernel": self.trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                "gamma": self.trial.suggest_categorical("gamma", ["scale", "auto"])
            }
            model = SVC(**params)
        else:
            model = SVC()
        model.fit(X_train, y_train)
        return model

    def naive_bayes(self, X_train, y_train):
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model

    def knn(self, X_train, y_train):
        if self.trial:
            params = {
                "n_neighbors": self.trial.suggest_int("n_neighbors", 3, 15),
                "weights": self.trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": self.trial.suggest_int("p", 1, 2)  # 1: Manhattan, 2: Euclidean
            }
            model = KNeighborsClassifier(**params)
        else:
            model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        return model

    def adaboost(self, X_train, y_train):
        if self.trial:
            params = {
                "n_estimators": self.trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
            }
            model = AdaBoostClassifier(**params)
        else:
            model = AdaBoostClassifier()
        model.fit(X_train, y_train)
        return model

    def gradient_boosting(self, X_train, y_train):
        if self.trial:
            params = {
                "n_estimators": self.trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": self.trial.suggest_int("max_depth", 3, 30),
                "subsample": self.trial.suggest_float("subsample", 0.5, 1.0)
            }
            model = GradientBoostingClassifier(**params)
        else:
            model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        return model

    def get_model(self, model_name: str, X_train, y_train, tune=False):
        logging.info(f"Model Building {model_name} | Tuning: {tune}")
        if model_name == "logistic_regression":
            return self.logistic_regression(X_train, y_train)
        elif model_name == "decision_tree":
            return self.decision_tree(X_train, y_train)
        elif model_name == "random_forest":
            return self.random_forest(X_train, y_train)
        elif model_name == "svm":
            return self.svm(X_train, y_train)
        elif model_name == "naive_bayes":
            return self.naive_bayes(X_train, y_train)
        elif model_name == "knn":
            return self.knn(X_train, y_train)
        elif model_name == "adaboost":
            return self.adaboost(X_train, y_train)
        elif model_name == "gradient_boosting":
            return self.gradient_boosting(X_train, y_train)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
