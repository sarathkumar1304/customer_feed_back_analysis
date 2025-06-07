from zenml import step
import pandas as pd

from src.data_splitting import DataSplitting
from typing import Tuple

@step
def data_splitting_step(df: pd.DataFrame,target:str)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data_splitting = DataSplitting()
    X_train, X_test, y_train, y_test = data_splitting.split_data(df,target=target)
    return X_train, X_test, y_train, y_test