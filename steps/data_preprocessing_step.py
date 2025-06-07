# steps/data_preprocessing_step.py

from zenml import step
from typing import List
import pandas as pd
from src.data_preprocessing import DataPreprocessor

@step(enable_cache=False)
def data_preprocessing_step(
    df: pd.DataFrame,
    cols_to_drop: List[str],
    col_to_clean: str,
    col_to_map: str
) -> pd.DataFrame:
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess(df, cols_to_drop, col_to_clean, col_to_map)
    return processed_df
