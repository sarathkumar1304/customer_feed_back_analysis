import pandas as pd
from zenml import step
from src.data_ingestion import DataIngestion

@step
def data_ingestion_step(path: str) -> pd.DataFrame:
    data_ingestion = DataIngestion()
    df = data_ingestion.ingest_data(path)
    return df