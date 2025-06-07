import pandas as pd
# from utils.logger import logger
import logging



class DataIngestion:
    def ingest_data(self,path:str)->pd.DataFrame:
        try:
            logging.info(f"Data is ingested from the path : {path}")
            df = pd.read_csv(path)
            logging.info(f"Data read succesfully from the {path}")
            return df
        except Exception as e:
            logging.warning(f"Data is not found on the given path :{path}")
            raise e
