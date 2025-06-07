import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from utils.logger import logger
import logging

class DataSplitting:
    def split_data(self,df:pd.DataFrame,target:str):
        """
        Splits the DataFrame into training and testing sets based on the specified feature and target columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data to be split.
            target (str): The name of the column to be used as the target variable for splitting.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training and testing sets
            as DataFrames for the features and Series for the target variable.
        """

        logging.info("Splitting the data into training and testing sets")
        logging.info(f"Shape of the DataFrame before splitting: {df.shape}")
        df = df[['Review Text', target]]
        X= df.drop(target,axis=1)
        y = df[target]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info(f"Shape of the training set: {X_train.shape}")
        logging.info(f"Shape of the testing set: {X_test.shape}")
        return X_train, X_test, y_train, y_test