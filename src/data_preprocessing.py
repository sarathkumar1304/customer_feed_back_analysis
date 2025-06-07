import pandas as pd
import re
from typing import List
from nltk.corpus import stopwords
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required resources only once
nltk.download('stopwords', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        logging.info("Initialized DataPreprocessor with NLTK stopwords.")

    def remove_unwanted_columns(self, df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
        logging.info(f"Removing unwanted columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')
        logging.info(f"Remaining columns: {df.columns.tolist()}")
        return df

    def remove_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_shape = df.shape
        df = df.dropna()
        logging.info(f"Removed null values: {initial_shape[0] - df.shape[0]} rows dropped.")
        return df

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = re.sub(r'\d+', '', text)      # remove numbers
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        return text

    def clean_review_column(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        logging.info(f"Cleaning text column: '{text_column}'")
        df[text_column] = df[text_column].astype(str).apply(self.clean_text)
        logging.info(f"Finished cleaning text in column: '{text_column}'")
        return df

    def map_sentiment(self, df: pd.DataFrame, rating_column: str, sentiment_column: str = "Sentiment") -> pd.DataFrame:
        logging.info(f"Mapping sentiment from rating column: '{rating_column}' to '{sentiment_column}'")
        df[sentiment_column] = df[rating_column].apply(lambda x: "positive" if x > 3 else "negative")
        logging.info(f"Sentiment mapping completed: {df[sentiment_column].value_counts().to_dict()}")
        return df

    def encode_sentiment(self, df: pd.DataFrame, sentiment_column: str = "Sentiment") -> pd.DataFrame:
        logging.info(f"Encoding sentiment column: '{sentiment_column}'")
        sentiment_map = {"positive": 1, "negative": 0}
        df[sentiment_column] = df[sentiment_column].map(sentiment_map)
        logging.info(f"Sentiment encoding done. Unique values: {df[sentiment_column].unique().tolist()}")
        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        cols_to_drop: List[str],
        col_to_clean: str,
        col_to_map: str
    ) -> pd.DataFrame:
        logging.info("Starting data preprocessing...")
        df = self.remove_unwanted_columns(df, cols_to_drop)
        df = self.remove_null_values(df)
        df = self.clean_review_column(df, col_to_clean)
        df = self.map_sentiment(df, col_to_map)
        df = self.encode_sentiment(df, sentiment_column="Sentiment")
        df.reset_index(drop=True, inplace=True)
        logging.info("Data preprocessing completed successfully.")
        return df
