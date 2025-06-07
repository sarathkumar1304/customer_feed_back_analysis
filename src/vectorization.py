import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
class TfidfVectorization:
    def __init__(self):
        """
        Initializes the TfidfVectorizer for use on text data.
        """
        self.vectorizer = TfidfVectorizer(max_features=500)

    def fit_transform(self, X_train: pd.DataFrame):
        """
        Fits the vectorizer to the training data and transforms it.

        Parameters:
            X_train (pd.DataFrame): The training data to fit and transform.

        Returns:
            sparse matrix: The transformed training data in sparse matrix form.
        """
        logging.info("TF-IDF Vectorizer: Fitting and transforming training data.")
        transformed_data = self.vectorizer.fit_transform(X_train)

        # Save the trained vectorizer (not model!)
        vectorizer_dir = "models"
        os.makedirs(vectorizer_dir, exist_ok=True)
        vectorizer_path = os.path.join(vectorizer_dir, "vectorizer.pkl")
        joblib.dump(self.vectorizer, vectorizer_path)
        logging.info(f"Vectorizer saved locally at {vectorizer_path}")

        return transformed_data

    
    def transform(self, X_test:pd.DataFrame):
        """
        Transforms the test data using the already-fitted vectorizer.

        Parameters:
            X_test (pd.Series): The test data to transform.

        Returns:
            sparse matrix: The transformed test data in sparse matrix form.
        """
        logging.info("TF-IDF Vectorizer: Transforming test data")
        return self.vectorizer.transform(X_test)
    

# Example usage
if __name__ == "__main__":
    # # Example DataFrame (replace with actual data loading)
    # df_train = pd.DataFrame({
    #     'Review Text': ['Good product! Highly recommend.', 'Just okay.', 'Worst product ever!']
    # })
    # df_test = pd.DataFrame({
    #     'Review Text': ['Amazing quality!', 'Not good at all.']
    # })

    # # Initialize the TF-IDF Vectorizer
    # tfidf_vectorizer = TfidfVectorization()
    # tf_x_train = tfidf_vectorizer.fit_transform(df_train['Review Text'])
    # tf_x_test = tfidf_vectorizer.transform(df_test['Review Text'])

    # print("TF-IDF Vectors for Training Data:\n", tf_x_train.toarray())
    # print("TF-IDF Vectors for Test Data:\n", tf_x_test.toarray())
    pass