from steps.data_ingestion_step import data_ingestion_step
from steps.data_preprocessing_step import data_preprocessing_step   
from steps.data_splitting_step import data_splitting_step
from zenml import pipeline
from steps.vectorization_step import vectorization_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step


@pipeline
def training_pipeline():
    """
    Pipeline for data ingestion.
    
    Args:
        data_ingestion_step: Step to ingest data.
    """
    df = data_ingestion_step(path="data/Womens_Clothing_E_Commerce_Reviews.csv")
    cleaned_df = data_preprocessing_step(df,cols_to_drop=['Title',"Clothing ID","Unnamed: 0"],col_to_clean="Review Text",col_to_map="Rating")
    X_train, X_test, y_train, y_test = data_splitting_step(df=cleaned_df,target="Sentiment")
    tf_X_train, tf_X_test = vectorization_step(X_train, X_test)
    model = model_building_step(model_name="logistic_regression",X_train=tf_X_train,y_train=y_train,X_test=tf_X_test,y_test=y_test)
    metrics= model_evaluation_step(model=model, X_test=tf_X_test, y_test=y_test)
    return model



    