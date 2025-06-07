import os
from pipelines.training_pipeline import training_pipeline
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.predictor import predictor
from steps.prediction_service_loader import prediction_service_loader
from steps.dynamic_importer import dynamic_importer

requirements_file= os.path.join(os.path.dirname(__file__),"requirements.txt")

@pipeline
def continuous_deployment_pipeline():
    # Run the training pipeline
    """
    This pipeline trains a model and deploys it to the prediction service.
    
    The pipeline will:
    1. Run the training pipeline
    2. Deploy the trained model to the prediction service
    
    The prediction service will be updated with the newly trained model
    if the model is better than the currently deployed one.
    """
    trained_model = training_pipeline()
    
    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers= 3,deploy_decision = True,model = trained_model)
    
    
@pipeline(enable_cache= False)
def inference_pipeline():
    
    # Load batch data for inference
    
    """
    This pipeline performs inference using a deployed model service.

    The inference pipeline will:
    1. Load batch data for inference.
    2. Load the deployed model service from the specified pipeline and step.
    3. Run prediction on the batch data using the loaded model service.
    """

    batch_data = dynamic_importer()
    
    # load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name = "continuous_deployment_pipeline",
        step_name = "mlflow_model_deployer_step",)
    
    
    # Run prediction on the batch data
    predictor(service = model_deployment_service,input_data = batch_data)