### 🏛️ Architecture

![Architecture](assets/customer.drawio.svg)
# 📊 Customer Feedback Sentiment Prediction 

This project is a complete end-to-end pipeline to predict whether customer feedback is **positive** or **negative**, using machine learning and MLOps tools. It leverages **ZenML** for orchestration, **MLflow** for experiment tracking and model deployment, and **Streamlit** for the interactive web interface.



## 🚀 Features
- Preprocessing with **TF-IDF Vectorization**
- Sentiment classification using a trained ML model (`.pkl`)
- MLOps stack with **ZenML**, **MLflow**
- Web interface for real-time predictions using **Streamlit**


## 🛠️ Setup Instructions

Follow these steps to set up and run the project:

### 🔁 1. Create and activate virtual environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 📦 2. Install dependencies

```bash
pip install -r requirements.txt
```

### ⚙️ 3. Initialize and configure ZenML

```bash
zenml init
zenml integration install mlflow -y
```

### 🧪 4. Register MLflow experiment tracker

```bash
zenml experiment-tracker register customer_feedback_tracker --flavor=mlflow
```

### 🚀 5. Register MLflow model deployer

```bash
zenml model-deployer register customer_feedback_model_deployer --flavor=mlflow
```

### 🧱 6. Register and set the ZenML stack

```bash
zenml stack register customer_feedback_stack \
    -a default \
    -o default \
    -e customer_feedback_tracker \
    -d customer_feedback_model_deployer \
    --set
```

---

## 🧪 Train & Deploy the Model

### 🔧 Run the training pipeline

```bash
python3 run_pipeline.py
```

### 🚀 Deploy the trained model

```bash
python3 run_deployment.py
```

---

## 💻 Launch the Streamlit Web App

Once the model is deployed, run the web application using:

```bash
streamlit run app.py
```

You can now enter customer feedback in the web interface and get predictions for whether it's **Positive** or **Negative**.

---

## 📝 Example Feedback for Testing

```text
Positive: "This product is fantastic, exceeded my expectations!"
Negative: "Very disappointed. I want a refund."
```

---

## 📂 Project Structure

```
Customer_Feedback_Analysis/
│──.zen/
      │──config.yaml
|│──assets/
       │──images of the project
│── data/customer_feedback_analysis.csv     # Raw and processed data
|── models
     |── pickle file                        # Trained model pickle file
|──myenv             
│── notebooks/
      |-- analyze_plots                     # Source code for all residual plots
      |-- EDA.ipynb                         # Jupyter Notebooks for EDA & model training
│── pipelines/                              # Python scripts for modular implementation
        |── deployment_pipeline.py          # Contininous pipeline and inference pipeline
        |── training_pipeline.py            # training machine learning pipeline
|── src/
      |── data_ingestion.py                 # data ingestions
      |── data_preprocessing.py             # data cleaning
      |── data_splitting.py                 # data split into train and test 
      |── model_building.py                 # model building for model training
      |── model_evaluation.py               # model evaluation of trained model
      |── vectorization.py                  # vectorization of the features
|── steps/
      |── data_ingestion_step.py           # data ingestions step contain zenml step to track flow of data 
      |── data_preprocessing_step.py       # data preprocessing  step like cleaning fill null values
      |── data_splitting_step.py           # data spliited into train and test step
      |── model_building_step.py           # model building step
      |── model_evaluation_step.py         # model evaluation step 
      |── vectorization_step.py
      |── dynamic_importer.py              # import sample data for testing
      |──prediction_service_loader.py      # mlflow prediction service loader
      |──predictor.py                      # prediction 
│── run_pipeline.py                        # run whole pipeline at one place 
│── run_deployment.py                      # Model deployment process
│── README.md                              # Project documentation
│── requirements.txt                       # List of dependencies
│──app.py
│── DockerFile
│── docker-compose.yml

```

## 🧠 Tech Stack

* **Python 3**
* **Scikit-learn**
* **TF-IDF Vectorizer**
* **ZenML**
* **MLflow**
* **Streamlit**
* **Pandas**


## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request.



## 📃 License

This project is licensed under the MIT License.




