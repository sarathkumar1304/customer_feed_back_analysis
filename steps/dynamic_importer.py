import pandas as pd
from zenml import step

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    data = {
        'Review Text': [
            "This product is really awesome, and worth for price. I would recommend it to my friends and family. The quality is top-notch, and the customer service is excellent. Overall, a fantastic experience!"
        ]
    }
    
    df = pd.DataFrame(data)
    json_data = df.to_json(orient="split")
    return json_data
