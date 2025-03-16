from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


# Define input data model
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float


# Define predict endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert input features into an array
    input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                            features.AveBedrms, features.Population, features.AveOccup]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return result as JSON
    return {"Predicted House Price": round(prediction, 2)}