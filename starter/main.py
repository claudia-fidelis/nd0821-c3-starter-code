import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


from typing import Union 
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data
import pandas as pd

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

class TaggedItem(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example='State-gov')
    fnlgt: int = Field(example=77516)
    education: str = Field(example='Bachelors')
    education_num: int = Field(alias='education-num', example=13)
    marital_status: str = Field(alias='marital-status', example='Never-married')
    occupation: str = Field(example='Adm-clerical')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(alias='capital-gain', example=2174)
    capital_loss: int = Field(alias='capital-loss', example=0)
    hours_per_week: int = Field(alias='hours-per-week', example=40)
    native_country: str = Field(alias='native-country', example='United-States')

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the Census Data model API! =)"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/model/")
async def api_inference(item: TaggedItem):

    model = pickle.dump(model, open(model/model.pkl, 'wb'))
    encoder = pickle.dump(encoder, open(model/encoder.pkl, 'wb'))
    lb = pickle.dump(lb, open(model/lb.pkl, 'wb'))

    X, _, _, _ = process_data(
        pd.DataFrame(item).set_index(0).transpose().rename(
            columns={
                'education_num': 'education-num', 
                'marital_status':'marital-status', 
                'capital_gain': 'capital-gain',
                'capital_loss': 'capital-loss',
                'hours_per_week': 'hours-per-week',
                'native_country': 'native-country'
                }
        ), 
        categorical_features=cat_features, 
        training=False, 
        encoder=encoder, 
        lb=lb
    )
    pred = inference(model, X)
    return {"pred": f"Models prediction is {pred[0]}"}