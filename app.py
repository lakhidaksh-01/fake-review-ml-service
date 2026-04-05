from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    text = review.text

    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]

    return {
        "fake_probability": float(prob),
        "trust_score": float(1 - prob),
    }