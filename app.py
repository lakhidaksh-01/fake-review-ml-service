from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    text = review.text

    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Get vector values
    vector = X.toarray()[0]


    return {
        "fake_probability": float(prob),
        "trust_score": float(1 - prob),
    }