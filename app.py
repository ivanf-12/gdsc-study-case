from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from transformers import pipeline

# text classifier from hugging face
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

app = FastAPI()

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

@app.get("/sentiment-analysis/{text}")
def sentiment_analysis(text: str):
    sentiment_result = classifier(text)
    return {
        "result": sentiment_result,
    }