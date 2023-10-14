from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from transformers import pipeline

model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

app = FastAPI()

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

@app.get("/sentiment-analysis/{text}")
def sentiment_analysis(text: str):
    return {
        "result": sentiment_task(text),
    }