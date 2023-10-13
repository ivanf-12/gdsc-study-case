from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# text classifier from hugging face
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

app = FastAPI()

class ItemSentiment(BaseModel): 
    Sentence: str #"I am so happy today"

@app.post('/')
async def scoring_endpoint(item:ItemSentiment):
    sentiment_result = classifier(item.Sentence)
    return sentiment_result 