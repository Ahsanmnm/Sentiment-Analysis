import uvicorn
import nltk
import logging  # Add this line
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import predictEN
from pydantic import BaseModel,validator
from typing import ClassVar


# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Create FastAPI app
app = FastAPI()


# Define request payload model
class TextPayload(BaseModel):
    text: str
    ModuleType: ClassVar
    
    @validator("text")
    def validate_text(cls, value):
        if not isinstance(value, str):
            raise ValueError("Input should be a valid string")
        return value

# Index route
@app.get('/')
def index():
    try:
        logging.info("Received request for index.")
        return {'message': 'Sentiment analysis API'}
    except Exception as error:
        logging.error(f"Error handling index request: {error}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Predict route
@app.post('/predict')
async def predict(payload: TextPayload):
    try:
        text = payload.text
        logging.info(f"Received prediction request for text: {text}")
        data = await predictEN(text)
        logging.info(f"Prediction result: {data}")
        return {"data": data}
    except HTTPException as http_error:
        logging.error(f"HTTP Exception while predicting sentiment: {http_error.detail}")
        raise
    except Exception as error:
        logging.error(f"Error predicting sentiment: {error}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run the server
if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
