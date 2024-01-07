import numpy as np
import nltk
import pickle
import re
import os
from nltk.corpus import stopwords
from keras.models import load_model
from keras.utils import pad_sequences
from fastapi import HTTPException


project_dir = os.getcwd()

# Construct the file path relative to the project directory
tokenizer_path = os.path.join(project_dir, 'savedmodel', 'tokenizerEN.pickle')
model_path = os.path.join(project_dir, 'savedmodel', 'sentiment_analysis_modelEN.h5')

# Load the tokenizer and model
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model(model_path)

# Preprocess the text
def preprocess_text(text):
    # Check if text is not null
    if isinstance(text, str):
        # Remove special characters and convert to lowercase
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back into a string
        text = " ".join(tokens)
    return text

class EmptyTextException(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Input text cannot be empty")


async def predictEN(new_text):
    try:
        if not new_text:
            raise EmptyTextException()

        new_text = preprocess_text(new_text)
        new_text = tokenizer.texts_to_sequences([new_text])
        new_text = pad_sequences(new_text, maxlen=129)

        prediction = model.predict(new_text)[0]

        sentiments = ['negative', 'neutral', 'positive']
        sentiment = sentiments[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)), 2)
        percent = round(confidence * 100)
    except EmptyTextException as empty_text_error:
        raise empty_text_error
    except Exception as error:
        print(error)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return {'sentiment': sentiment, 'percentage': f'{percent}'}

