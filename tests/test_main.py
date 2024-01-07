import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from deploy.main import app
from deploy.model import predictEN


from fastapi.testclient import TestClient

client = TestClient(app)

def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Sentiment analysis API'}



def test_predict_valid_input():
    payload = {"text": "This is a positive sentence."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "data" in response.json()
    assert "sentiment" in response.json()["data"]
    assert "percentage" in response.json()["data"]

def test_predict_invalid_input():
    payload = {"invalid_key": "This is an invalid input."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # 422 Unprocessable Entity for invalid input
    assert "detail" in response.json()
    assert "text" in response.json()["detail"][0]["loc"]

def test_predict_internal_server_error():
    # Simulate an internal server error by providing invalid text (which raises an exception in your model prediction)
    payload = {"text": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# Add more tests as needed
def test_boundary_input_length():
    # Test with the maximum allowed input length
    max_length_text = "a" * 129  # Assuming max length is 129 in your pad_sequences
    payload = {"text": max_length_text}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_empty_input():
    # Test with an empty input
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Check the actual response status code

def test_long_text_input():
    # Test with a very long input text
    long_text = "a" * 1000
    payload = {"text": long_text}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_invalid_text_format():
    # Test with an invalid text format
    payload = {"text": 123}  # Invalid input format (expecting a string)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Expecting Unprocessable Entity for invalid input format

def test_invalid_input_characters():
    # Test with invalid characters in the input text
    invalid_text = "This is an input with special characters: @$%^"
    payload = {"text": invalid_text}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200  # Assuming you handle special characters appropriately



def test_whitespace_input():
    # Test with whitespace-only input
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200  # Check the actual response status code


import concurrent.futures

def test_multiple_requests():
    # Define a function to make a prediction request
    def make_prediction(text):
        payload = {"text": text}
        return client.post("/predict", json=payload)

    # List of texts for predictions
    texts = ["This is text 1", "This is text 2", "This is text 3"]

    # Use ThreadPoolExecutor to simulate multiple simultaneous requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the prediction requests
        futures = [executor.submit(make_prediction, text) for text in texts]

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # Check the responses
        for future in futures:
            response = future.result()
            assert response.status_code == 200
