# Sentiment Analysis API

This project is a sentiment analysis API that categorizes text data into positive, negative, or neutral sentiments. It is built using [FastAPI](https://fastapi.tiangolo.com/).

## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Run the API Server](#3-run-the-api-server)
- [API Endpoints](#api-endpoints)
  - [1. Home (GET)](#1-home-get)
  - [2. Predict (POST)](#2-predict-post)
- [Testing](#testing)
- [Issues and Contributions](#issues-and-contributions)
- [License](#license)

## Overview

This project implements a sentiment analysis model exposed as a FastAPI-based API. It allows users to submit text data and receive predictions about the sentiment expressed in the text.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Sentiment-Analysis-API
```
### 2. Install Dependencies

Before running the Sentiment Analysis API, make sure to install the required Python packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```
### 3. Run the API Server

Once the dependencies are installed, you can start the Sentiment Analysis API server. Use the following command to run the server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## API Endpoints

The Sentiment Analysis API provides the following endpoints:

### 1. Home (GET)

- **Endpoint:** `/`
- **Description:** Get information about the Sentiment Analysis API.

**Example:**

```bash
curl http://localhost:8000/
```

### 2. Predict (POST)

- **Endpoint:** `/predict`
- **Description:** Perform sentiment analysis on text data.

**Example Request:**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "This is a positive sentence."}' http://localhost:8000/predict
```
![OpenAI Logo](Sentiment-Analysis\images\Screenshot_2.png)

## Testing

The Sentiment Analysis API comes with a suite of tests to ensure its proper functionality. Before running the tests, make sure the API server is running.

### Running Tests

To run the tests, execute the following command in your terminal:

```bash
pytest test/test_main.py
```