
# End-to-End MNIST Digit Recognition Service

This project provides a web service for recognizing handwritten digits (0-9) from images. It is built with FastAPI, Scikit-learn, and packaged with Docker.

## 🎯 Project Goal

The primary goal is to productize a trained MNIST classification model into a scalable and easy-to-deploy web service. This includes providing a RESTful API endpoint, unit tests, and a Docker container for deployment.

## 🛠️ Technology Stack

- **API Framework**: FastAPI
- **ML Model**: Scikit-learn (`SGDClassifier`)
- **Deployment**: Docker
- **Testing**: Pytest
- **Python Version**: 3.11+

---

## 🚀 Getting Started

### 1. Environment Setup

It is recommended to use a Python virtual environment to manage dependencies.

**Create the virtual environment:**

```bash
# On Windows
python -m venv .venv

# On macOS/Linux
source .venv/bin/activate
```

**Activate the environment:**

```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

**Install the required packages:**

```bash
pip install -r requirements.txt
```

### 2. Running the Service Locally

Once the dependencies are installed, you can run the service using `uvicorn`.

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

On the first run, the application will automatically download the MNIST dataset, train a simple classifier, and save it as `mnist_model.joblib`. Subsequent runs will load the saved model.

### 3. Running with Docker

This project includes a `Dockerfile` for easy containerization.

**Build the Docker image:**

```bash
docker build -t mnist-api .
```

**Run the Docker container:**

```bash
docker run -p 8000:8000 mnist-api
```

The service will be accessible at `http://localhost:8000`.

---

## 📝 API Usage

### POST /predict/mnist

This endpoint accepts a PNG or JPG image of a handwritten digit and returns the predicted number with a confidence score.

- **URL**: `/predict/mnist`
- **Method**: `POST`
- **Body**: `multipart/form-data`
  - **file**: The image file to be classified.

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/predict/mnist" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/digit_image.png"
```

**Example Success Response:**

```json
{
  "prediction": "7",
  "confidence": 0.987654321
}
```

**Example Error Response:**

```json
{
  "detail": "File provided is not an image."
}
```

---

## ✅ Testing

Unit tests are provided to ensure the model's preprocessing and prediction logic works correctly. The test suite achieves over 90% code coverage for `model.py`.

To run the tests, use `pytest`:

```bash
pytest
```

To get a detailed coverage report:

```bash
pytest --cov=model
```
