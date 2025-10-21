
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import model
import io

# Create the FastAPI app
app = FastAPI(
    title="MNIST Digit Recognition API",
    description="An API to predict handwritten digits from images.",
    version="1.0.0"
)

# Load the ML model at startup
ml_model = model.load_model()

@app.get("/", tags=["General"], summary="Health check endpoint")
def read_root():
    """
    A simple health check endpoint to confirm the service is running.
    """
    return {"message": "Welcome to the MNIST Digit Recognition API!"}

@app.post("/predict/mnist", tags=["Prediction"], summary="Predict a digit from an image")
def predict_mnist(file: UploadFile = File(...) ):
    """
    Receives an image file (PNG, JPG, etc.), preprocesses it, and returns
    the predicted digit along with a confidence score.

    - **file**: The image file of a handwritten digit.
    """
    # Ensure the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the file content into a buffer
        image_data = io.BytesIO(file.file.read())

        # Preprocess the image
        processed_image = model.preprocess_image(image_data)
        if processed_image is None:
            raise HTTPException(status_code=400, detail="Could not preprocess the image. It might be in an unsupported format or corrupted.")

        # Make a prediction
        prediction, confidence = model.predict(ml_model, processed_image)

        if prediction is None:
            raise HTTPException(status_code=500, detail="Model failed to make a prediction.")

        # Return the result
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run the app locally, use the command:
# uvicorn main:app --reload
