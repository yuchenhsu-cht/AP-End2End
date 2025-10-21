
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import model
import io
import os

# Create the FastAPI app
app = FastAPI(
    title="MNIST Digit Recognition API",
    description="An API to predict handwritten digits from images.",
    version="1.0.0"
)

# Mount the static directory to serve UI files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Load the ML model at startup
ml_model = model.load_model()

@app.get("/", tags=["UI"], summary="Serve the user interface")
def read_root():
    """
    Serves the main HTML page for the user interface.
    """
    return FileResponse(os.path.join(static_dir, "index.html"))

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
