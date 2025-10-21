import pytest
import numpy as np
from model import load_model, preprocess_image, predict, MODEL_FILE
import os
import io

# Fixture to load the model once for all tests
@pytest.fixture(scope="module")
def trained_model():
    # Ensure the model exists, if not, this will train it.
    model = load_model()
    return model

# Path to the test image
TEST_IMAGE_PATH = "tests/test_digit_7.png"

def test_preprocess_image_valid():
    """
    Tests that a valid image is preprocessed into the correct shape (1, 784).
    """
    assert os.path.exists(TEST_IMAGE_PATH), "Test image is missing!"
    
    processed_image = preprocess_image(TEST_IMAGE_PATH)
    
    assert processed_image is not None, "Preprocessing failed for a valid image."
    assert isinstance(processed_image, np.ndarray), "Preprocessed image is not a numpy array."
    assert processed_image.shape == (1, 784), f"Incorrect image shape: {processed_image.shape}"
    assert np.max(processed_image) <= 1.0, "Image not normalized correctly."
    assert np.min(processed_image) >= 0.0, "Image not normalized correctly."

def test_preprocess_image_invalid_file():
    """
    Tests that preprocessing returns None for a non-existent file.
    """
    processed_image = preprocess_image("tests/non_existent_image.png")
    assert processed_image is None, "Preprocessing should fail for a non-existent file."

def test_preprocess_image_corrupted_file(tmp_path):
    """
    Tests that preprocessing returns None for a file that is not a valid image.
    """
    # Create a dummy non-image file
    p = tmp_path / "not_an_image.txt"
    p.write_text("this is not an image")
    processed_image = preprocess_image(str(p))
    assert processed_image is None

def test_predict_valid(trained_model):
    """
    Tests the prediction function with a valid, preprocessed image.
    """
    # Preprocess the valid test image
    processed_image = preprocess_image(TEST_IMAGE_PATH)
    assert processed_image is not None, "Test setup failed: could not preprocess image."
    
    # Get prediction
    prediction, confidence = predict(trained_model, processed_image)
    
    assert prediction is not None, "Prediction is None for a valid image."
    assert confidence is not None, "Confidence is None for a valid image."
    
    assert isinstance(prediction, str), "Prediction should be a string."
    assert isinstance(confidence, float), "Confidence should be a float."
    
    # While we can't guarantee it predicts '7', we can check it's a single digit.
    assert prediction.isdigit() and len(prediction) == 1, "Prediction is not a single digit."
    assert 0.0 <= confidence <= 1.0, "Confidence score is out of range [0, 1]."

def test_predict_invalid_input(trained_model):
    """
    Tests that prediction returns None, None for invalid input.
    """
    prediction, confidence = predict(trained_model, None)
    assert prediction is None
    assert confidence is None

def test_predict_exception(trained_model):
    """
    Tests the exception handling in the predict function.
    """
    # Create a numpy array with an incorrect shape that will cause an error
    malformed_input = np.zeros((1, 100))
    prediction, confidence = predict(trained_model, malformed_input)
    assert prediction is None
    assert confidence is None

def test_load_model_creates_new_model(tmp_path):
    """
    Tests that load_model() trains a new model if one doesn't exist.
    """
    # Temporarily move the real model file if it exists
    original_model_path = MODEL_FILE
    backup_model_path = os.path.join(tmp_path, "mnist_model.joblib.bak")
    
    model_existed = False
    if os.path.exists(original_model_path):
        model_existed = True
        os.rename(original_model_path, backup_model_path)

    try:
        # This should now trigger the training path
        # We pass a mock to avoid the actual training time in a unit test
        # For this test, we will let it run to ensure it works end-to-end
        new_model = load_model()
        assert new_model is not None
        assert os.path.exists(original_model_path), "load_model did not create a new model file."
    finally:
        # Cleanup: move the original model back
        if model_existed:
            # On Windows, os.rename will fail if the destination exists.
            # So, we remove the newly created file before restoring the backup.
            if os.path.exists(original_model_path):
                os.remove(original_model_path)
            os.rename(backup_model_path, original_model_path)