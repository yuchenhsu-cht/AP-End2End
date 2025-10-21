
import numpy as np
from PIL import Image
import joblib
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os

MODEL_FILE = "mnist_model.joblib"

def train_and_save_model():
    """
    Fetches MNIST data, trains an MLPClassifier (neural network), and saves it.
    """
    print("Fetching MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    
    X = mnist.data
    y = mnist.target
    # Convert string labels to integers for MLPClassifier
    y = y.astype(np.uint8)
    
    # Normalize the data
    X = X / 255.0
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training MLPClassifier (Neural Network). This may take a few minutes...")
    # This is a simple neural network. For this dataset, it provides better accuracy.
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        max_iter=30, 
        random_state=42,
        verbose=True, # To see the training progress
        early_stopping=True # Stop training when validation score is not improving
    )
    clf.fit(X_train, y_train)
    
    print(f"Calculating accuracy...")
    accuracy = clf.score(X_test, y_test)
    print(f"New Model accuracy: {accuracy:.4f}")
    
    print(f"Saving new model to {MODEL_FILE}...")
    joblib.dump(clf, MODEL_FILE)
    print("Model saved.")

def load_model():
    """
    Loads the trained model from the file.
    If the model file doesn't exist, it trains and saves a new one.
    """
    if not os.path.exists(MODEL_FILE):
        print("Model file not found. Training a new model...")
        train_and_save_model()
        
    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    print("Model loaded.")
    return model

def preprocess_image(image_file):
    """
    Preprocesses an uploaded image file to be compatible with the MNIST model.
    - Opens the image
    - Converts to grayscale ('L')
    - Resizes to 28x28 pixels
    - Converts to a numpy array
    - Inverts colors (MNIST is white digit on black background)
    - Normalizes pixel values to be between 0 and 1
    - Flattens the 28x28 image to a 1D array of 784 features
    """
    try:
        img = Image.open(image_file).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # Invert colors if necessary. Standard MNIST is white digit on black bg.
        # Most user-drawn images are black digit on white bg.
        # We check the average color of the corners vs the center.
        corners_mean = np.mean(img_array[:5, :5]) + np.mean(img_array[-5:, -5:])
        center_mean = np.mean(img_array[10:18, 10:18])
        
        # If corners are brighter than the center, it's likely black on white, so invert.
        if corners_mean > center_mean:
            img_array = 255 - img_array

        img_array = img_array / 255.0
        
        # Flatten the image to a 1D array of 784 features
        img_flat = img_array.flatten()
        
        return img_flat.reshape(1, -1) # Reshape for a single prediction
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict(model, processed_image):
    """
    Makes a prediction on a preprocessed image.
    Returns the predicted digit and the confidence score.
    """
    if processed_image is None:
        return None, None
        
    try:
        probabilities = model.predict_proba(processed_image)
        prediction = model.classes_[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        return str(prediction), confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Main guard to run training
if __name__ == '__main__':
    train_and_save_model()
