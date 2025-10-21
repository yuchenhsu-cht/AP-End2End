
import numpy as np
from PIL import Image

def generate_test_image(file_path, digit_shape):
    """
    Generates a simple 28x28 PNG image with a specified shape on a white background.
    The digit will be black.
    """
    # Create a white background
    image_array = np.full((28, 28), 255, dtype=np.uint8)
    
    # Draw the digit shape in black
    for r, c in digit_shape:
        if 0 <= r < 28 and 0 <= c < 28:
            image_array[r, c] = 0
            
    # Create an image from the array
    img = Image.fromarray(image_array, 'L')
    img.save(file_path)
    print(f"Test image saved to {file_path}")

if __name__ == '__main__':
    # Define a simple shape for the digit '7'
    # This is a crude representation for testing purposes
    shape_of_7 = []
    # Top horizontal line
    for i in range(5, 20):
        shape_of_7.append((6, i))
    # Diagonal line
    for i in range(7, 22):
        shape_of_7.append((i, 19 - (i - 7)))

    generate_test_image("tests/test_digit_7.png", shape_of_7)
