
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# The model will be trained and saved on the first run if it doesn't exist.
# Ensure the container has permissions to write the model file.
RUN touch mnist_model.joblib && chmod 666 mnist_model.joblib

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the app
# Use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
