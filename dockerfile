# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all necessary files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn flask joblib

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
