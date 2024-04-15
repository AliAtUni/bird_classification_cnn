# Use an official Python runtime as a parent image
FROM python:3.11.8-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install MLflow and other necessary packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the port MLflow will use
EXPOSE 5000

# Command to start MLflow and the training script
CMD ["sh", "start.sh"]
