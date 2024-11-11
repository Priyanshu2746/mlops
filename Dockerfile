# Use the official Python image with the required version
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and YOLO model weights to the container
COPY requirements.txt .
COPY yolov8n.pt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
