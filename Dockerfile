# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for PyTorch and Transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic langgraph langchain langchain-google-genai mcp torch transformers sentence-transformers

# Copy the entire project structure into the container
COPY . .

# Expose the port the FastAPI server runs on
EXPOSE 8000

# Change working directory to where the app runs
WORKDIR /app/agent_api

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]