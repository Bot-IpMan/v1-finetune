FROM python:3.10-slim

WORKDIR /app

# Install minimal dependencies for serving the model
# We pin versions to ensure compatibility with the training environment.
RUN pip install --no-cache-dir torch>=2.1.0 transformers>=4.40.0 peft>=0.8.2 fastapi>=0.110.0 uvicorn[standard]>=0.27.0

# Copy server code into the image
COPY api_server.py .

# Expose the port used by the API
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]