FROM python:3.10-slim

# Install system dependencies for bitsandbytes and other libs
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command; user can override in docker-compose
CMD ["python", "train.py", "--help"]