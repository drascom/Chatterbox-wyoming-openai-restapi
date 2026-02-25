FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create required directories for the application
RUN mkdir -p model_cache reference_audio outputs voices logs \

# Expose service ports used by split roles:
# - TTS API/UI: 8000
# - Wyoming TTS: 10200
# - Wyoming STT: 10300
# - Internal STT HTTP API: 10400
EXPOSE 8000 10200 10300 10400

# Command to run the application
CMD ["python3", "server.py"]
