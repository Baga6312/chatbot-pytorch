# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /app/venv

# Activate venv and install dependencies
COPY requirements.txt .
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy project files
COPY . .

# Expose port (if you plan to add a web interface later)
# EXPOSE 5000

# Run the chatbot
CMD ["python", "chat.py"]
