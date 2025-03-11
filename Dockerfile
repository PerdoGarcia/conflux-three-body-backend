# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory
RUN mkdir -p /app/nltk_data

# Download necessary NLTK packages
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data'); nltk.download('punkt', download_dir='/app/nltk_data')"

# Copy the rest of the application
COPY . .

# Set up specific logging configuration for better debugging
RUN mkdir -p /app/logs
ENV LOG_DIR=/app/logs

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application with verbose logging
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--log-level=debug", "app:app"]