FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY crop_ai_engine_v3.py .
COPY train_crop_model.py .

# Create necessary directories
RUN mkdir -p models training_data outputs

# Optional: Download and prepare dataset (commented out to keep image small)
# Uncomment if you want dataset included in Docker image
# RUN cd training_data && \
#     wget http://madm.dfki.de/files/sentinel/EuroSAT.zip && \
#     unzip EuroSAT.zip && \
#     rm EuroSAT.zip

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=crop_ai_engine_v3.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run application
CMD ["python", "crop_ai_engine_v3.py"]
