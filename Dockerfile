# Use a lightweight Python base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PostgreSQL and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to launch the dashboard
CMD ["streamlit", "run", "terminal.py", "--server.port=8501", "--server.address=0.0.0.0"]