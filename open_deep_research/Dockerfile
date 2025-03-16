FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Plotly separately to ensure it's properly installed
RUN pip install --no-cache-dir plotly>=5.15.0

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Use start.sh to properly validate API keys and start the dashboard
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set the entry point to our startup script
ENTRYPOINT ["/app/start.sh"]
