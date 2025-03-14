FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir "langgraph-cli[inmem]" openrouter

# Expose the LangGraph server port
EXPOSE 2024

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
ENV TAVILY_API_KEY=${TAVILY_API_KEY}

CMD ["langgraph", "dev", "--host", "0.0.0.0"]
