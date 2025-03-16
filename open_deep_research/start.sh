#!/bin/bash

# Startup script for Advanced Research Dashboard
# This script checks for the presence of required API keys and provides 
# helpful information if they are missing

# Check for .env file and load it if it exists
if [ -f ".env" ]; then
  echo "📄 Found .env file, loading environment variables..."
  export $(grep -v '^#' .env | xargs)
fi

# Check for Anthropic API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠️  ERROR: ANTHROPIC_API_KEY environment variable is not set"
  echo
  echo "You must provide a valid Anthropic API key to use this dashboard."
  echo "Options:"
  echo "1. Create a .env file in the project root with: ANTHROPIC_API_KEY=your_key_here"
  echo "2. Run the container with: docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your_key_here advanced-research-dashboard"
  echo
  echo "If you don't have an API key, you can get one at: https://console.anthropic.com/"
  exit 1
fi

# Start the Streamlit dashboard
echo "🚀 Starting Advanced Research Dashboard..."
streamlit run llm_advanced_dashboard.py
