#!/bin/bash
set -e

# Define variables
PROJECT_ID="axiomatic-ai-001"
SERVICE_NAME="photonics-research-dashboard"
REGION="us-central1"

# Get the service URL
echo "Getting Cloud Run service URL..."
SERVICE_URL=$(./run-gcloud.sh run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format "value(status.url)")

echo "Your Photonics Research Dashboard is available at: ${SERVICE_URL}"

# Perform a preflight check to ensure the service is responding
echo "Performing preflight check..."
RESPONSE=$(docker run --rm curlimages/curl:7.83.1 -s -o /dev/null -w "%{http_code}" ${SERVICE_URL})

if [ "$RESPONSE" == "200" ]; then
  echo "✅ Service is up and running (HTTP 200)"
else
  echo "⚠️ Service returned HTTP ${RESPONSE}. It may still be starting up or there might be an issue."
fi
