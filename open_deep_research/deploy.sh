#!/bin/bash
set -e

# Define variables
PROJECT_ID="axiomatic-ai-001"  # Use the same project ID as the example
SERVICE_NAME="photonics-research-dashboard"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="${TIMESTAMP}"

# Authenticate with Google Container Registry
echo "Authenticating with Google Container Registry..."
./run-gcloud.sh auth configure-docker gcr.io

# Build and tag the Docker image
echo "Building the Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} -t ${IMAGE_NAME}:latest .

# Push the images to Google Container Registry
echo "Pushing the Docker images to Google Container Registry..."
docker push ${IMAGE_NAME}:${TAG}
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
./run-gcloud.sh run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:${TAG} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars-file .env

# Get the service URL
echo "Deployment completed! Getting service URL..."
SERVICE_URL=$(./run-gcloud.sh run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format "value(status.url)")

echo "Your Photonics Research Dashboard is available at: ${SERVICE_URL}"
