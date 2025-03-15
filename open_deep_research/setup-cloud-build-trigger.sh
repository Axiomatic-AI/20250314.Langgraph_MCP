#!/bin/bash
set -e

# Define variables
PROJECT_ID="axiomatic-ai-001"  # Use the same project ID as the example
REPO_NAME="20250314.Langgraph_MCP"
BRANCH_NAME="master"
TRIGGER_NAME="photonics-research-dashboard-build-deploy"
REGION="us-central1"

# Display instructions
echo "=== Google Cloud Build Trigger Setup ==="
echo "This script will guide you through setting up a Cloud Build trigger."
echo "You will need to do this manually in the Google Cloud Console."
echo ""
echo "1. Go to Cloud Build > Triggers: https://console.cloud.google.com/cloud-build/triggers?project=${PROJECT_ID}"
echo "2. Click 'Create Trigger'"
echo "3. Set the following:"
echo "   - Name: ${TRIGGER_NAME}"
echo "   - Region: ${REGION}"
echo "   - Event: Push to a branch"
echo "   - Repository: ${REPO_NAME} (connect if not already connected)"
echo "   - Branch: ${BRANCH_NAME}"
echo "   - Configuration: Cloud Build configuration file (yaml or json)"
echo "   - Cloud Build configuration file location: cloudbuild.yaml"
echo ""
echo "After creating the trigger, you can manually run it or push to your branch."
echo ""
echo "To check the status of Cloud Run deployments:"
echo "./check-cloud-run-url.sh"

echo ""
echo "After deployment, access your dashboard at:"
echo "https://photonics-research-dashboard-[hash].a.run.app"
