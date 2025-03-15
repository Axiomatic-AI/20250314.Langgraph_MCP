#!/bin/bash

# This script runs gcloud commands using a Docker container for proper isolation

# Pull the Google Cloud SDK image if not already available
docker pull gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Run the gcloud command passed as arguments
docker run --rm \
  -v "$HOME/.config/gcloud:/root/.config/gcloud" \
  gcr.io/google.com/cloudsdktool/cloud-sdk:slim \
  gcloud "$@"
