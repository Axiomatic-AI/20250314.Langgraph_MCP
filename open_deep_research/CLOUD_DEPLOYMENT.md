# Photonics Research Dashboard Cloud Deployment Guide

This guide provides instructions for deploying the Photonics Research Dashboard to Google Cloud Platform using Cloud Build and Cloud Run. All operations are performed within Docker containers to ensure proper isolation and security.

## Prerequisites

1. A Google Cloud account with access to the `axiomatic-ai-001` project
2. The following APIs enabled in your project:
   - Cloud Build API
   - Cloud Run API
   - Secret Manager API
   - Container Registry API
3. GitHub repository connected to Cloud Build

## Deployment Options

### Option 1: Automated Deployment with Cloud Build Trigger (Recommended)

This option sets up a CI/CD pipeline that automatically deploys your application whenever you push to GitHub.

1. **Connect your GitHub repository to Cloud Build**
   - Go to [Cloud Build > Triggers](https://console.cloud.google.com/cloud-build/triggers?project=axiomatic-ai-001)
   - Click "Connect Repository"
   - Select "GitHub (Cloud Build GitHub App)"
   - Follow the prompts to authenticate with GitHub and select your repository

2. **Create a Cloud Build trigger**
   - After connecting your repository, click "Create Trigger"
   - Set the following:
     - Name: `photonics-research-dashboard-build-deploy`
     - Region: `us-central1`
     - Event: Push to a branch
     - Repository: `20250314.Langgraph_MCP`
     - Branch: `^master$`
     - Configuration: Cloud Build configuration file (yaml or json)
     - Cloud Build configuration file location: `cloudbuild.yaml`

3. **Set up secrets in Secret Manager**
   - Go to [Secret Manager](https://console.cloud.google.com/security/secret-manager?project=axiomatic-ai-001)
   - Create the following secrets:
     - `ANTHROPIC_API_KEY`: Your Anthropic API key
     - `TAVILY_API_KEY`: Your Tavily API key
     - `OPENROUTER_API_KEY`: Your OpenRouter API key (optional)

4. **Run the trigger**
   - Go to Cloud Build > Triggers
   - Find your trigger and click "Run Trigger"
   - Select the branch (master) and click "Run"

### Option 2: Manual Deployment

If you prefer to deploy manually, you can use the provided scripts:

1. **Ensure Docker is running** on your local machine

2. **Run the deployment script**
   ```bash
   ./deploy.sh
   ```

3. **Check the deployment status**
   ```bash
   ./check-cloud-run-url.sh
   ```

## Accessing Your Deployed Dashboard

After deployment, your dashboard will be available at a URL like:
```
https://photonics-research-dashboard-[hash].a.run.app
```

To get the exact URL, run:
```bash
gcloud run services describe photonics-research-dashboard --platform managed --region us-central1 --format "value(status.url)"
```

## Troubleshooting

### API Keys
- Ensure all required API keys are properly set in Secret Manager
- Check Cloud Build logs for any errors related to missing or invalid API keys

### Deployment Issues
- Check Cloud Build logs for build or deployment errors
- Ensure your Dockerfile is properly configured
- Verify that the Cloud Run service has enough memory and CPU allocated

### Performance Optimization
- The dashboard is configured to use Model Context Protocol (MCP) to minimize resource usage
- Memory usage is kept lean and should not exceed 10 GB per workspace
- Preflight checks are implemented to avoid running programs that will fail

## Security Notes

- All software runs within Docker containers to ensure proper isolation and security
- API keys are stored securely in Secret Manager and never exposed in code
- The Cloud Run service is configured with appropriate permissions
