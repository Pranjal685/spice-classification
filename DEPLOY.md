# Deployment Guide: Hugging Face Spaces

This guide outlines the precise steps to deploy the containerized Spice Classification system directly to Hugging Face Spaces using their Docker SDK.

## 1. Prerequisites
Ensure you have created a Space on Hugging Face:
- **Type:** Docker (Blank)
- **Settings:** Ensure you have added your `GEMINI_API_KEY` to the Space's **Variables and Secrets** settings.

## 2. Git Initialization and Deployment

Run the following commands in your local project terminal. Be sure to replace `<USERNAME>` and `<SPACE_NAME>` with your actual Hugging Face username and Space name.

```bash
# 1. Initialize Git and commit your final codebase
git init
git add .
git commit -m "chore: initial production release for HF Spaces"

# 2. Add the Hugging Face Space as a remote origin
git remote add origin https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>

# 3. Push to the main branch to trigger the automated Docker build
git push -u origin main
```

*(Note: If Hugging Face asks for authentication, use a Write-access Access Token generated from your HF account settings instead of your account password).*

## 3. Verify Live Deployment

Once the Hugging Face Space finishes its "Building" and "Starting" phases:
1. **Access the Live UI:** Navigate to `https://<USERNAME>-<SPACE_NAME>.hf.space/` to test the frontend.
2. **Access the API Docs:** Navigate to `https://<USERNAME>-<SPACE_NAME>.hf.space/docs` to test the FastAPI interactive Swagger UI.
3. **Health Check:** Navigate to `https://<USERNAME>-<SPACE_NAME>.hf.space/api/v1/health` to confirm `fallback_ready` is true (validating that the environment secret was injected properly).
