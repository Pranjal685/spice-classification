# Deployment Guide: Render.com (Free Docker Hosting)

Since Hugging Face Spaces may require a paid tier for certain Docker configurations, **Render.com** is the recommended alternative. It provides a robust **Free Web Service** tier that perfectly supports Dockerized FastAPI applications.

Since your code is already pushed to GitHub, deploying to Render takes just a few clicks.

## 1. Prerequisites
- A GitHub account (which you already have).
- Create a free account on [Render.com](https://render.com/).

## 2. Deploying to Render
1. Log in to Render and click **New +** -> **Web Service**.
2. Under "Connect a repository", select your GitHub account and find the **`spice-classification`** repository. Click **Connect**.
3. Fill out the deployment settings:
   - **Name:** `spice-classification` (or any name you prefer)
   - **Region:** Choose the region closest to you.
   - **Branch:** `master` (or `main` depending on your default branch)
   - **Environment:** Select **Docker**.
   - **Instance Type:** Select **Free** (0.1 CPU, 512 MB RAM).
4. Expand the **Advanced** section to add Environment Variables:
   - Click **Add Environment Variable**.
   - **Key:** `GEMINI_API_KEY`
   - **Value:** *(Paste your Google Gemini API Key here)*
5. Click **Create Web Service**.

## 3. Monitor and Access
- Render will now pull your repository, build the Docker container (optimizing via the slim CPU configurations we set up in the Dockerfile), and start the FastAPI Uvicorn server.
- The build process usually takes 2-5 minutes. You can monitor the logs directly in the Render dashboard.
- Once the status says **Live**, you can access your application at the provided URL (e.g., `https://spice-classification-xxxx.onrender.com`).
- To access the interactive API docs, append `/docs` to the URL.

*(Note: Render's free tier spins down after 15 minutes of inactivity. The first request after it sleeps may take 30-60 seconds to spin back up, but subsequent requests will be instant).*
