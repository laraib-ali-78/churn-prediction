# Deployment Guide: Telecom Churn Intelligence Platform

This repository is pre-configured for instant zero-configuration deployment across multiple cloud platforms.

---

## 1. Primary Path: Streamlit Community Cloud (Recommended & Free)

Deploying directly from your GitHub repository takes less than 2 minutes.

### Step-by-Step Instructions:
1. Push your repository to GitHub (e.g. `https://github.com/<your-username>/churn-prediction`).
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.
3. Click the blue **"Create app"** or **"New app"** button.
4. Select **"I already have an app"**.
5. In the configuration dialog, configure:
   - **Repository**: `<your-username>/churn-prediction`
   - **Branch**: `main`
   - **Main file path**: `app/app.py`
   - **App URL**: `churn-prediction-<custom-suffix>.streamlit.app` (or default generated)
6. Under **Advanced settings**, set **Python version**: `3.11`.
7. Click **"Deploy!"**.
8. Within 60 seconds, your application will build dependencies from `requirements.txt` and launch live.

Expected Public URL Pattern:
`https://<your-username>-churn-prediction.streamlit.app`

---

## 2. Fallback Option A: Hugging Face Spaces (Streamlit SDK)

1. Create a free account on [huggingface.co](https://huggingface.co).
2. Click on your profile -> **"New Space"**.
3. Fill in:
   - **Space Name**: `churn-prediction`
   - **License**: `mit`
   - **Select Space SDK**: `Streamlit`
   - **Space Hardware**: `CPU basic (Free)`
4. Clone the newly created Hugging Face Space repository or connect your GitHub repository.
5. Ensure `requirements.txt` and `app/app.py` are present.
6. Set the `app_file` in the Space `README.md` frontmatter:
   ```yaml
   ---
   title: Telecom Churn Prediction
   emoji: 📡
   colorFrom: blue
   colorTo: red
   sdk: streamlit
   app_file: app/app.py
   pinned: false
   ---
   ```
7. Hugging Face Spaces will automatically build the environment and host your app live.

---

## 3. Fallback Option B: Render (Web Service via Docker)

1. Sign up for a free account at [render.com](https://render.com).
2. Connect your GitHub account and click **"New +"** -> **"Web Service"**.
3. Select your `churn-prediction` repository.
4. Render will detect the included `Dockerfile` automatically:
   - **Environment**: `Docker`
   - **Region**: Nearest to your users (e.g., Oregon or Frankfurt)
   - **Branch**: `main`
   - **Plan type**: `Free`
5. Click **"Create Web Service"**.
6. Render builds the Docker container and exposes port `8501`. Your live web app will be available at `https://churn-prediction.onrender.com`.
