# Setup Guide

Follow these steps to configure the Multi-Agent Data Analyst with Google Sheets integration.

## Prerequisites

- Python 3.10+
- A Google Cloud account
- A Google Drive folder with spreadsheets you want to analyze

---

## Step 1: Install uv (if not installed)

```bash
pip install uv
```

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that includes `uvx` for running Python tools directly.

---

## Step 2: Test MCP server runs

Verify the mcp-google-sheets server can be launched:

```bash
uvx mcp-google-sheets@latest
```

This should start the MCP server. Press `Ctrl+C` to stop it. If you see errors, check that uv is properly installed.

---

## Step 3: Set up Google Cloud

### 3.1 Create a Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown at the top
3. Click "New Project"
4. Enter a name and click "Create"

### 3.2 Enable APIs

1. Go to **APIs & Services > Library**
2. Search for and enable:
   - **Google Sheets API**
   - **Google Drive API**

### 3.3 Create a Service Account

1. Go to **IAM & Admin > Service Accounts**
2. Click **"Create Service Account"**
3. Enter a name (e.g., `sheets-mcp-agent`)
4. Click "Create and Continue"
5. For role, select **Editor**
6. Click "Done"

### 3.4 Download the JSON Key

1. Click on your service account in the list
2. Go to the **Keys** tab
3. Click **Add Key > Create new key**
4. Select **JSON** and click "Create"
5. The key file will download automatically
6. Rename it to `service_account.json`
7. Move it to `credentials/service_account.json` in this project

---

## Step 4: Share your Google Drive folder with the service account

1. Open `credentials/service_account.json` and find the `client_email` field
   - It looks like: `your-service@your-project.iam.gserviceaccount.com`
2. Go to Google Drive
3. Navigate to the folder containing your spreadsheets
4. Right-click the folder → "Share"
5. Paste the service account email
6. Select **Editor** access
7. Click "Send" (uncheck "Notify people" if prompted)

---

## Step 5: Copy the folder ID into .env

1. In Google Drive, open the folder you shared
2. Look at the URL: `https://drive.google.com/drive/folders/THIS_PART_IS_THE_ID`
3. Copy the folder ID (the part after `/folders/`)
4. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` and set:
   ```
   OPENAI_API_KEY=sk-your-actual-key
   SERVICE_ACCOUNT_PATH=./credentials/service_account.json
   DRIVE_FOLDER_ID=your-folder-id-from-step-3
   ```

---

## Step 6: Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Step 7: Run the app

```bash
streamlit run app.py
```

The app will open in your browser. Ask questions about your spreadsheets in natural language!

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Service account file not found" | Check that `credentials/service_account.json` exists |
| "DRIVE_FOLDER_ID not configured" | Add the folder ID to your `.env` file |
| "Permission denied" | Share the Drive folder with the service account email |
| "API not enabled" | Enable Google Sheets API and Google Drive API in Cloud Console |
| "No tools found" | Run `uvx mcp-google-sheets@latest` to verify the MCP server works |

---

## Architecture

```
┌──────────────┐     stdio      ┌───────────────────┐     API     ┌───────────────┐
│  Your App    │ ◄────────────► │ mcp-google-sheets │ ◄─────────► │ Google Sheets │
│ (Streamlit)  │                │   (MCP Server)    │             │ Google Drive  │
└──────────────┘                └───────────────────┘             └───────────────┘
```

The app communicates with mcp-google-sheets via stdio transport. The MCP server handles all Google API authentication using your service account credentials.
