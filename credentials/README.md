# Credentials Folder

This folder holds your Google Service Account JSON key file for authenticating with Google Sheets and Google Drive APIs.

## ⚠️ IMPORTANT: This folder is gitignored

**Never commit your service account JSON key to version control!** The `credentials/*.json` pattern is added to `.gitignore` to prevent accidental commits.

## How to Get Your Service Account JSON Key

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click the project dropdown at the top and select "New Project"
3. Give your project a name and click "Create"

### Step 2: Enable Required APIs

1. In your project, go to **APIs & Services > Library**
2. Search for and enable:
   - **Google Sheets API**
   - **Google Drive API**

### Step 3: Create a Service Account

1. Go to **IAM & Admin > Service Accounts**
2. Click **"Create Service Account"**
3. Enter a name (e.g., "sheets-mcp-agent") and click "Create and Continue"
4. For role, select **Editor** (or more restrictive if preferred)
5. Click "Done"

### Step 4: Download the JSON Key

1. Click on your newly created service account
2. Go to the **Keys** tab
3. Click **Add Key > Create new key**
4. Select **JSON** format
5. Click "Create" - the key file will download automatically
6. **Rename the file to `service_account.json`** and place it in this folder

### Step 5: Share Your Google Drive Folder

1. Open the JSON key file and find the `client_email` field (looks like: `your-service@your-project.iam.gserviceaccount.com`)
2. Go to Google Drive and navigate to the folder containing your spreadsheets
3. Right-click the folder and select "Share"
4. Add the service account email with **Editor** access
5. Copy the folder ID from the URL (the part after `/folders/`)
6. Add the folder ID to your `.env` file as `DRIVE_FOLDER_ID`

## File Structure

After setup, this folder should contain:

```
credentials/
├── .gitkeep
├── README.md
└── service_account.json  ← Your downloaded JSON key (gitignored)
```

## Troubleshooting

- **"Permission denied" errors**: Make sure you shared the Drive folder with the service account email
- **"API not enabled" errors**: Double-check that both Google Sheets API and Google Drive API are enabled
- **"File not found" errors**: Verify the path in your `.env` matches where you placed the JSON file
