# =============================================================================
# DOCKERFILE - LangGraph Studio Environment
# =============================================================================
# This Dockerfile creates an environment for running LangGraph Studio.
# It installs Python and all required dependencies from requirements.txt.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "langgraph-cli[inmem]"

# Copy the rest of the application
COPY . .

# Expose the LangGraph Studio port
EXPOSE 8000

# Run LangGraph dev server
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8000"]
