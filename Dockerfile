FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose the port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "email_triage_openenv.server.app:app", "--host", "0.0.0.0", "--port", "8000"]