# Use stable Python version
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
