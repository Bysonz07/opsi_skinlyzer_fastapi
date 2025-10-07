FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .

# Add Rust + compilers (for building pydantic-core)
RUN apt-get update && apt-get install -y curl gcc g++ rustc

RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
