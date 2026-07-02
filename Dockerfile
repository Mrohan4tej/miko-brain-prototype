# 1. Base Image: Use a lightweight Python version
FROM python:3.9-slim

# 2. Set Environment Variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures logs appear immediately in the console
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set Working Directory inside the container
WORKDIR /app

# 4. Install System Dependencies
# 'build-essential' is often needed for compiling math libraries (numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python Dependencies
COPY requirements.txt .
# We use --no-cache-dir to keep the image small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the Application Code
# This copies everything (src, data, main.py) into /app
COPY . .

# 7. Expose the port FastAPI uses
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]