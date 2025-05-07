# Start from a lightweight Python base
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your data folder
COPY data/ ./data/

# Copy your app code
COPY . .

# Expose the port Cloud Run uses
ENV PORT 8080
EXPOSE 8080

# Launch the app
CMD ["python", "app.py"]