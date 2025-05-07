# Base image with Python installed
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install any Python dependencies (each participant can provide their own requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to Python
ENTRYPOINT ["python"]
