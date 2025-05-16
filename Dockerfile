# The recipe for building the container image for the Python application.

# Official lightweight Python image
FROM python:3.10-slim
## This tells Docker to use an official Python base image, version 3.11, with a slimmed-down Debian OS. 
## It's small (faster download/build) but fully functional. Slim omits unnecessary extras to keep the image size down. 

# Set environment variables for headless matplotlib 
ENV MPLBACKEND=Agg
# matplotlib needs a backend to render plots. "Agg" is a non-interactive backend used to save figures, not display them on screen.

# Install build dependencies (OS-level dependencies for matplotlib, numpy & scipy)
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app
## Keeps code organized and in one place in the container.

# Copy requirements and install Python dependencies
COPY requirements.txt .
## Copy the requirements.txt file to the working directory in the container to install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt
## Installs Python dependencies listed in requirements.txt.

# Copy the current directory contents into the container
COPY . .
## Copy the entire current directory (including your Python scripts) into the working directory in the container.

# Run the script when the container starts
CMD ["python", "Project2.py"]
## Tells Docker to run the Python script "Project2.py" when the container starts.