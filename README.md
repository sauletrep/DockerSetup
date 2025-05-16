# DockerSetup

# Truncated Gaussian Random Fields in Docker

## Overview
I began this project by developing a Python script that visualizes the behavior of Gaussian Random Fields (GRFs) when truncated under different spatial correlation functions. This built upon a similar project I had previously implemented in R, with the goal of translating and improving it in Python.

## Project Structure

- `Project2.py`: Main Python script for simulation and plotting.
- `requirements.txt`: Lists the Python dependencies.
- `Dockerfile`: Instructions for building the container image.
- Output figures are saved to the local project folder during execution.

## Dockerization Process

### Step 1: Define Python Dependencies
Once the Python code was running correctly, I moved on to containerising the project using Docker.

After installing Docker and enabling Docker support in VS Code, an empty requirements.txt and a sample Dockerfile were automatically generated. Since my Python script uses numpy, matplotlib.pyplot and scipy, I wrote three libraries in the requirements.txt file. I checked the exact versions I had installed locally using the terminal to ensure compatibility.

### Step 2: Create the Dockerfile
Initially, I attempted to build an image using the auto-generated Dockerfile, but it failed. I cleared the file and wrote a minimal working Dockerfile from scratch, reusing the existing filename. The initial version of the Dockerfile included:

* Selecting a lightweight official Python base image (python:3.10-slim)

* Setting a working directory inside the container (/app)

* Copying the requirements.txt and installing dependencies via pip

* Copying the current directory contents

* Running the Python script

### Dockerfile Contents
```
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt. 
RUN pip install --no-binary=:all: -r requirements.txt

# Copy the current directory contents into the container
COPY..

# Run the script when the container starts
CMD ["python", "Project2.py"]
```

Even with the simplified Dockerfile, the build continued to fail because some required system-level packages were missing. Initially, I attempted to add these dependencies one by one as the build process threw errors like “No such file or directory,” but this approach was slow and inefficient, as each build took several minutes.

After researching the typical system requirements for numpy, scipy, and matplotlib, I included the following dependencies in the Dockerfile:

* build-essential: Includes GCC and other tools for compiling Python packages with C extensions.
* libfreetype6-dev: Required for rendering fonts in matplotlib plots.
* libpng-dev: Development files for PNG image support. Required by matplotlib to save figures in PNG format.
* pkg-config: Helps locate libraries during the build process.
* libopenblas-dev: Provides optimized linear algebra routines used by numpy and scipy.
* liblapack-dev: Another key library for numerical linear algebra. It's used by numpy and scipy for matrix operations.

### System-level build dependencies
```
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
```
Since matplotlib needs a backend to render plots, I also added environment variables for headless matplotlib with 
```
ENV MPLBACKEND=Agg
```
With these additions, the image built successfully, and the container was able to run the Python script without issues. To build I used the command
```
docker build --no-cache -t truncated-gaussian-random-fields .
```
This builds the image and names it truncated-gaussian-random-fields . I restricted to no cache to ensure all system-level dependencies were reinstalled. To verify that everything worked as expected, I ran a container from that image using the following command:
```
docker run --rm -v $(pwd):/app docker build --no-cache -t truncated-gaussian-random-fields
```
The --rm flag removes the container after it stops. I also mounted the current working directory into the container with -v $(pwd):/app so that any output files (e.g., plots) were saved locally. Since my script generates figures and saves them to disk, I was able to confirm that the container executed correctly by checking the output files. The figures were produced exactly as they were when running the code outside Docker, indicating that the environment was set up properly.

## Notes
During the process, Visual Studio Code's Docker extension automatically generated a few additional files to assist with container management and configuration:

.dockerignore: Tells Docker which files to exclude from the image (like cache files or version control folders).
docker-compose.yml: Used for defining multi-container setups (not needed for this simple project).
docker-compose.debug.yml: Supports debugging containers in development.

Although I didn’t actively use these files for my project, it was useful to see how the Docker extension prepares for more advanced use cases beyond a single-container workflow.

Additionally the image contains 1 high vulnerability. This is typical in slim base images.

# Summary
Through this process, I successfully containerized a Python project for simulating Truncated Gaussian Random Fields. The main challenge was identifying and installing the correct system-level dependencies required by matplotlib, numpy, and scipy when using a slim Python image.
