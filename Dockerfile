# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/

# Replace "torch==2.3.0" with the CPU-only version of PyTorch, which is smaller in size
RUN sed -i 's/torch==2.3.0/torch @ https:\/\/download.pytorch.org\/whl\/cpu\/torch-2.3.0%2Bcpu-cp310-cp310-linux_x86_64.whl/' requirements.txt

# Install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download and unzip the test suite
RUN apt-get update && apt-get install -y wget unzip \
    && wget https://samate.nist.gov/SARD/downloads/test-suites/2015-10-27-php-vulnerability-test-suite.zip \
    && unzip 2015-10-27-php-vulnerability-test-suite.zip \
    && rm 2015-10-27-php-vulnerability-test-suite.zip \
    && apt-get remove -y wget unzip \
    && apt-get autoremove -y \
    && apt-get clean

# Copy the current directory contents into the container at /app
COPY *.py /app

# Run main.py when the container launches
CMD ["python", "main.py"]