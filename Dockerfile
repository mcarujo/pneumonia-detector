# Use an official Python runtime as a parent image
FROM python:3.7.5-stretch

# Install any needed packages specified in requirements.txt
RUN apt update -y && \
    apt install libgl1-mesa-glx -y 

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Defining the directory path
WORKDIR /app/app

# Run flask when the container launches
CMD ["flask", "run",  "--host=0.0.0.0"]
