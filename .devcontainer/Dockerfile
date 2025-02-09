# Use the official Ubuntu image as the base image
FROM ubuntu:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install packages
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install pandas numpy vectorbt yfinance

# Set environment variables so the virtual environment is used by default
ENV PATH="/venv/bin:$PATH"

# Set git global variables
RUN git config --global user.email "lminvernizzi@gmail.com" && \
    git config --global user.name "Leonardo Invernizzi"

# Set the working directory inside the container
WORKDIR /scripts

# Copy the .bashrc file to the home directory of the root user
COPY _bashrc /root/.bashrc

# Declare the mount point as a volume
VOLUME ["/scripts"]
