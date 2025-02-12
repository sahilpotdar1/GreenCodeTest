# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    maven \
    default-jdk \
    software-properties-common \
    wget \
    git \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Google Test
RUN mkdir -p /usr/src/gtest && \
    cd /usr/src/gtest && \
    wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
    tar -xzf v1.14.0.tar.gz && \
    cd googletest-1.14.0 && \
    cmake . && \
    make && \
    make install && \
    ldconfig

# # Install NUnit (for .NET projects)
# RUN apt-get update && apt-get install -y \
#     mono-complete \
#     && rm -rf /var/lib/apt/lists/*

# Install Mono and NUnit
RUN apt-get update && \
apt-get install -y --no-install-recommends \
mono-complete \
nunit-console && \
rm -rf /var/lib/apt/lists/*

# Install Python dependencies and pytest
COPY requirements.txt .
# Install Python dependencies without hash checking
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --no-deps -r requirements.txt
# Install all Python dependencies, including missing ones
RUN pip install --no-cache-dir \
    flask \
    blinker \
    itsdangerous \
    Werkzeug \
    codecarbon[cli] \
    fief-client \
    questionary \
    rich \
    typer \
    pyyaml \
    svglib \
    pyhanko \
    xhtml2pdf \
    paramiko
# Copy the application code
COPY . .

# Install Flask for web app
RUN pip install Flask python-dotenv


# Set up tool paths in .env
RUN echo "PYTEST_PATH=$(which pytest)" >> .env
RUN echo "MAVEN_PATH=$(which mvn)" >> .env
RUN echo "JAVAC_PATH=$(which javac)" >> .env
RUN echo "NUNIT_PATH=$(which nunit-console)" >> .env
RUN echo "GTEST_PATH=/usr/src/gtest" >> .env
RUN echo "GTEST_CMAKE_PATH=$(which cmake)" >> .env

# Set additional environment variables
ENV EXPECTED_PASSWORD=gcrtrial135#

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Expose port for Flask
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT ["./entrypoint.sh"]
