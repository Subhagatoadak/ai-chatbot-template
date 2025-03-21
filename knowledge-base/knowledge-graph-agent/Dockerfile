# Use an official Python runtime as a parent image
FROM python:3.9

# Install system dependencies required for building spaCy and its dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /knowledge-graph-agent

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# Copy application code into the container
COPY data/ data/
COPY agent.py .

# Expose port 8000
EXPOSE 8001

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8001"]
