FROM python:3.9-slim

WORKDIR /weaviate-agent

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir weaviate-client
RUN pip install --upgrade --no-cache-dir weaviate-client

# Copy the application code
COPY . .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
