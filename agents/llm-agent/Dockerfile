FROM python:3.9-slim

# Set working directory to /app
WORKDIR /llm-agent

# Copy all files from the llm-agent directory into the container
COPY . .

# Install dependencies. Make sure your agent code uses these libraries.
RUN pip install --no-cache-dir openai fastapi uvicorn requests python-dotenv


# Expose port 5000 for the LLM agent API
EXPOSE 5080

# Run the FastAPI app. This assumes you have a main.py with a FastAPI app called "app".
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5080"]