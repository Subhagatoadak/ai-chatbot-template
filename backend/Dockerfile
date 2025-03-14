FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies from the backend requirements file.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend and agents directories from the project root.
COPY . .


# Set PYTHONPATH to include the project root, so that the agents module is discoverable.
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]
