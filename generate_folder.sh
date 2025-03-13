#!/bin/bash
# This script creates the project folder structure for the chatbot backend, agents,
# knowledge-base, and related config files.
# It should be run inside the project's root folder (e.g., project-root/).
# The frontend folder is intentionally excluded.

# Create directories
mkdir -p agents/query-orchestration-agent
mkdir -p agents/intent-agent
mkdir -p agents/llm-agent

mkdir -p knowledge-base/structured-data-agent
mkdir -p knowledge-base/unstructured-data-agent
mkdir -p knowledge-base/knowledge-graph-agent

mkdir -p backend/app/routes
mkdir -p backend/app/models

mkdir -p config
mkdir -p nginx
mkdir -p kubernetes

# Create files for agents
touch agents/query-orchestration-agent/__init__.py
touch agents/query-orchestration-agent/agent.py
touch agents/query-orchestration-agent/Dockerfile

touch agents/intent-agent/__init__.py
touch agents/intent-agent/agent.py
touch agents/intent-agent/Dockerfile

touch agents/llm-agent/__init__.py
touch agents/llm-agent/agent.py
touch agents/llm-agent/config.py
touch agents/llm-agent/Dockerfile

# Create files for knowledge-base
touch knowledge-base/structured-data-agent/__init__.py
touch knowledge-base/structured-data-agent/agent.py
touch knowledge-base/structured-data-agent/Dockerfile

touch knowledge-base/unstructured-data-agent/__init__.py
touch knowledge-base/unstructured-data-agent/agent.py
touch knowledge-base/unstructured-data-agent/Dockerfile

touch knowledge-base/knowledge-graph-agent/__init__.py
touch knowledge-base/knowledge-graph-agent/agent.py
touch knowledge-base/knowledge-graph-agent/Dockerfile

# Create files for backend
touch backend/app/__init__.py
touch backend/app/main.py
touch backend/app/routes/__init__.py
touch backend/app/routes/chat.py
touch backend/app/models/message.py
touch backend/requirements.txt
touch backend/Dockerfile

# Create config files
touch config/config.yaml
touch config/.env

# Create nginx configuration file
touch nginx/default.conf

# Create root-level files
touch docker-compose.yml

# Create Kubernetes configuration files
touch kubernetes/backend-deployment.yaml
touch kubernetes/backend-service.yaml
touch kubernetes/frontend-deployment.yaml
touch kubernetes/frontend-service.yaml
touch kubernetes/nginx-deployment.yaml
touch kubernetes/nginx-config.yaml
touch kubernetes/ingress.yaml

# Output success message
echo "Project folder structure created successfully (excluding the frontend folder)."
