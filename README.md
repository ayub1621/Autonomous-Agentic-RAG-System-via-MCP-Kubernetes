# Autonomous Agentic RAG System via MCP & Kubernetes

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU/GPU-EE4C2C.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-000000.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Kubernetes](https://img.shields.io/badge/Deployment-Kubernetes-326CE5.svg)

An enterprise-grade, autonomous generative AI backend designed to serve intelligent routing and data synthesis in a highly scalable environment. This project demonstrates advanced capabilities in agentic orchestration, semantic routing, and cloud-native MLOps.

Unlike standard linear RAG pipelines, this system utilizes a cyclical **LangGraph** state machine and decoupled **Model Context Protocol (MCP)** servers, allowing the agent to dynamically access live financial APIs and simulated enterprise CRM databases based on real-time decision-making.

## 🏗️ Architecture

The system is broken down into modular microservices to ensure separation of concerns and high scalability:

1. **PyTorch Semantic Router:** A high-speed (sub-15ms) upstream classifier using `sentence-transformers/all-MiniLM-L6-v2`. It calculates cosine similarity between user query embeddings and pre-computed domain prototypes to bypass expensive LLM calls for simple routing decisions.
2. **MCP Data Servers:** Independent FastMCP endpoints exposing external data tools:
   - **Finance Server:** Real-time stock market data via `yfinance`.
   - **CRM Server:** Mock enterprise user data managed via an in-memory SQLite database.
3. **LangGraph Orchestrator:** The core state machine that manages conversation memory, executes the routed MCP tools, and synthesizes the final context using Google's Gemini GenAI models.
4. **FastAPI Backend:** An asynchronous REST API exposing the agentic workflow.
5. **Kubernetes Infrastructure:** Containerized via Docker and deployed to a `kind` cluster with declarative manifests.

## 📂 Project Structure

```text
autonomous-agent-system/
├── agent_api/               # FastAPI application and LangGraph orchestration
│   ├── app.py               # REST endpoints
│   └── graph.py             # LangGraph state machine definition
├── mcp_servers/             # FastMCP data tools
│   ├── crm_server.py        # SQLite enterprise database simulator
│   └── finance_server.py    # yfinance market data integration
├── pytorch_router/          # ML embedding router
│   ├── router.py            # Centroid-based intent classification
│   └── README.md            # Hugging Face Model Card
├── k8s_manifests/           # Kubernetes configuration files
│   └── deployment.yaml      # Pod deployment and NodePort service mapping
├── Dockerfile               # Unified container configuration
└── requirements.txt         # Project dependencies

```

## 🚀 Getting Started

### Prerequisites

* Docker Desktop
* `kind` (Kubernetes IN Docker)
* `kubectl`
* Python 3.10+
* A valid Google Gemini API Key

### Local Kubernetes Deployment

1. **Spin up the cluster:**
```bash
kind create cluster --name agentic-rag-cluster

```


2. **Build and load the Docker image:**
```bash
docker build -t agent-api:v1 .
kind load docker-image agent-api:v1 --name agentic-rag-cluster

```


3. **Deploy the application:**
*(Ensure you update the `GOOGLE_API_KEY` environment variable in `k8s_manifests/deployment.yaml` before applying)*
```bash
kubectl apply -f k8s_manifests/deployment.yaml

```


4. **Expose the service:**
```bash
kubectl port-forward svc/agent-api-service 8000:8000

```



## ⚡ API Usage

Once the pod is running and port-forwarded, you can access the interactive Swagger UI at `http://localhost:8000/docs` or send a POST request directly:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/invoke_agent' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the current trend for Apple stock?"
}'

```

**Expected JSON Response:**

```json
{
  "query": "What is the current trend for Apple stock?",
  "domain": "financial_data",
  "context": "{'symbol': 'AAPL', 'current_price': 150.25, '5_day_trend': 'Upward'}",
  "final_answer": "Based on the provided context, the trend for Apple stock (AAPL) is upward, with a current price of 150.25."
}

```

## 👨‍💻 Author

**Ayub**
Google Cloud Certified Professional Machine Learning Engineer

```

```
