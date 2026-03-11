# Sentiment Analysis ML Service (Production-Style Deployment)

## Overview

This project demonstrates an end-to-end machine learning deployment pipeline:

- Trained a BERT-based sentiment analysis model
- Built a FastAPI inference service
- Containerized using Docker
- Added automated testing with pytest
- Implemented CI using GitHub Actions
- Deployed to Kubernetes (Minikube)
- Configured Ingress routing

The system follows production-style architecture patterns.

---

## Architecture

Client  
↓  
Ingress  
↓  
Kubernetes Service  
↓  
Deployment (Pods)  
↓  
FastAPI  
↓  
Inference Layer  
↓  
Trained BERT Model  

---

## Tech Stack

- Python
- PyTorch
- Transformers (HuggingFace)
- FastAPI
- Docker
- GitHub Actions (CI)
- Kubernetes (Minikube)
- Nginx Ingress

---

## Key Features

- Separation of training and inference
- Lazy model loading
- Health check endpoint (`/health`)
- Predict endpoint (`/predict`)
- Liveness & readiness probes in Kubernetes
- CI pipeline that runs tests on every push
- Docker image hosted on Docker Hub
- Kubernetes deployment with 2 replicas

---

## Running Locally (Docker)

```bash
docker build -t <dockerhub-username>/sentiment-service:latest .
docker run -p 8000:8000 <dockerhub-username>/sentiment-service:latest

```

Test:
```bash
curl http://127.0.0.1:8000/health
```

Test:
```bash

curl http://127.0.0.1:8000/health
```

Test:
```bash
curl http://127.0.0.1:8000/health
```

## CI Pipeline

On every push:
-Install dependencies
-Run pytest
-validate API logic without loading the full model

## Future Improvements

-Automated CD pipeline (image build + push)
-Model versioning
-Monitoring & logging
-Horizontal Pod Autoscaler
-Cloud deployment (EKS/GKE/AKS)