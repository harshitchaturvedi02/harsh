# ML Ticket Router - Deployment Guide

## Overview

This guide covers deployment options for the ML-based ticket routing system, including local development, Docker deployment, and production deployment strategies.

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Redis (for caching)
- PostgreSQL (optional, for production)
- 8GB+ RAM recommended
- 10GB+ disk space for models and data

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ml-ticket-router
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Train Initial Model

```bash
# Generate synthetic data and train model
python scripts/train_model.py --synthetic --n-samples 10000
```

### 6. Start Redis (if not using Docker)

```bash
redis-server
```

### 7. Run the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Docker Deployment

### 1. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f ml-router

# Stop services
docker-compose down
```

### 2. Train Model in Docker

```bash
# Execute training script in container
docker-compose exec ml-router python scripts/train_model.py --synthetic
```

### 3. Access Services

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Production Deployment

### 1. Kubernetes Deployment

Create Kubernetes manifests:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-ticket-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-ticket-router
  template:
    metadata:
      labels:
        app: ml-ticket-router
    spec:
      containers:
      - name: ml-router
        image: ml-ticket-router:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-ticket-router-service
spec:
  selector:
    app: ml-ticket-router
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
kubectl apply -f deployment.yaml
```

### 2. AWS Deployment

#### Using ECS (Elastic Container Service)

1. Push image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com
docker tag ml-ticket-router:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/ml-ticket-router:latest
docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/ml-ticket-router:latest
```

2. Create ECS task definition and service

#### Using Lambda (Serverless)

1. Install Mangum for ASGI to AWS Lambda adapter:
```bash
pip install mangum
```

2. Create Lambda handler:
```python
# lambda_handler.py
from mangum import Mangum
from src.api.main import app

handler = Mangum(app)
```

3. Deploy using SAM or Serverless Framework

### 3. Google Cloud Platform

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/[PROJECT-ID]/ml-ticket-router

# Deploy to Cloud Run
gcloud run deploy ml-ticket-router \
  --image gcr.io/[PROJECT-ID]/ml-ticket-router \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

### 4. Azure Deployment

#### Using Azure Container Instances

```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name ml-ticket-router \
  --image ml-ticket-router:latest \
  --dns-name-label ml-ticket-router \
  --ports 8000 \
  --cpu 2 \
  --memory 4
```

## Production Considerations

### 1. Security

- Use HTTPS/TLS for all communications
- Implement proper API key rotation
- Use secrets management (AWS Secrets Manager, Azure Key Vault, etc.)
- Enable network policies and firewalls
- Regular security audits

### 2. Scaling

- Horizontal scaling with load balancer
- Auto-scaling based on CPU/memory metrics
- Consider GPU instances for neural network models
- Implement request queuing for batch processing

### 3. Monitoring

- Set up Prometheus alerts
- Configure Grafana dashboards
- Implement distributed tracing (Jaeger/Zipkin)
- Log aggregation (ELK stack)
- Error tracking (Sentry)

### 4. Model Management

- Implement A/B testing for new models
- Blue-green deployments
- Model versioning with MLflow
- Automated retraining pipelines
- Model performance monitoring

### 5. Data Management

- Regular database backups
- Data retention policies
- GDPR compliance
- Audit logging

## Performance Optimization

### 1. Caching Strategy

- Redis for prediction caching
- Model warm-up on startup
- Feature caching for repeated requests

### 2. Model Optimization

- Model quantization for smaller size
- ONNX conversion for faster inference
- Batch prediction endpoints
- GPU acceleration for large volumes

### 3. API Optimization

- Request/response compression
- Connection pooling
- Async processing for long operations
- Rate limiting per client

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase container memory limits
   - Reduce model size or batch size
   - Implement model pruning

2. **Slow Predictions**
   - Check Redis connectivity
   - Verify model is loaded in memory
   - Profile code for bottlenecks

3. **High Error Rate**
   - Check logs for specific errors
   - Verify all dependencies are installed
   - Ensure models are properly loaded

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check model performance
curl http://localhost:8000/api/v1/model/performance \
  -H "X-API-Key: your-api-key"

# View metrics
curl http://localhost:8000/metrics
```

## Backup and Recovery

### 1. Model Backup

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz data/models/

# Restore models
tar -xzf models-backup-20231201.tar.gz
```

### 2. Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U user -d tickets > tickets-backup.sql

# Restore
psql -h localhost -U user -d tickets < tickets-backup.sql
```

## Cost Optimization

1. **Right-sizing instances** based on actual usage
2. **Spot instances** for training workloads
3. **Reserved instances** for production
4. **Auto-scaling** to handle variable load
5. **Model caching** to reduce compute

## Support

For issues and questions:
- Check logs: `docker-compose logs ml-router`
- Review documentation
- Submit issues to repository
- Contact support team