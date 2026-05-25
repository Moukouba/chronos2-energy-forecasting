# Energy Forecasting Pipeline - Deployment Guide

## Local Development Setup

### 1. Prerequisites
- Python 3.9+
- pip or conda
- Git

### 2. Installation Steps

```bash
# Clone repository
git clone <repository-url>
cd energy_forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Update config/config.yaml with your settings
nano config/config.yaml
```

### 4. Run Locally

```bash
# Terminal 1: Start API
python main.py api --port 8000

# Terminal 2: Start Dashboard
python main.py dashboard

# Access:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

## Docker Deployment

### Single Container (API Only)

```bash
# Build image
docker build -t energy-forecast:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/outputs:/app/outputs \
  energy-forecast:latest

# Access: http://localhost:8000
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs

# Stop services
docker-compose down
```

### Custom Docker Build

```bash
# Build for production (multi-stage)
docker build -t energy-forecast:prod --target production .

# Build for development
docker build -t energy-forecast:dev --target development .
```

## Cloud Deployment

### AWS EC2 with Docker

```bash
# SSH into instance
ssh -i key.pem ec2-user@instance-ip

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo usermod -aG docker ec2-user

# Clone and run
git clone <repo-url>
cd energy_forecast
docker-compose up -d

# Configure security group:
# - Inbound: 8000 (API), 8501 (Dashboard) from 0.0.0.0/0
```

### Google Cloud Platform

```bash
# Deploy to Cloud Run
gcloud run deploy energy-forecast \
  --source . \
  --platform managed \
  --region us-central1 \
  --port 8000

# Deploy Dashboard to Cloud Run
gcloud run deploy energy-forecast-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --port 8501 \
  --command streamlit run app.py
```

### Kubernetes Deployment

```bash
# Create deployment
kubectl apply -f k8s/deployment.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Scale replicas
kubectl scale deployment energy-forecast --replicas=3

# View pods
kubectl get pods -l app=energy-forecast
```

Example Kubernetes manifest (k8s/deployment.yaml):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: energy-forecast
spec:
  replicas: 3
  selector:
    matchLabels:
      app: energy-forecast
  template:
    metadata:
      labels:
        app: energy-forecast
    spec:
      containers:
      - name: api
        image: energy-forecast:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Production Configuration

### Environment Variables

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_DIR=/var/log/energy-forecast

# Data
DATA_PATH=/data/model_ready.parquet
OUTPUT_DIR=/models

# Security
SECRET_KEY=your-secret-key
API_KEY=your-api-key
```

### SSL/TLS Configuration

```bash
# Using Nginx reverse proxy
server {
    listen 443 ssl;
    server_name forecast.example.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring & Logging

```bash
# Using ELK Stack
# Elasticsearch for storage
# Logstash for processing
# Kibana for visualization

# Or use CloudWatch (AWS)
# Or use Stackdriver (GCP)
# Or use Azure Monitor (Azure)
```

### Backup & Recovery

```bash
# Backup volumes
docker-compose exec api tar czf - /app/outputs | \
  aws s3 cp - s3://backup-bucket/energy-forecast-$(date +%Y%m%d).tar.gz

# Restore from backup
aws s3 cp s3://backup-bucket/energy-forecast-latest.tar.gz - | \
  tar xz -C /app/outputs
```

## Performance Tuning

### API Server

```bash
# Increase workers for high traffic
python main.py api --workers 8

# Use uvicorn with gunicorn
gunicorn src.api.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Database Connection Pooling

```python
# Add to config for production
database:
  pool_size: 20
  max_overflow: 40
  pool_pre_ping: true
```

### Caching

```bash
# Enable Redis caching for predictions
cache:
  backend: redis
  url: redis://localhost:6379/0
  ttl: 3600
```

## Security Best Practices

1. **API Security**
   - Use API keys for authentication
   - Implement rate limiting
   - Use HTTPS/TLS
   - Validate all inputs

2. **Data Security**
   - Encrypt sensitive data at rest
   - Use secure key management
   - Implement access controls
   - Log all access attempts

3. **Code Security**
   - Keep dependencies updated
   - Use code scanning tools
   - Implement CI/CD security checks
   - Regular security audits

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs <container-id>

# Check port availability
docker ps
netstat -ltn | grep :8000

# Rebuild without cache
docker build --no-cache -t energy-forecast:latest .
```

### Out of memory
```bash
# Increase Docker memory limit
docker run -m 4g energy-forecast:latest

# Or in docker-compose.yaml
services:
  api:
    mem_limit: 4g
```

### Model loading fails
```bash
# Check model file exists
ls -la outputs/

# Check file permissions
chmod 644 outputs/*

# Rebuild model
python main.py train
```

## Scaling Strategies

1. **Horizontal Scaling**
   - Use load balancer (Nginx, HAProxy, AWS ELB)
   - Run multiple API instances
   - Use Docker Swarm or Kubernetes

2. **Vertical Scaling**
   - Increase server resources (CPU, RAM)
   - Use GPU for model inference
   - Optimize code for performance

3. **Caching**
   - Cache predictions with Redis
   - Use CDN for static assets
   - Implement query caching

## Monitoring & Alerts

```bash
# Health check endpoint
curl http://localhost:8000/health

# Prometheus metrics (if enabled)
curl http://localhost:8000/metrics

# Structured logging
tail -f logs/energy_forecast_*.log | jq '.'
```

## Support & Documentation

- API Documentation: http://localhost:8000/docs
- Repository: https://github.com/energy/forecasting
- Issues: https://github.com/energy/forecasting/issues
