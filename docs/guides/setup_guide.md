# MLOps Demo Setup Guide

This guide will help you set up the complete MLOps demo environment on your local machine.

## 🔧 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **CPU**: Multi-core processor recommended

### Required Software

#### 1. Docker & Docker Compose
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

#### 2. Python 3.9+
```bash
# Check Python version
python3 --version

# Install pip if needed
sudo apt-get install python3-pip  # Ubuntu/Debian
# or
brew install python3  # macOS
```

#### 3. Git
```bash
# Install Git
sudo apt-get install git  # Ubuntu/Debian
# or
brew install git  # macOS

# Verify installation
git --version
```

---

## 📥 Clone the Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-demo

# Verify project structure
ls -la
```

Expected structure:
```
mlops-demo/
├── api/                 # FastAPI application
├── data/               # Data and datasets
├── models/             # ML model implementations
├── monitoring/         # Monitoring and drift detection
├── tests/              # Test suites
├── docs/               # Documentation
├── docker-compose.yml  # Container orchestration
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
└── README.md          # Project overview
```

---

## 🏗 Environment Setup

### Option 1: Docker Setup (Recommended)

#### 1. Build and Start Services
```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps
```

#### 2. Verify Services
```bash
# Check API health
curl http://localhost:8000/health

# Check MLflow
curl http://localhost:5000

# Check Prometheus
curl http://localhost:9090

# Check Grafana
curl http://localhost:3000
```

#### 3. Access Service UIs
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Grafana Dashboards**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jupyter Notebooks**: http://localhost:8888

### Option 2: Local Development Setup

#### 1. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv mlops-env

# Activate environment
source mlops-env/bin/activate  # Linux/macOS
# or
mlops-env\\Scripts\\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

#### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(mlflow|fastapi|prometheus)"
```

#### 3. Start Services Manually
```bash
# Terminal 1: Start MLflow
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Start monitoring (optional)
python monitoring/drift_detection.py
```

---

## 🎯 Initial Setup and Validation

### 1. Create Sample Datasets
```bash
# Using Docker
docker-compose exec api python data/create_datasets.py

# Using local setup
python data/create_datasets.py
```

### 2. Train Initial Models
```bash
# Using Docker
docker-compose exec api python scripts/train_all_models.py

# Using local setup
python scripts/train_all_models.py
```

### 3. Validate API Endpoints
```bash
# Test iris prediction
curl -X POST "http://localhost:8000/api/v1/iris/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'

# Test health endpoint
curl http://localhost:8000/health

# Test metrics endpoint
curl http://localhost:8000/metrics
```

### 4. Run Tests
```bash
# Using Docker
docker-compose exec api pytest tests/ -v

# Using local setup
pytest tests/ -v
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check what's using the ports
sudo lsof -i :8000
sudo lsof -i :5000
sudo lsof -i :3000

# Stop conflicting services
sudo kill -9 <PID>
```

#### 2. Docker Issues
```bash
# Clean up Docker resources
docker-compose down
docker system prune -f

# Rebuild with no cache
docker-compose build --no-cache
docker-compose up -d
```

#### 3. Permission Issues
```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER .
```

#### 4. Memory Issues
```bash
# Check system resources
docker stats

# Increase Docker memory limits
# Docker Desktop: Settings > Resources > Memory
```

### Verification Commands

```bash
# Check all services are running
docker-compose ps

# Check service logs
docker-compose logs api
docker-compose logs mlflow
docker-compose logs grafana

# Check API responses
curl -f http://localhost:8000/health || echo "API not ready"
curl -f http://localhost:5000 || echo "MLflow not ready"
curl -f http://localhost:3000 || echo "Grafana not ready"
```

---

## 📱 Development Tools Setup

### 1. IDE Configuration

#### VS Code Extensions
- Python
- Docker
- Jupyter
- REST Client
- YAML

#### PyCharm Setup
- Enable Docker integration
- Configure Python interpreter
- Set up run configurations

### 2. Git Configuration
```bash
# Set up Git hooks
pre-commit install

# Configure Git
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. Environment Variables
```bash
# Create .env file (optional)
cat > .env << EOF
PYTHONPATH=/app
MLFLOW_TRACKING_URI=http://localhost:5000
DEBUG=true
EOF
```

---

## 🚀 Next Steps

Once setup is complete:

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Check MLflow**: Visit http://localhost:5000
3. **View Dashboards**: Visit http://localhost:3000
4. **Run Demos**: Follow the demo scripts
5. **Read Documentation**: Explore the docs/ directory

### Recommended Learning Path

1. **API Exploration** (30 mins)
   - Test all model endpoints
   - Understand request/response formats
   - Try batch predictions

2. **MLflow Familiarization** (45 mins)
   - Browse experiments
   - Compare model runs
   - Understand model registry

3. **Monitoring Overview** (30 mins)
   - View Grafana dashboards
   - Understand metrics
   - Set up alerts

4. **Development Workflow** (60 mins)
   - Make code changes
   - Run tests
   - Deploy updates

---

## 📞 Getting Help

### Resources
- **Documentation**: Check docs/ directory
- **Issues**: Create GitHub issues
- **Logs**: Check service logs for errors
- **Community**: MLOps community forums

### Support Checklist
Before asking for help:
- [ ] Checked service logs
- [ ] Verified system requirements
- [ ] Ran troubleshooting commands
- [ ] Searched existing issues
- [ ] Prepared system information

---

## ✅ Setup Verification Checklist

- [ ] Docker and Docker Compose installed
- [ ] Repository cloned successfully
- [ ] All services start without errors
- [ ] API endpoints respond correctly
- [ ] MLflow UI accessible
- [ ] Grafana dashboards load
- [ ] Sample datasets created
- [ ] Models trained successfully
- [ ] Tests pass
- [ ] Development tools configured

**Congratulations! 🎉 Your MLOps demo environment is ready!**