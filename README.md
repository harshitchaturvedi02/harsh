# Intelligent Ticket Routing System

A machine learning-powered ticket routing system that automatically assigns incoming support tickets to the most appropriate person or team using natural language processing, classification models, and continuous learning from user feedback.

## 🚀 Features

### Core Functionality
- **Intelligent Routing**: ML-powered automatic ticket assignment based on content analysis
- **Multi-Model Ensemble**: Combines Random Forest, Gradient Boosting, Logistic Regression, SVM, and Neural Networks
- **Natural Language Processing**: Advanced text analysis using spaCy, TF-IDF, topic modeling, and BERT embeddings
- **Workload Balancing**: Considers team member capacity and current workload
- **Priority Handling**: Automatic priority detection and appropriate routing

### Advanced Features
- **Explainable AI**: SHAP and LIME explanations for routing decisions
- **Continuous Learning**: Feedback loop for model improvement
- **Performance Tracking**: Comprehensive metrics and drift detection
- **Real-time API**: FastAPI-based REST API with async support
- **Scalable Architecture**: Docker-based deployment with monitoring

### Analytics & Monitoring
- **Performance Metrics**: Accuracy, precision, recall, customer satisfaction
- **Business Intelligence**: Resolution times, workload distribution, trends
- **Model Monitoring**: Drift detection and retraining recommendations
- **Visualization**: Grafana dashboards for system monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   Mobile App    │    │  Third-party    │
│                 │    │                 │    │  Integration    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Nginx Proxy         │
                    │   (Load Balancer)        │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     FastAPI Server       │
                    │  (Ticket Routing API)    │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────┴─────────┐  ┌─────────┴─────────┐  ┌─────────┴─────────┐
│   ML Pipeline     │  │   Database        │  │   Cache Layer     │
│                   │  │                   │  │                   │
│ • Text Processing │  │ • PostgreSQL      │  │ • Redis           │
│ • Feature Extract │  │ • User Data       │  │ • Session Store   │
│ • Model Ensemble  │  │ • Tickets         │  │ • Model Cache     │
│ • Explainability  │  │ • Feedback        │  │                   │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- PostgreSQL (if not using Docker)
- Redis (if not using Docker)

### Quick Start with Docker

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ticket-routing-system.git
cd ticket-routing-system
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the services**
```bash
docker-compose up -d
```

4. **Initialize the database**
```bash
docker-compose exec api python -c "
from ticket_routing_system.api.database import init_db
init_db()
"
```

5. **Generate sample data (optional)**
```bash
docker-compose exec api python -m ticket_routing_system.utils.data_generator
```

The system will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin123)
- MLflow: http://localhost:5000

### Manual Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Set up database**
```bash
# PostgreSQL
createdb ticket_routing_db

# Redis
redis-server
```

3. **Configure environment**
```bash
export DATABASE_URL="postgresql://user:password@localhost/ticket_routing_db"
export REDIS_URL="redis://localhost:6379"
```

4. **Run the application**
```bash
uvicorn ticket_routing_system.api.main:app --reload
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# ML Settings
MODEL_PATH=models/ticket_classifier_latest.joblib
SPACY_MODEL=en_core_web_sm

# API Settings
SECRET_KEY=your-secret-key
API_WORKERS=4
RATE_LIMIT_PER_MINUTE=60

# Features
ENABLE_EXPLAINABILITY=true
ENABLE_FEEDBACK_LOOP=true
AUTO_RETRAIN=true
```

### Model Configuration

Configure ML models in your code:

```python
from ticket_routing_system.models.ml_models import TicketClassifier

classifier = TicketClassifier(model_type="ensemble")
# Available types: "ensemble", "random_forest", "gradient_boosting"
```

## 📚 API Usage

### Create a Ticket

```bash
curl -X POST "http://localhost:8000/tickets" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Database connection timeout",
       "description": "Our application cannot connect to the database server. Error: Connection timeout after 30 seconds.",
       "priority": "high",
       "creator_id": 1001
     }'
```

### Get Routing Recommendation

```bash
curl -X POST "http://localhost:8000/route" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "API returning 500 error",
       "description": "The REST API is returning internal server error for all requests",
       "priority": "critical"
     }'
```

### Submit Feedback

```bash
curl -X POST "http://localhost:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "ticket_id": 123,
       "rating": 5,
       "was_correctly_routed": true,
       "resolution_quality": 4,
       "response_time_satisfaction": 5
     }'
```

### Get Explanation

```bash
curl "http://localhost:8000/explain/123?explanation_type=comprehensive"
```

## 🧠 Machine Learning Pipeline

### Text Processing Pipeline

1. **Preprocessing**: Clean text, remove URLs, normalize whitespace
2. **Feature Extraction**:
   - Basic features: length, word count, punctuation
   - NLP features: entities, keywords, sentiment
   - Department signals: keyword matching
   - Urgency indicators: priority keywords, caps, exclamation marks
   - Technical complexity: technical term density

3. **Advanced Features**:
   - TF-IDF vectors (5000 features)
   - Topic modeling (20 topics)
   - BERT embeddings (768 dimensions)

### Model Ensemble

The system uses multiple models for robust predictions:

- **Random Forest**: Feature importance, handles mixed data types
- **Gradient Boosting**: Sequential learning, handles non-linear patterns
- **Logistic Regression**: Linear relationships, fast inference
- **SVM**: Complex decision boundaries with RBF kernel
- **Neural Network**: Deep learning for high-dimensional data

### Continuous Learning

1. **Feedback Collection**: User ratings and routing correctness
2. **Performance Monitoring**: Track accuracy, satisfaction, resolution times
3. **Drift Detection**: Statistical tests for model degradation
4. **Automatic Retraining**: Triggered by performance thresholds

## 📊 Monitoring & Analytics

### Key Metrics

- **ML Performance**: Accuracy, Precision, Recall, F1-Score
- **Business Metrics**: Customer satisfaction, resolution times
- **Operational Metrics**: Workload distribution, response times
- **System Metrics**: API latency, error rates, throughput

### Dashboards

Access monitoring dashboards:
- **Grafana**: System and business metrics
- **MLflow**: Model tracking and experiments
- **API Docs**: Interactive API documentation

### Alerts

Configure alerts for:
- Model accuracy drops below threshold
- High error rates or latency
- System resource utilization
- Failed model training jobs

## 🔍 Explainable AI

### SHAP Explanations
```python
from ticket_routing_system.explainability.explainer import RoutingExplainer

explainer = RoutingExplainer(classifier, text_processor)
explanation = explainer.explain_prediction(ticket_data)
print(explanation['shap_explanation'])
```

### LIME Explanations
Local interpretable model-agnostic explanations for individual predictions.

### Rule-based Explanations
Human-readable rules based on domain knowledge:
- Department keyword matching
- Urgency detection
- Technical complexity assessment
- Priority-based routing

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load tests
pytest tests/load/

# All tests with coverage
pytest --cov=ticket_routing_system tests/
```

### Generate Test Data
```bash
python -m ticket_routing_system.utils.data_generator
```

### Model Evaluation
```bash
python -m ticket_routing_system.evaluation.evaluate_model
```

## 🚀 Deployment

### Production Deployment

1. **Configure production settings**
```bash
cp .env.example .env.production
# Edit production settings
```

2. **Build and deploy**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Set up SSL certificates**
```bash
# Using Let's Encrypt
certbot --nginx -d your-domain.com
```

4. **Configure monitoring**
```bash
# Set up log aggregation
# Configure alerting rules
# Set up backup schedules
```

### Scaling

For high-traffic deployments:

1. **Horizontal scaling**: Multiple API instances behind load balancer
2. **Database optimization**: Read replicas, connection pooling
3. **Caching**: Redis cluster for distributed caching
4. **Model serving**: Separate model inference service

## 🔧 Development

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run in development mode
uvicorn ticket_routing_system.api.main:app --reload
```

### Code Style

The project uses:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black ticket_routing_system/
isort ticket_routing_system/

# Check style
flake8 ticket_routing_system/
mypy ticket_routing_system/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure code quality checks pass
5. Submit a pull request

## 📈 Performance

### Benchmarks

Typical performance on modern hardware:

- **Inference latency**: <100ms per ticket
- **Training time**: 5-10 minutes on 10K tickets
- **Throughput**: 1000+ requests/second
- **Accuracy**: 85-90% routing accuracy

### Optimization Tips

1. **Model optimization**: Feature selection, hyperparameter tuning
2. **Caching**: Cache model predictions and features
3. **Batch processing**: Process multiple tickets together
4. **Hardware**: Use GPU for neural network training

## 🛠️ Troubleshooting

### Common Issues

**Model not loading**
```bash
# Check model file exists
ls -la models/

# Verify permissions
chmod 644 models/*.joblib
```

**Database connection errors**
```bash
# Test connection
docker-compose exec postgres psql -U ticket_user -d ticket_routing_db -c "SELECT 1;"
```

**High memory usage**
```bash
# Monitor memory
docker stats

# Reduce model size
# Use feature selection
# Implement model quantization
```

### Logs

Check application logs:
```bash
# Docker logs
docker-compose logs -f api

# File logs
tail -f logs/application.log
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ticket-routing-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ticket-routing-system/discussions)

## 🙏 Acknowledgments

- **spaCy**: Natural language processing
- **scikit-learn**: Machine learning algorithms
- **FastAPI**: Modern web framework
- **SHAP**: Model explainability
- **PostgreSQL**: Reliable database
- **Docker**: Containerization platform

---

Built with ❤️ for better customer support experiences.
