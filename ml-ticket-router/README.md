# ML-Based Ticket Routing System

A sophisticated machine learning system for automatically routing support tickets to the most appropriate person or team based on ticket content, historical data, and user feedback.

## Features

- **Natural Language Processing**: Advanced text analysis using spaCy and transformers
- **Multiple Classification Models**: Support for various ML algorithms (Random Forest, XGBoost, Neural Networks)
- **Continuous Learning**: Feedback loop integration for model improvement
- **Scalable Architecture**: Designed to handle high volumes of tickets
- **Explainable AI**: Provides reasoning behind routing decisions using SHAP/LIME
- **REST API**: FastAPI-based service for easy integration
- **Real-time Processing**: Asynchronous ticket processing with Redis queue
- **Comprehensive Monitoring**: MLflow integration for model tracking

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Ticket Input  │────▶│  NLP Pipeline   │────▶│  ML Classifier  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Feedback Loop   │◀────│ Route Decision  │◀────│  Explainability │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-ticket-router
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Quick Start

1. **Prepare your data**:
```bash
python scripts/prepare_data.py --input data/raw/tickets.csv
```

2. **Train the model**:
```bash
python scripts/train_model.py --config config/model_config.yaml
```

3. **Start the API server**:
```bash
uvicorn src.api.main:app --reload --port 8000
```

4. **Test the system**:
```bash
curl -X POST "http://localhost:8000/api/v1/route-ticket" \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "12345", "description": "Cannot login to system", "priority": "high"}'
```

## Project Structure

```
ml-ticket-router/
├── src/
│   ├── models/          # ML model implementations
│   ├── preprocessing/   # Data preprocessing pipelines
│   ├── api/            # FastAPI application
│   └── utils/          # Utility functions
├── data/
│   ├── raw/            # Raw ticket data
│   ├── processed/      # Preprocessed data
│   └── models/         # Trained model artifacts
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for analysis
├── config/             # Configuration files
└── scripts/            # Training and deployment scripts
```

## Configuration

The system can be configured via `config/model_config.yaml`:

```yaml
model:
  type: "ensemble"  # Options: random_forest, xgboost, neural_network, ensemble
  parameters:
    n_estimators: 100
    max_depth: 10

preprocessing:
  text_features:
    - tfidf
    - word_embeddings
    - sentiment
  
training:
  test_size: 0.2
  cross_validation: 5
```

## API Endpoints

- `POST /api/v1/route-ticket`: Route a single ticket
- `POST /api/v1/route-batch`: Route multiple tickets
- `POST /api/v1/feedback`: Submit routing feedback
- `GET /api/v1/model/performance`: Get model performance metrics
- `GET /api/v1/model/explain/{ticket_id}`: Get routing explanation

## Performance Metrics

The system tracks:
- Accuracy, Precision, Recall, F1-Score
- Routing time (p50, p95, p99)
- User satisfaction scores
- Resolution time improvements

## Deployment

### Docker Deployment
```bash
docker build -t ml-ticket-router .
docker run -p 8000:8000 ml-ticket-router
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/deployment.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License