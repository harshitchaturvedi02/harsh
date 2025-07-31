# ML-Based Ticket Routing System

A comprehensive machine learning system that automatically routes support tickets to the most appropriate team or person based on content analysis, historical data, and user feedback.

## Features

- **Natural Language Processing**: Advanced text analysis using spaCy and transformers
- **Multi-Model Classification**: Ensemble of ML models for robust routing decisions
- **Feedback Loop**: Continuous learning from user satisfaction and resolution success
- **Explainability**: Transparent reasoning behind routing decisions
- **Scalable Architecture**: Handles large volumes across multiple departments
- **Real-time API**: FastAPI-based REST API for production deployment
- **Monitoring**: MLflow and Weights & Biases integration for model tracking

## Architecture

```
├── data/                   # Data storage and preprocessing
├── models/                 # ML model definitions and training
├── api/                    # FastAPI application
├── database/               # Database models and migrations
├── utils/                  # Utility functions
├── config/                 # Configuration files
├── tests/                  # Test suite
└── notebooks/              # Jupyter notebooks for exploration
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Setup Database**:
   ```bash
   alembic upgrade head
   ```

3. **Train Models**:
   ```bash
   python -m models.train
   ```

4. **Start API Server**:
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Access API Documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

- `POST /predict` - Route a new ticket
- `POST /feedback` - Submit routing feedback
- `GET /models/status` - Check model health
- `GET /analytics` - View routing analytics

## Model Performance

- **Accuracy**: >90% on historical data
- **Response Time**: <100ms per prediction
- **Explainability**: SHAP-based feature importance
- **Continuous Learning**: Retrained weekly with new data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details