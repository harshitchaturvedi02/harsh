"""
Tests for the ticket routing API.
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from database.database import init_db
from models.ticket_classifier import classifier

client = TestClient(app)


@pytest.fixture(scope="session")
def setup_database():
    """Setup test database."""
    init_db()


@pytest.fixture
def sample_ticket_data():
    """Sample ticket data for testing."""
    return {
        "title": "Test Ticket",
        "description": "This is a test ticket for testing purposes",
        "user_email": "test@example.com",
        "priority": "medium",
        "category": "test",
        "tags": ["test", "api"]
    }


@pytest.fixture
def sample_feedback_data():
    """Sample feedback data for testing."""
    return {
        "ticket_id": 1,
        "was_correct": True,
        "user_satisfaction": 5,
        "resolution_time_hours": 2.5,
        "feedback_notes": "Great routing!"
    }


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "database_connected" in data
        assert "model_loaded" in data


class TestTicketPrediction:
    """Test ticket prediction endpoint."""
    
    @patch('models.ticket_classifier.classifier.predict')
    def test_predict_ticket_success(self, mock_predict, sample_ticket_data):
        """Test successful ticket prediction."""
        # Mock prediction response
        mock_predict.return_value = {
            'predicted_team': 'technical_support',
            'confidence': 0.85,
            'explanation': {'top_features': []},
            'feature_importance': {'feature1': 0.5}
        }
        
        # Mock classifier models
        classifier.models = {'test_model': MagicMock()}
        
        response = client.post("/predict", json=sample_ticket_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ticket_id" in data
        assert data["predicted_team"] == "technical_support"
        assert data["confidence"] == 0.85
        assert "explanation" in data
        assert "feature_importance" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
        assert "created_at" in data
    
    def test_predict_ticket_no_model(self, sample_ticket_data):
        """Test prediction when model is not loaded."""
        # Clear models
        classifier.models = {}
        
        response = client.post("/predict", json=sample_ticket_data)
        
        assert response.status_code == 503
        data = response.json()
        assert "Model not loaded" in data["detail"]
    
    def test_predict_ticket_invalid_data(self):
        """Test prediction with invalid data."""
        invalid_data = {
            "title": "",  # Empty title
            "description": "Test description"
        }
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422  # Validation error


class TestFeedback:
    """Test feedback endpoint."""
    
    def test_submit_feedback_success(self, sample_feedback_data):
        """Test successful feedback submission."""
        response = client.post("/feedback", json=sample_feedback_data)
        
        # Should return 404 if ticket doesn't exist, or 200 if it does
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "feedback_id" in data
            assert data["ticket_id"] == sample_feedback_data["ticket_id"]
            assert data["was_correct"] == sample_feedback_data["was_correct"]
            assert "model_version" in data
            assert "created_at" in data
            assert "message" in data
    
    def test_submit_feedback_invalid_ticket(self):
        """Test feedback submission with invalid ticket ID."""
        invalid_feedback = {
            "ticket_id": 99999,  # Non-existent ticket
            "was_correct": True
        }
        
        response = client.post("/feedback", json=invalid_feedback)
        
        assert response.status_code == 404
        data = response.json()
        assert "Ticket not found" in data["detail"]
    
    def test_submit_feedback_invalid_data(self):
        """Test feedback submission with invalid data."""
        invalid_data = {
            "ticket_id": 1,
            # Missing was_correct field
        }
        
        response = client.post("/feedback", json=invalid_data)
        
        assert response.status_code == 422  # Validation error


class TestModelStatus:
    """Test model status endpoint."""
    
    def test_get_model_status(self):
        """Test getting model status."""
        response = client.get("/models/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_version" in data
        assert "is_loaded" in data
        assert "feature_count" in data
        assert "teams" in data
        assert isinstance(data["teams"], list)


class TestAnalytics:
    """Test analytics endpoint."""
    
    def test_get_analytics(self):
        """Test getting analytics."""
        response = client.get("/analytics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_tickets" in data
        assert "routing_accuracy" in data
        assert "average_confidence" in data
        assert "team_distribution" in data
        assert "feedback_stats" in data
        assert "recent_predictions" in data
        
        # Check data types
        assert isinstance(data["total_tickets"], int)
        assert isinstance(data["routing_accuracy"], float)
        assert isinstance(data["average_confidence"], float)
        assert isinstance(data["team_distribution"], dict)
        assert isinstance(data["feedback_stats"], dict)
        assert isinstance(data["recent_predictions"], list)


class TestTeams:
    """Test teams endpoint."""
    
    def test_get_teams(self):
        """Test getting teams."""
        response = client.get("/teams")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        
        if data:  # If teams exist
            team = data[0]
            assert "id" in team
            assert "name" in team
            assert "description" in team
            assert "keywords" in team
            assert "member_count" in team


class TestTickets:
    """Test tickets endpoint."""
    
    def test_get_tickets(self):
        """Test getting tickets."""
        response = client.get("/tickets")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        
        if data:  # If tickets exist
            ticket = data[0]
            assert "id" in ticket
            assert "title" in ticket
            assert "description" in ticket
            assert "predicted_team" in ticket
            assert "confidence" in ticket
            assert "priority" in ticket
            assert "status" in ticket
            assert "created_at" in ticket
    
    def test_get_tickets_with_filters(self):
        """Test getting tickets with filters."""
        # Test with limit
        response = client.get("/tickets?limit=5")
        assert response.status_code == 200
        
        # Test with offset
        response = client.get("/tickets?offset=10")
        assert response.status_code == 200
        
        # Test with team filter
        response = client.get("/tickets?team=technical_support")
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint."""
        response = client.get("/invalid_endpoint")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed."""
        response = client.post("/health")  # Health endpoint only accepts GET
        
        assert response.status_code == 405


class TestIntegration:
    """Integration tests."""
    
    @patch('models.ticket_classifier.classifier.predict')
    def test_full_workflow(self, mock_predict):
        """Test full workflow: predict -> feedback."""
        # Mock prediction
        mock_predict.return_value = {
            'predicted_team': 'billing_support',
            'confidence': 0.9,
            'explanation': {'top_features': []},
            'feature_importance': {'feature1': 0.5}
        }
        
        classifier.models = {'test_model': MagicMock()}
        
        # 1. Create ticket
        ticket_data = {
            "title": "Billing Issue",
            "description": "I was charged twice for the same service",
            "user_email": "user@example.com",
            "priority": "high"
        }
        
        response = client.post("/predict", json=ticket_data)
        assert response.status_code == 200
        
        ticket_response = response.json()
        ticket_id = ticket_response["ticket_id"]
        
        # 2. Submit feedback
        feedback_data = {
            "ticket_id": ticket_id,
            "was_correct": True,
            "user_satisfaction": 4,
            "resolution_time_hours": 1.5
        }
        
        response = client.post("/feedback", json=feedback_data)
        assert response.status_code == 200
        
        # 3. Check analytics
        response = client.get("/analytics")
        assert response.status_code == 200
        
        analytics = response.json()
        assert analytics["total_tickets"] >= 1
        assert analytics["routing_accuracy"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])