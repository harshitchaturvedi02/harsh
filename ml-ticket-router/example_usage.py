#!/usr/bin/env python3
"""
Example usage of the ML Ticket Router system.
"""

import requests
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "demo-key-123"  # Use your actual API key

def route_single_ticket():
    """Example: Route a single ticket."""
    print("=== Single Ticket Routing ===")
    
    ticket = {
        "ticket_id": "EXAMPLE-001",
        "description": "I can't login to my account. It keeps saying invalid password.",
        "priority": "high"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/route-ticket",
        headers={"X-API-Key": API_KEY},
        json=ticket
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Ticket ID: {result['ticket_id']}")
        print(f"Assigned to: {result['assigned_to']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Processing time: {result['processing_time_ms']:.1f}ms")
        print(f"Alternatives: {json.dumps(result['alternative_assignments'], indent=2)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def route_batch_tickets():
    """Example: Route multiple tickets in batch."""
    print("\n=== Batch Ticket Routing ===")
    
    batch_request = {
        "tickets": [
            {
                "ticket_id": "BATCH-001",
                "description": "Need invoice for last month",
                "priority": "medium"
            },
            {
                "ticket_id": "BATCH-002",
                "description": "Security alert: suspicious login attempt",
                "priority": "critical"
            },
            {
                "ticket_id": "BATCH-003",
                "description": "Feature request: add dark mode",
                "priority": "low"
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/route-batch",
        headers={"X-API-Key": API_KEY},
        json=batch_request
    )
    
    if response.status_code == 200:
        results = response.json()
        for result in results:
            print(f"\nTicket: {result['ticket_id']} -> {result['assigned_to']} ({result['confidence']:.2%})")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def submit_feedback():
    """Example: Submit feedback for a routing decision."""
    print("\n=== Submit Feedback ===")
    
    feedback = {
        "ticket_id": "EXAMPLE-001",
        "predicted_class": "Technical Support",
        "feedback_type": "positive",
        "satisfaction_score": 5,
        "resolution_time": 2.5  # hours
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/feedback",
        headers={"X-API-Key": API_KEY},
        json=feedback
    )
    
    if response.status_code == 200:
        print("Feedback submitted successfully")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_model_performance():
    """Example: Get current model performance metrics."""
    print("\n=== Model Performance ===")
    
    response = requests.get(
        f"{API_URL}/api/v1/model/performance",
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        metrics = response.json()
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
        print(f"Total Predictions: {metrics['total_predictions']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_routing_explanation():
    """Example: Get explanation for a routing decision."""
    print("\n=== Routing Explanation ===")
    
    # First, route a ticket with explanation
    ticket = {
        "ticket_id": "EXPLAIN-001",
        "description": "URGENT: Account hacked! Someone changed my password!",
        "priority": "critical"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/route-ticket?include_explanation=true",
        headers={"X-API-Key": API_KEY},
        json=ticket
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Ticket routed to: {result['assigned_to']}")
        print(f"Confidence: {result['confidence']:.2%}")
        if result.get('explanation'):
            print(f"Explanation: {result['explanation']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def check_health():
    """Example: Check system health."""
    print("\n=== System Health Check ===")
    
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status']}")
        print("Components:")
        for component, status in health['components'].items():
            print(f"  {component}: {'✓' if status else '✗'}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    """Run all examples."""
    print("ML Ticket Router - Example Usage")
    print("================================\n")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("❌ API is not responding. Please start the server:")
            print("   uvicorn src.api.main:app --reload --port 8000")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the server:")
        print("   uvicorn src.api.main:app --reload --port 8000")
        return
    
    print("✓ API is running\n")
    
    # Run examples
    try:
        check_health()
        route_single_ticket()
        route_batch_tickets()
        submit_feedback()
        get_model_performance()
        get_routing_explanation()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have trained a model first:")
        print("  python scripts/train_model.py --synthetic")

if __name__ == "__main__":
    main()