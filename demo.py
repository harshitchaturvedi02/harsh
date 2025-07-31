#!/usr/bin/env python3
"""
Demonstration script for the Intelligent Ticket Routing System

This script showcases the key features of the system:
1. Data generation
2. Model training
3. Ticket routing
4. Explainability
5. Feedback processing
6. Performance evaluation
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from ticket_routing_system.utils.data_generator import TicketDataGenerator
from ticket_routing_system.models.ml_models import TicketClassifier
from ticket_routing_system.nlp.text_processor import TextProcessor, FeatureExtractor
from ticket_routing_system.feedback.feedback_loop import FeedbackLoop
from ticket_routing_system.explainability.explainer import RoutingExplainer
from ticket_routing_system.evaluation.metrics import RoutingMetrics


class TicketRoutingDemo:
    """Demonstration class for the ticket routing system"""
    
    def __init__(self):
        self.data_generator = TicketDataGenerator()
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor(self.text_processor)
        self.classifier = TicketClassifier()
        self.feedback_loop = None
        self.explainer = None
        self.metrics = RoutingMetrics()
        
        # Demo data
        self.users = []
        self.tickets = []
        self.feedback = []
        
        print("🎯 Intelligent Ticket Routing System Demo")
        print("=" * 50)
    
    def step_1_generate_data(self):
        """Step 1: Generate synthetic data"""
        print("\n📊 Step 1: Generating Synthetic Data")
        print("-" * 40)
        
        # Generate sample dataset
        dataset = self.data_generator.generate_complete_dataset(
            num_users=15,
            num_tickets=500,
            days_back=60
        )
        
        self.users = dataset['users']
        self.tickets = dataset['tickets']
        self.feedback = dataset['feedback']
        
        # Print statistics
        print(f"✅ Generated {len(self.users)} users across {len(set(u['department'] for u in self.users))} departments")
        print(f"✅ Generated {len(self.tickets)} tickets with {len(self.feedback)} feedback entries")
        
        # Show department distribution
        dept_counts = {}
        for ticket in self.tickets:
            dept = ticket['department']
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
        print("\n📈 Ticket Distribution by Department:")
        for dept, count in sorted(dept_counts.items()):
            print(f"  • {dept.replace('_', ' ').title()}: {count} tickets")
    
    def step_2_train_model(self):
        """Step 2: Train the ML model"""
        print("\n🤖 Step 2: Training Machine Learning Models")
        print("-" * 40)
        
        # Prepare training data
        X = []
        y = []
        
        print("🔄 Extracting features from tickets...")
        for ticket in self.tickets:
            if ticket['assignee_id'] is not None:
                # Prepare ticket data for feature extraction
                ticket_data = {
                    'title': ticket['title'],
                    'description': ticket['description'],
                    'priority': ticket['priority']
                }
                
                # Extract features
                features = self.feature_extractor.extract_ticket_features(ticket_data)
                X.append(features)
                y.append(ticket['assignee_id'])
        
        print(f"✅ Extracted features from {len(X)} tickets")
        print(f"📏 Feature vector size: {len(X[0]) if X else 0}")
        
        # Fit text processing components
        texts = [f"{t['title']} {t['description']}" for t in self.tickets]
        print("🔄 Fitting TF-IDF vectorizer...")
        self.text_processor.fit_tfidf(texts)
        
        print("🔄 Fitting topic model...")
        self.text_processor.fit_topic_model(texts)
        
        # Train the model
        print("🔄 Training ensemble of ML models...")
        import numpy as np
        X_array = np.array(X)
        y_array = np.array(y)
        
        training_results = self.classifier.train(X_array, y_array)
        
        print("✅ Model training completed!")
        print("\n📊 Training Results:")
        for model_name, results in training_results.items():
            if 'error' not in results:
                print(f"  • {model_name.replace('_', ' ').title()}:")
                print(f"    - Test Accuracy: {results.get('test_accuracy', 0):.3f}")
                print(f"    - CV Score: {results.get('cv_mean', 0):.3f} ± {results.get('cv_std', 0):.3f}")
        
        # Initialize feedback loop and explainer
        self.feedback_loop = FeedbackLoop(self.classifier, self.text_processor)
        self.explainer = RoutingExplainer(self.classifier, self.text_processor)
        
        # Initialize explainer with training data
        sample_size = min(100, len(X_array))
        sample_indices = np.random.choice(len(X_array), sample_size, replace=False)
        X_sample = X_array[sample_indices]
        
        self.explainer.initialize_explainers(X_sample, self.feature_extractor.get_feature_names())
        
        print("✅ Explainer initialized")
    
    def step_3_demonstrate_routing(self):
        """Step 3: Demonstrate ticket routing"""
        print("\n🎯 Step 3: Demonstrating Intelligent Ticket Routing")
        print("-" * 40)
        
        # Create sample tickets for demonstration
        demo_tickets = [
            {
                'title': 'Database connection timeout error',
                'description': 'Our production database is experiencing connection timeouts. Multiple users affected. Error code: 500. This is critical for our operations.',
                'priority': 'critical'
            },
            {
                'title': 'Billing inquiry about recent charges',
                'description': 'I was charged twice for my subscription this month. Can you please help me get a refund for the duplicate charge?',
                'priority': 'medium'
            },
            {
                'title': 'Feature request for dark mode',
                'description': 'It would be great to have a dark mode option in the application. Many users have requested this feature.',
                'priority': 'low'
            },
            {
                'title': 'Security vulnerability report',
                'description': 'I found a potential XSS vulnerability in the user profile page. This could allow malicious scripts to be executed.',
                'priority': 'high'
            }
        ]
        
        # Create user lookup for workload balancing
        user_data = {
            i: {
                'current_workload': user['current_workload'],
                'workload_capacity': user['workload_capacity'],
                'skills': user['skills'],
                'department': user['department']
            }
            for i, user in enumerate(self.users)
        }
        
        print("🔄 Routing demonstration tickets...\n")
        
        for i, ticket_data in enumerate(demo_tickets, 1):
            print(f"🎫 Ticket {i}: {ticket_data['title']}")
            print(f"   Priority: {ticket_data['priority'].upper()}")
            print(f"   Description: {ticket_data['description'][:100]}...")
            
            # Extract features
            features = self.feature_extractor.extract_ticket_features(ticket_data)
            features = features.reshape(1, -1)
            
            # Get routing prediction
            predictions, probabilities = self.classifier.predict(features, user_data)
            predicted_assignee_idx = predictions[0]
            confidence = probabilities[0][predicted_assignee_idx]
            
            # Get assignee info
            assignee = self.users[predicted_assignee_idx]
            
            print(f"   ➡️  Routed to: {assignee['full_name']} ({assignee['department'].replace('_', ' ').title()})")
            print(f"   🎯 Confidence: {confidence:.1%}")
            
            # Get top 3 alternatives
            top_3_indices = probabilities[0].argsort()[-3:][::-1]
            print("   📋 Alternative suggestions:")
            for idx in top_3_indices:
                if idx != predicted_assignee_idx:
                    alt_assignee = self.users[idx]
                    alt_confidence = probabilities[0][idx]
                    print(f"      • {alt_assignee['full_name']} ({alt_assignee['department'].replace('_', ' ').title()}): {alt_confidence:.1%}")
            
            print()
    
    def step_4_demonstrate_explainability(self):
        """Step 4: Demonstrate explainable AI features"""
        print("\n🔍 Step 4: Demonstrating Explainable AI")
        print("-" * 40)
        
        # Use the first demo ticket for explanation
        ticket_data = {
            'title': 'Database connection timeout error',
            'description': 'Our production database is experiencing connection timeouts. Multiple users affected. Error code: 500. This is critical for our operations.',
            'priority': 'critical'
        }
        
        print(f"🎫 Explaining routing for: {ticket_data['title']}")
        print()
        
        # Get comprehensive explanation
        explanation = self.explainer.explain_prediction(ticket_data, "comprehensive")
        
        print(f"🎯 Predicted Assignee: User {explanation['predicted_assignee']}")
        print(f"📊 Confidence: {explanation['confidence']:.1%}")
        print()
        
        # Show rule-based explanation (most interpretable)
        if 'rule_based_explanation' in explanation:
            rules = explanation['rule_based_explanation'].get('rules_triggered', [])
            if rules:
                print("📋 Key Decision Factors:")
                for rule in rules:
                    print(f"  • {rule['explanation']} (confidence: {rule['confidence']:.1%})")
                print()
        
        # Show feature importance
        if 'feature_importance' in explanation:
            features = explanation['feature_importance'].get('feature_contributions', [])[:5]
            if features:
                print("🏆 Top Contributing Features:")
                for feature in features:
                    print(f"  • {feature['feature']}: {feature['contribution']:.3f}")
                print()
        
        # Generate human-readable report
        print("📄 Detailed Explanation Report:")
        print("-" * 30)
        report = self.explainer.generate_explanation_report(ticket_data)
        print(report)
    
    def step_5_demonstrate_feedback_loop(self):
        """Step 5: Demonstrate feedback processing"""
        print("\n🔄 Step 5: Demonstrating Feedback Loop & Continuous Learning")
        print("-" * 40)
        
        # Analyze existing feedback
        print("📊 Analyzing feedback data...")
        
        # Process feedback for analysis
        feedback_analysis = self.feedback_loop.feedback_processor.analyze_feedback(self.feedback)
        
        print("✅ Feedback Analysis Results:")
        metrics = feedback_analysis.get('metrics', {})
        
        if metrics:
            print(f"  • Total Feedback Entries: {metrics.get('total_feedback', 0)}")
            print(f"  • Average Rating: {metrics.get('avg_rating', 0):.2f}/5.0")
            print(f"  • Routing Accuracy: {metrics.get('routing_accuracy', 0):.1%}")
            print(f"  • Average Resolution Quality: {metrics.get('avg_resolution_quality', 0):.2f}/5.0")
            print(f"  • Response Time Satisfaction: {metrics.get('avg_response_satisfaction', 0):.2f}/5.0")
        
        # Show recommendations
        recommendations = feedback_analysis.get('recommendation', [])
        if recommendations:
            print("\n💡 System Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        # Show department performance
        dept_performance = metrics.get('department_performance', {})
        if dept_performance:
            print("\n🏢 Performance by Department:")
            for dept, perf in dept_performance.items():
                print(f"  • {dept.replace('_', ' ').title()}:")
                print(f"    - Tickets: {perf['count']}")
                print(f"    - Avg Rating: {perf['avg_rating']:.2f}/5.0")
                print(f"    - Routing Accuracy: {perf['routing_accuracy']:.1%}")
        
        # Demonstrate retraining decision
        print(f"\n🤖 Retraining Needed: {'Yes' if feedback_analysis.get('needs_retraining', False) else 'No'}")
        if feedback_analysis.get('needs_retraining', False):
            print("   📈 Model performance is below threshold - retraining recommended")
        else:
            print("   ✅ Model performance is satisfactory")
    
    def step_6_evaluate_performance(self):
        """Step 6: Evaluate system performance"""
        print("\n📈 Step 6: Performance Evaluation & Metrics")
        print("-" * 40)
        
        # Prepare evaluation data
        resolved_tickets = [t for t in self.tickets if t['status'] in ['resolved', 'closed']]
        
        print(f"📊 Evaluating performance on {len(resolved_tickets)} resolved tickets...")
        
        # Simulate predictions for evaluation
        y_true = []
        y_pred = []
        y_prob = []
        
        for ticket in resolved_tickets[:100]:  # Limit for demo
            if ticket['assignee_id'] is not None:
                # Get actual assignee
                y_true.append(ticket['assignee_id'])
                
                # Get model prediction
                ticket_data = {
                    'title': ticket['title'],
                    'description': ticket['description'],
                    'priority': ticket['priority']
                }
                
                features = self.feature_extractor.extract_ticket_features(ticket_data)
                features = features.reshape(1, -1)
                
                predictions, probabilities = self.classifier.predict(features)
                y_pred.append(predictions[0])
                y_prob.append(probabilities[0])
        
        if y_true and y_pred:
            import numpy as np
            
            # Evaluate ML performance
            ml_metrics = self.metrics.evaluate_model_performance(
                np.array(y_true), 
                np.array(y_pred), 
                np.array(y_prob)
            )
            
            print("🤖 Machine Learning Metrics:")
            print(f"  • Accuracy: {ml_metrics['accuracy']:.3f}")
            print(f"  • Precision: {ml_metrics['precision']:.3f}")
            print(f"  • Recall: {ml_metrics['recall']:.3f}")
            print(f"  • F1-Score: {ml_metrics['f1_score']:.3f}")
            
            if 'top_3_accuracy' in ml_metrics:
                print(f"  • Top-3 Accuracy: {ml_metrics['top_3_accuracy']:.3f}")
        
        # Evaluate business metrics
        business_metrics = self.metrics.evaluate_business_metrics([], self.feedback)
        
        if 'error' not in business_metrics:
            print("\n💼 Business Metrics:")
            print(f"  • Customer Satisfaction: {business_metrics.get('avg_customer_satisfaction', 0):.2f}/5.0")
            print(f"  • Routing Accuracy (Feedback): {business_metrics.get('routing_accuracy_feedback', 0):.1%}")
            print(f"  • Resolution Quality: {business_metrics.get('avg_resolution_quality', 0):.2f}/5.0")
        
        # Evaluate operational metrics
        operational_metrics = self.metrics.evaluate_operational_metrics(self.tickets, self.users)
        
        if 'error' not in operational_metrics:
            print("\n⚙️ Operational Metrics:")
            if 'avg_resolution_time_hours' in operational_metrics:
                print(f"  • Avg Resolution Time: {operational_metrics['avg_resolution_time_hours']:.1f} hours")
            
            if 'workload_balance_coefficient' in operational_metrics:
                balance_coeff = operational_metrics['workload_balance_coefficient']
                if balance_coeff is not None:
                    balance_status = "Good" if balance_coeff < 0.3 else "Fair" if balance_coeff < 0.6 else "Poor"
                    print(f"  • Workload Balance: {balance_status} (coefficient: {balance_coeff:.3f})")
        
        print("\n✅ Performance evaluation completed!")
    
    def step_7_show_api_examples(self):
        """Step 7: Show API usage examples"""
        print("\n🌐 Step 7: API Usage Examples")
        print("-" * 40)
        
        print("The system provides a comprehensive REST API. Here are some examples:")
        print()
        
        # Example API calls
        examples = [
            {
                'title': 'Create a Ticket',
                'method': 'POST',
                'endpoint': '/tickets',
                'payload': {
                    'title': 'Database connection timeout',
                    'description': 'Cannot connect to production database',
                    'priority': 'high',
                    'creator_id': 1001
                }
            },
            {
                'title': 'Get Routing Recommendation',
                'method': 'POST',
                'endpoint': '/route',
                'payload': {
                    'title': 'API returning 500 error',
                    'description': 'REST API internal server error',
                    'priority': 'critical'
                }
            },
            {
                'title': 'Submit Feedback',
                'method': 'POST',
                'endpoint': '/feedback',
                'payload': {
                    'ticket_id': 123,
                    'rating': 5,
                    'was_correctly_routed': True,
                    'resolution_quality': 4
                }
            },
            {
                'title': 'Get Explanation',
                'method': 'GET',
                'endpoint': '/explain/123?explanation_type=comprehensive',
                'payload': None
            }
        ]
        
        for example in examples:
            print(f"📡 {example['title']}:")
            print(f"   {example['method']} http://localhost:8000{example['endpoint']}")
            
            if example['payload']:
                print("   Payload:")
                print(f"   {json.dumps(example['payload'], indent=6)}")
            print()
        
        print("🚀 Start the API server with:")
        print("   docker-compose up -d")
        print("   or")
        print("   uvicorn ticket_routing_system.api.main:app --reload")
        print()
        print("📚 Interactive API documentation available at:")
        print("   http://localhost:8000/docs")
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            self.step_1_generate_data()
            self.step_2_train_model()
            self.step_3_demonstrate_routing()
            self.step_4_demonstrate_explainability()
            self.step_5_demonstrate_feedback_loop()
            self.step_6_evaluate_performance()
            self.step_7_show_api_examples()
            
            print("\n🎉 Demo Completed Successfully!")
            print("=" * 50)
            print("The Intelligent Ticket Routing System demonstrates:")
            print("✅ Automated ML-powered ticket routing")
            print("✅ Multi-model ensemble for robust predictions")
            print("✅ Explainable AI with SHAP, LIME, and rule-based explanations")
            print("✅ Continuous learning through feedback loops")
            print("✅ Comprehensive performance monitoring")
            print("✅ Production-ready REST API")
            print("✅ Scalable architecture with Docker deployment")
            print()
            print("🚀 Ready for production deployment!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise


def main():
    """Main demo function"""
    # Ensure we have necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the demo
    demo = TicketRoutingDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()