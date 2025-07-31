"""
Synthetic data generator for testing and demonstration
"""
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from faker import Faker
import json

fake = Faker()


class TicketDataGenerator:
    """Generate synthetic ticket data for testing"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Department templates
        self.department_templates = {
            'technical_support': {
                'keywords': ['bug', 'error', 'crash', 'not working', 'broken', 'server', 'database', 'api'],
                'titles': [
                    "Application crashes on startup",
                    "Database connection timeout",
                    "API returning 500 error",
                    "Login functionality not working",
                    "Performance issues with dashboard",
                    "Integration failing with third-party service"
                ],
                'priorities': {'low': 0.2, 'medium': 0.5, 'high': 0.25, 'critical': 0.05}
            },
            'billing': {
                'keywords': ['payment', 'invoice', 'billing', 'charge', 'refund', 'subscription', 'account'],
                'titles': [
                    "Incorrect billing amount",
                    "Payment method not working",
                    "Need refund for cancelled service",
                    "Subscription renewal issues",
                    "Invoice not received",
                    "Credit card declined"
                ],
                'priorities': {'low': 0.3, 'medium': 0.6, 'high': 0.1, 'critical': 0.0}
            },
            'sales': {
                'keywords': ['demo', 'trial', 'purchase', 'pricing', 'features', 'quote', 'proposal'],
                'titles': [
                    "Request for product demo",
                    "Pricing information needed",
                    "Trial extension request",
                    "Feature comparison with competitors",
                    "Custom quote for enterprise plan",
                    "Partnership inquiry"
                ],
                'priorities': {'low': 0.4, 'medium': 0.5, 'high': 0.1, 'critical': 0.0}
            },
            'product': {
                'keywords': ['feature request', 'enhancement', 'improvement', 'suggestion', 'feedback'],
                'titles': [
                    "Feature request: Dark mode",
                    "Improvement suggestion for user interface",
                    "Enhancement for mobile app",
                    "New integration request",
                    "Feedback on recent update",
                    "Suggestion for workflow optimization"
                ],
                'priorities': {'low': 0.6, 'medium': 0.3, 'high': 0.1, 'critical': 0.0}
            },
            'security': {
                'keywords': ['security', 'vulnerability', 'breach', 'password', 'login', 'authentication'],
                'titles': [
                    "Security vulnerability report",
                    "Password reset not working",
                    "Suspicious login activity",
                    "Two-factor authentication issues",
                    "Data breach concern",
                    "Access permissions problem"
                ],
                'priorities': {'low': 0.1, 'medium': 0.3, 'high': 0.4, 'critical': 0.2}
            }
        }
        
        # User profiles
        self.user_profiles = {
            'technical_support': {
                'skills': ['debugging', 'database', 'api', 'server_admin', 'troubleshooting'],
                'capacity_range': (8, 15),
                'departments': ['technical_support', 'product']
            },
            'billing': {
                'skills': ['accounting', 'customer_service', 'payment_processing', 'refunds'],
                'capacity_range': (10, 20),
                'departments': ['billing']
            },
            'sales': {
                'skills': ['sales', 'customer_relations', 'product_knowledge', 'negotiation'],
                'capacity_range': (5, 12),
                'departments': ['sales']
            },
            'product': {
                'skills': ['product_management', 'user_experience', 'requirements_analysis'],
                'capacity_range': (6, 10),
                'departments': ['product', 'technical_support']
            },
            'security': {
                'skills': ['security', 'compliance', 'risk_assessment', 'incident_response'],
                'capacity_range': (4, 8),
                'departments': ['security', 'technical_support']
            }
        }
    
    def generate_users(self, num_users: int = 20) -> List[Dict[str, Any]]:
        """Generate synthetic user data"""
        users = []
        
        for i in range(num_users):
            # Choose random department
            dept_type = random.choice(list(self.user_profiles.keys()))
            profile = self.user_profiles[dept_type]
            
            user = {
                'username': fake.user_name(),
                'email': fake.email(),
                'full_name': fake.name(),
                'department': dept_type,
                'skills': random.sample(profile['skills'], k=random.randint(2, len(profile['skills']))),
                'workload_capacity': random.randint(*profile['capacity_range']),
                'current_workload': 0,  # Will be updated based on tickets
                'is_active': random.choice([True, True, True, False]),  # 75% active
                'performance_metrics': {
                    'avg_resolution_time': random.uniform(4, 48),
                    'satisfaction_score': random.uniform(3.5, 5.0),
                    'tickets_resolved': random.randint(50, 500)
                }
            }
            users.append(user)
        
        return users
    
    def generate_tickets(self, num_tickets: int = 1000, 
                        users: List[Dict[str, Any]] = None,
                        days_back: int = 90) -> List[Dict[str, Any]]:
        """Generate synthetic ticket data"""
        
        if users is None:
            users = self.generate_users()
        
        tickets = []
        
        # Create user lookup
        user_lookup = {i: user for i, user in enumerate(users)}
        
        for i in range(num_tickets):
            # Choose random department
            department = random.choice(list(self.department_templates.keys()))
            template = self.department_templates[department]
            
            # Generate ticket content
            title = random.choice(template['titles'])
            
            # Add some variation to titles
            if random.random() < 0.3:
                title = f"{title} - {fake.word().capitalize()}"
            
            # Generate description
            description = self._generate_description(department, template['keywords'])
            
            # Choose priority based on department distribution
            priority = np.random.choice(
                list(template['priorities'].keys()),
                p=list(template['priorities'].values())
            )
            
            # Choose assignee (prefer users from same department)
            suitable_users = [
                (idx, user) for idx, user in user_lookup.items()
                if user['is_active'] and (
                    user['department'] == department or 
                    department in self.user_profiles.get(user['department'], {}).get('departments', [])
                )
            ]
            
            if not suitable_users:
                # Fallback to any active user
                suitable_users = [(idx, user) for idx, user in user_lookup.items() if user['is_active']]
            
            assignee_idx, assignee = random.choice(suitable_users)
            
            # Generate timestamps
            created_at = fake.date_time_between(
                start_date=f'-{days_back}d',
                end_date='now'
            )
            
            # Determine if ticket is resolved
            is_resolved = random.random() < 0.8  # 80% resolution rate
            
            if is_resolved:
                resolution_hours = self._generate_resolution_time(priority, assignee)
                resolved_at = created_at + timedelta(hours=resolution_hours)
                status = random.choice(['resolved', 'closed'])
            else:
                resolution_hours = None
                resolved_at = None
                status = random.choice(['open', 'in_progress'])
            
            ticket = {
                'id': i + 1,
                'title': title,
                'description': description,
                'status': status,
                'priority': priority,
                'department': department,
                'assignee_id': assignee_idx,
                'creator_id': random.randint(1000, 9999),  # Random customer ID
                'created_at': created_at,
                'updated_at': created_at + timedelta(hours=random.uniform(0, 24)),
                'resolved_at': resolved_at,
                'resolution_time_hours': resolution_hours,
                'tags': self._generate_tags(department, template['keywords']),
                'metadata': {
                    'customer_tier': random.choice(['basic', 'premium', 'enterprise']),
                    'channel': random.choice(['email', 'web', 'phone', 'chat']),
                    'urgency_keywords': self._extract_urgency_keywords(description)
                }
            }
            
            tickets.append(ticket)
        
        # Update user workloads based on open tickets
        self._update_user_workloads(tickets, user_lookup)
        
        return tickets
    
    def _generate_description(self, department: str, keywords: List[str]) -> str:
        """Generate realistic ticket description"""
        templates = [
            f"I'm experiencing issues with {random.choice(keywords)}. {fake.text(max_nb_chars=200)}",
            f"There seems to be a problem with {random.choice(keywords)}. Can you please help? {fake.text(max_nb_chars=150)}",
            f"Need assistance with {random.choice(keywords)}. {fake.text(max_nb_chars=180)}",
            f"Having trouble with {random.choice(keywords)}. This is affecting our workflow. {fake.text(max_nb_chars=160)}",
        ]
        
        description = random.choice(templates)
        
        # Add urgency indicators randomly
        if random.random() < 0.2:
            urgency_words = ['URGENT', 'ASAP', 'HIGH PRIORITY', 'CRITICAL']
            description = f"{random.choice(urgency_words)}: {description}"
        
        # Add technical details for technical tickets
        if department == 'technical_support' and random.random() < 0.5:
            tech_details = [
                "Error code: 500",
                "Browser: Chrome 91.0",
                "OS: Windows 10",
                "Steps to reproduce: 1) Login 2) Navigate to dashboard 3) Error occurs"
            ]
            description += f"\n\nTechnical details:\n{random.choice(tech_details)}"
        
        return description
    
    def _generate_resolution_time(self, priority: str, assignee: Dict[str, Any]) -> float:
        """Generate realistic resolution time based on priority and assignee"""
        base_times = {
            'critical': (0.5, 4),
            'high': (2, 12),
            'medium': (8, 48),
            'low': (24, 120)
        }
        
        min_time, max_time = base_times[priority]
        
        # Adjust based on assignee performance
        performance_factor = assignee['performance_metrics']['avg_resolution_time'] / 24.0
        
        resolution_time = random.uniform(min_time, max_time) * performance_factor
        return max(0.5, resolution_time)  # Minimum 30 minutes
    
    def _generate_tags(self, department: str, keywords: List[str]) -> List[str]:
        """Generate relevant tags for tickets"""
        tags = [department]
        
        # Add some keywords as tags
        tags.extend(random.sample(keywords, k=random.randint(1, 3)))
        
        # Add general tags
        general_tags = ['customer-issue', 'support', 'internal', 'external']
        tags.extend(random.sample(general_tags, k=random.randint(0, 2)))
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_urgency_keywords(self, description: str) -> List[str]:
        """Extract urgency keywords from description"""
        urgency_keywords = ['urgent', 'asap', 'critical', 'high priority', 'emergency', 'immediately']
        found_keywords = []
        
        description_lower = description.lower()
        for keyword in urgency_keywords:
            if keyword in description_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _update_user_workloads(self, tickets: List[Dict[str, Any]], 
                              user_lookup: Dict[int, Dict[str, Any]]):
        """Update user workloads based on open tickets"""
        workload_counts = {}
        
        for ticket in tickets:
            if ticket['status'] in ['open', 'in_progress']:
                assignee_id = ticket['assignee_id']
                workload_counts[assignee_id] = workload_counts.get(assignee_id, 0) + 1
        
        for user_id, count in workload_counts.items():
            if user_id in user_lookup:
                user_lookup[user_id]['current_workload'] = count
    
    def generate_feedback(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate feedback data for resolved tickets"""
        feedback_data = []
        
        resolved_tickets = [t for t in tickets if t['status'] in ['resolved', 'closed']]
        
        # Generate feedback for ~70% of resolved tickets
        tickets_with_feedback = random.sample(
            resolved_tickets, 
            k=int(len(resolved_tickets) * 0.7)
        )
        
        for ticket in tickets_with_feedback:
            # Determine if routing was correct (simulate based on department match)
            was_correctly_routed = random.random() < 0.85  # 85% accuracy
            
            # Generate ratings based on routing correctness and resolution time
            if was_correctly_routed:
                rating = random.choices([3, 4, 5], weights=[0.1, 0.4, 0.5])[0]
                resolution_quality = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
                response_satisfaction = random.choices([3, 4, 5], weights=[0.2, 0.3, 0.5])[0]
            else:
                rating = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
                resolution_quality = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
                response_satisfaction = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
            
            # Adjust based on resolution time
            if ticket['resolution_time_hours'] and ticket['resolution_time_hours'] > 72:
                response_satisfaction = max(1, response_satisfaction - 1)
            
            # Generate comments occasionally
            comments = None
            if random.random() < 0.3:
                if rating >= 4:
                    comments = random.choice([
                        "Great service, very helpful!",
                        "Quick resolution, thank you.",
                        "Professional and efficient.",
                        "Exactly what I needed."
                    ])
                else:
                    comments = random.choice([
                        "Took too long to resolve.",
                        "Could have been handled better.",
                        "Not satisfied with the solution.",
                        "Had to follow up multiple times."
                    ])
            
            feedback = {
                'ticket_id': ticket['id'],
                'rating': rating,
                'was_correctly_routed': was_correctly_routed,
                'resolution_quality': resolution_quality,
                'response_time_satisfaction': response_satisfaction,
                'comments': comments,
                'created_at': ticket['resolved_at'] + timedelta(hours=random.uniform(1, 48))
            }
            
            feedback_data.append(feedback)
        
        return feedback_data
    
    def generate_complete_dataset(self, num_users: int = 25, num_tickets: int = 1000,
                                 days_back: int = 90) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a complete dataset with users, tickets, and feedback"""
        
        print(f"Generating {num_users} users...")
        users = self.generate_users(num_users)
        
        print(f"Generating {num_tickets} tickets...")
        tickets = self.generate_tickets(num_tickets, users, days_back)
        
        print("Generating feedback data...")
        feedback = self.generate_feedback(tickets)
        
        dataset = {
            'users': users,
            'tickets': tickets,
            'feedback': feedback
        }
        
        print(f"Dataset generated:")
        print(f"  - Users: {len(users)}")
        print(f"  - Tickets: {len(tickets)}")
        print(f"  - Feedback entries: {len(feedback)}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, List[Dict[str, Any]]], 
                    output_dir: str = "data"):
        """Save dataset to JSON files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for data_type, data in dataset.items():
            filename = os.path.join(output_dir, f"{data_type}.json")
            
            # Convert datetime objects to strings for JSON serialization
            serializable_data = []
            for item in data:
                serializable_item = {}
                for key, value in item.items():
                    if isinstance(value, datetime):
                        serializable_item[key] = value.isoformat()
                    else:
                        serializable_item[key] = value
                serializable_data.append(serializable_item)
            
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            print(f"Saved {len(data)} {data_type} to {filename}")


if __name__ == "__main__":
    # Generate sample dataset
    generator = TicketDataGenerator()
    dataset = generator.generate_complete_dataset(
        num_users=25,
        num_tickets=1000,
        days_back=90
    )
    generator.save_dataset(dataset)