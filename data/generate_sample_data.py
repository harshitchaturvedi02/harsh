"""
Generate sample training data for the ticket routing system.
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os

from config.settings import settings


def generate_sample_tickets() -> List[Dict[str, Any]]:
    """Generate sample ticket data for training."""
    
    sample_tickets = [
        # Technical Support Tickets
        {
            "title": "Login Error - Cannot access account",
            "description": "I'm getting an error message when trying to log in. The system says 'Invalid credentials' but I'm sure my password is correct. This started happening yesterday.",
            "team": "technical_support",
            "priority": "high",
            "category": "authentication"
        },
        {
            "title": "API Connection Timeout",
            "description": "Our application is experiencing timeout errors when connecting to the API. The requests are taking more than 30 seconds to respond, which is causing our system to fail.",
            "team": "technical_support",
            "priority": "urgent",
            "category": "api_issues"
        },
        {
            "title": "Database Performance Issues",
            "description": "The database queries are running very slowly. We're seeing response times of 5-10 seconds for simple SELECT statements. This is affecting our application performance.",
            "team": "technical_support",
            "priority": "high",
            "category": "performance"
        },
        {
            "title": "System Crash During Backup",
            "description": "The system crashed while running the nightly backup process. We need immediate assistance to restore the backup and prevent data loss.",
            "team": "technical_support",
            "priority": "urgent",
            "category": "system_crash"
        },
        
        # Billing Support Tickets
        {
            "title": "Incorrect Invoice Amount",
            "description": "I received an invoice for $500 but I should only be charged $300 according to my subscription plan. Please review and correct this billing error.",
            "team": "billing_support",
            "priority": "medium",
            "category": "billing_error"
        },
        {
            "title": "Payment Method Update",
            "description": "I need to update my credit card information. The current card on file has expired and I want to add my new card details.",
            "team": "billing_support",
            "priority": "medium",
            "category": "payment_update"
        },
        {
            "title": "Subscription Cancellation",
            "description": "I would like to cancel my subscription. Please process the cancellation and confirm that I won't be charged for the next billing cycle.",
            "team": "billing_support",
            "priority": "medium",
            "category": "cancellation"
        },
        {
            "title": "Refund Request",
            "description": "I was charged twice for the same service. Please process a refund for the duplicate charge of $150 that was made on March 15th.",
            "team": "billing_support",
            "priority": "high",
            "category": "refund"
        },
        
        # Product Support Tickets
        {
            "title": "How to Export Data",
            "description": "I need help exporting my data from the platform. Can you provide step-by-step instructions on how to export all my project data?",
            "team": "product_support",
            "priority": "low",
            "category": "how_to"
        },
        {
            "title": "Feature Usage Question",
            "description": "I'm trying to use the new collaboration feature but I'm not sure how to invite team members. Is there a tutorial or guide available?",
            "team": "product_support",
            "priority": "low",
            "category": "feature_help"
        },
        {
            "title": "Integration Setup",
            "description": "I'm setting up the integration with our CRM system. Can you help me configure the webhook settings and API keys?",
            "team": "product_support",
            "priority": "medium",
            "category": "integration"
        },
        {
            "title": "Mobile App Issue",
            "description": "The mobile app is not syncing properly with the web version. Changes made on the web don't appear on my phone. How can I fix this?",
            "team": "product_support",
            "priority": "medium",
            "category": "mobile_app"
        },
        
        # General Inquiries
        {
            "title": "Account Information Request",
            "description": "I need to update my contact information and verify my account details. Can you help me access my account settings?",
            "team": "general_inquiries",
            "priority": "low",
            "category": "account_management"
        },
        {
            "title": "Company Information",
            "description": "I'm interested in learning more about your company and services. Can you send me information about your pricing plans and features?",
            "team": "general_inquiries",
            "priority": "low",
            "category": "information_request"
        },
        {
            "title": "Partnership Inquiry",
            "description": "I represent a company that would like to explore partnership opportunities. Who should I contact for business development discussions?",
            "team": "general_inquiries",
            "priority": "medium",
            "category": "partnership"
        },
        {
            "title": "Privacy Policy Question",
            "description": "I have questions about your privacy policy and how you handle user data. Can you clarify your data retention policies?",
            "team": "general_inquiries",
            "priority": "medium",
            "category": "privacy"
        },
        
        # Bug Reports
        {
            "title": "Search Function Not Working",
            "description": "The search function is broken. When I search for files, it returns no results even when I know the files exist. This is a critical bug.",
            "team": "bug_reports",
            "priority": "high",
            "category": "search_bug"
        },
        {
            "title": "Email Notifications Missing",
            "description": "I'm not receiving email notifications for important updates. I've checked my spam folder and email settings, but notifications are still not coming through.",
            "team": "bug_reports",
            "priority": "medium",
            "category": "notification_bug"
        },
        {
            "title": "UI Display Issue",
            "description": "The user interface is displaying incorrectly on Chrome browser. Text is overlapping and buttons are not clickable. This makes the application unusable.",
            "team": "bug_reports",
            "priority": "high",
            "category": "ui_bug"
        },
        {
            "title": "Data Loss Bug",
            "description": "I created a new project and saved it, but when I refreshed the page, all my work was gone. This is unacceptable and I need my data recovered.",
            "team": "bug_reports",
            "priority": "urgent",
            "category": "data_loss"
        },
        
        # Feature Requests
        {
            "title": "Dark Mode Request",
            "description": "I would love to see a dark mode option for the application. Many users prefer dark themes, especially when working in low-light environments.",
            "team": "feature_requests",
            "priority": "low",
            "category": "ui_improvement"
        },
        {
            "title": "Bulk Export Feature",
            "description": "It would be very helpful to have a bulk export feature that allows users to export multiple projects at once. Currently, we have to export them one by one.",
            "team": "feature_requests",
            "priority": "medium",
            "category": "export_feature"
        },
        {
            "title": "Advanced Analytics Dashboard",
            "description": "We need more advanced analytics and reporting features. The current dashboard is too basic. Please add more detailed metrics and customizable reports.",
            "team": "feature_requests",
            "priority": "medium",
            "category": "analytics"
        },
        {
            "title": "Mobile App Enhancement",
            "description": "The mobile app needs more features. Currently, it's very limited compared to the web version. Please add offline mode and more editing capabilities.",
            "team": "feature_requests",
            "priority": "medium",
            "category": "mobile_enhancement"
        }
    ]
    
    # Add more variations
    variations = [
        # Technical variations
        {"title": "Server Down", "description": "The server is completely down and not responding to any requests.", "team": "technical_support"},
        {"title": "Database Connection Failed", "description": "Cannot connect to the database. Getting connection timeout errors.", "team": "technical_support"},
        {"title": "SSL Certificate Expired", "description": "The SSL certificate has expired and users are getting security warnings.", "team": "technical_support"},
        
        # Billing variations
        {"title": "Double Charged", "description": "I was charged twice for the same service. Need a refund immediately.", "team": "billing_support"},
        {"title": "Wrong Plan Charged", "description": "I'm on the basic plan but was charged for the premium plan.", "team": "billing_support"},
        {"title": "Payment Declined", "description": "My payment was declined but the money was taken from my account.", "team": "billing_support"},
        
        # Product variations
        {"title": "How to Use API", "description": "I need help understanding how to use the API. Can you provide documentation?", "team": "product_support"},
        {"title": "Tutorial Request", "description": "I need a tutorial on how to set up the dashboard and configure notifications.", "team": "product_support"},
        {"title": "Feature Explanation", "description": "Can you explain how the new collaboration feature works?", "team": "product_support"},
        
        # General variations
        {"title": "Password Reset", "description": "I forgot my password and need to reset it. The reset link is not working.", "team": "general_inquiries"},
        {"title": "Account Deletion", "description": "I want to delete my account and all associated data. How do I do this?", "team": "general_inquiries"},
        {"title": "Contact Information", "description": "I need to update my contact information and email address.", "team": "general_inquiries"},
        
        # Bug variations
        {"title": "Login Bug", "description": "The login page is not working. Users cannot sign in to their accounts.", "team": "bug_reports"},
        {"title": "File Upload Error", "description": "File upload is failing with an error message. Files are not being saved.", "team": "bug_reports"},
        {"title": "Performance Issue", "description": "The application is running very slowly. Pages take 10+ seconds to load.", "team": "bug_reports"},
        
        # Feature variations
        {"title": "Calendar Integration", "description": "Please add calendar integration so we can sync with Google Calendar.", "team": "feature_requests"},
        {"title": "Multi-language Support", "description": "We need support for multiple languages in the application.", "team": "feature_requests"},
        {"title": "Advanced Search", "description": "The current search is too basic. We need advanced search with filters.", "team": "feature_requests"}
    ]
    
    # Add variations to sample tickets
    for variation in variations:
        sample_tickets.append({
            "title": variation["title"],
            "description": variation["description"],
            "team": variation["team"],
            "priority": np.random.choice(["low", "medium", "high", "urgent"]),
            "category": "general"
        })
    
    return sample_tickets


def create_training_dataset():
    """Create and save training dataset."""
    # Generate sample data
    tickets = generate_sample_tickets()
    
    # Create DataFrame
    df = pd.DataFrame(tickets)
    
    # Add some noise and variations
    df['text'] = df['title'] + ' ' + df['description']
    
    # Create training data directory
    os.makedirs(settings.TRAINING_DATA_PATH, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(settings.TRAINING_DATA_PATH, 'training_data.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as JSON for API testing
    json_path = os.path.join(settings.TRAINING_DATA_PATH, 'training_data.json')
    df.to_json(json_path, orient='records', indent=2)
    
    # Print summary
    print(f"Generated {len(tickets)} sample tickets")
    print(f"Team distribution:")
    print(df['team'].value_counts())
    print(f"\nFiles saved:")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    
    return df


if __name__ == "__main__":
    create_training_dataset()