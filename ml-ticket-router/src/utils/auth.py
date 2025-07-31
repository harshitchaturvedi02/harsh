"""
Authentication module for API security.
"""

import os
import hashlib
import secrets
from typing import Optional, List
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery
from datetime import datetime, timedelta
import logging
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# API key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# In production, these would be stored in a database
VALID_API_KEYS = {
    "demo-key-123": {
        "name": "Demo User",
        "created_at": datetime.now(),
        "permissions": ["read", "write"],
        "rate_limit": 100
    }
}

# Load API keys from environment
if os.getenv("API_KEYS"):
    for key in os.getenv("API_KEYS").split(","):
        VALID_API_KEYS[key] = {
            "name": "Environment Key",
            "created_at": datetime.now(),
            "permissions": ["read", "write"],
            "rate_limit": 1000
        }


class APIKeyManager:
    """Manages API keys and permissions."""
    
    def __init__(self):
        self.keys = VALID_API_KEYS.copy()
        self.usage_stats = {}
        
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        # Generate secure random key
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        
        self.keys[api_key] = {
            "name": name,
            "created_at": datetime.now(),
            "permissions": permissions or ["read"],
            "rate_limit": 100
        }
        
        logger.info(f"Generated new API key for {name}")
        return api_key
        
    def validate_key(self, api_key: str) -> bool:
        """Validate if API key exists and is active."""
        return api_key in self.keys
        
    def check_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""
        if api_key not in self.keys:
            return False
            
        key_info = self.keys[api_key]
        return permission in key_info.get("permissions", [])
        
    def record_usage(self, api_key: str):
        """Record API key usage for rate limiting."""
        if api_key not in self.usage_stats:
            self.usage_stats[api_key] = []
            
        self.usage_stats[api_key].append(datetime.now())
        
        # Clean old entries (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.usage_stats[api_key] = [
            ts for ts in self.usage_stats[api_key] if ts > cutoff
        ]
        
    def check_rate_limit(self, api_key: str) -> bool:
        """Check if API key has exceeded rate limit."""
        if api_key not in self.keys:
            return False
            
        rate_limit = self.keys[api_key].get("rate_limit", 100)
        current_usage = len(self.usage_stats.get(api_key, []))
        
        return current_usage < rate_limit
        
    def revoke_key(self, api_key: str):
        """Revoke an API key."""
        if api_key in self.keys:
            del self.keys[api_key]
            logger.info(f"Revoked API key: {api_key[:10]}...")
            
    def list_keys(self) -> List[dict]:
        """List all API keys (without exposing the actual keys)."""
        return [
            {
                "key_prefix": key[:10] + "...",
                "name": info["name"],
                "created_at": info["created_at"],
                "permissions": info["permissions"],
                "rate_limit": info["rate_limit"]
            }
            for key, info in self.keys.items()
        ]


# Global API key manager instance
api_key_manager = APIKeyManager()


async def verify_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query)
) -> str:
    """
    Verify API key from header or query parameter.
    
    Args:
        api_key_header: API key from header
        api_key_query: API key from query parameter
        
    Returns:
        Valid API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    # Check if API key authentication is disabled (for development)
    if os.getenv("DISABLE_AUTH", "false").lower() == "true":
        return "dev-mode"
        
    # Get API key from header or query
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
        
    # Validate API key
    if not api_key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
        
    # Check rate limit
    if not api_key_manager.check_rate_limit(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
        
    # Record usage
    api_key_manager.record_usage(api_key)
    
    return api_key


async def require_permission(permission: str):
    """
    Dependency to require specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def permission_checker(api_key: str = Depends(verify_api_key)):
        if not api_key_manager.check_permission(api_key, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return api_key
        
    return permission_checker


def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_demo_keys():
    """Create demo API keys for testing."""
    demo_keys = [
        ("Demo User", ["read"]),
        ("Admin User", ["read", "write", "admin"]),
        ("ML Service", ["read", "write", "retrain"])
    ]
    
    generated_keys = []
    for name, permissions in demo_keys:
        key = api_key_manager.generate_api_key(name, permissions)
        generated_keys.append({
            "name": name,
            "key": key,
            "permissions": permissions
        })
        
    return generated_keys


# Middleware for API key validation in specific routes
class APIKeyMiddleware:
    """Middleware for API key validation."""
    
    def __init__(self, excluded_paths: List[str] = None):
        self.excluded_paths = excluded_paths or ["/", "/health", "/docs", "/openapi.json"]
        
    async def __call__(self, request, call_next):
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
            
        # Verify API key
        api_key = request.headers.get(API_KEY_NAME) or request.query_params.get("api_key")
        
        if not api_key or not api_key_manager.validate_key(api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"}
            )
            
        # Continue with request
        response = await call_next(request)
        return response