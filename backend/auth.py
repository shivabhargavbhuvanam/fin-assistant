"""
Module: auth.py
Description: Clerk JWT authentication for Smart Financial Coach.

Provides:
    - JWT verification using Clerk's public keys
    - get_current_user dependency for FastAPI
    - User ID extraction from verified tokens

Usage:
    @app.get("/protected")
    async def protected_route(user_id: str = Depends(get_current_user)):
        ...

Author: Smart Financial Coach Team
"""

import os
import jwt
import httpx
from typing import Optional
from functools import lru_cache
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

# Clerk configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")
CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_FRONTEND_API = os.getenv("CLERK_FRONTEND_API", "")

# JWT verification settings
# For development/demo, we can bypass auth
AUTH_BYPASS = os.getenv("AUTH_BYPASS", "false").lower() == "true"
AUTH_BYPASS_USER_ID = os.getenv("AUTH_BYPASS_USER_ID", "demo_user_123")


# =============================================================================
# Security Scheme
# =============================================================================

security = HTTPBearer(auto_error=False)


# =============================================================================
# JWT Verification
# =============================================================================

@lru_cache(maxsize=1)
def get_clerk_jwks_client():
    """
    Get Clerk JWKS client for JWT verification.
    
    Uses JWKS endpoint to fetch public keys for token verification.
    Cached to avoid repeated HTTP calls.
    """
    # Extract Clerk domain from secret key or use frontend API
    if CLERK_FRONTEND_API:
        jwks_url = f"https://{CLERK_FRONTEND_API}/.well-known/jwks.json"
    else:
        # Default fallback (user needs to configure)
        return None
    
    try:
        return jwt.PyJWKClient(jwks_url)
    except Exception as e:
        print(f"⚠️ Failed to initialize JWKS client: {e}")
        return None


def verify_clerk_token(token: str) -> Optional[dict]:
    """
    Verify a Clerk JWT token and return the claims.
    
    Args:
        token: The JWT token from Authorization header.
        
    Returns:
        Dict of token claims if valid, None otherwise.
    """
    if not token:
        return None
    
    # Development bypass
    if AUTH_BYPASS:
        return {"sub": AUTH_BYPASS_USER_ID}
    
    jwks_client = get_clerk_jwks_client()
    if not jwks_client:
        # If no JWKS configured, try to decode without verification (dev only!)
        if os.getenv("ENVIRONMENT", "development") == "development":
            try:
                # Decode without verification for development
                claims = jwt.decode(token, options={"verify_signature": False})
                return claims
            except Exception:
                return None
        return None
    
    try:
        # Get signing key from JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Verify and decode token
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False}  # Clerk doesn't always set audience
        )
        
        return claims
        
    except jwt.ExpiredSignatureError:
        print("⚠️ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"⚠️ Invalid token: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Token verification error: {e}")
        return None


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    FastAPI dependency to get the current authenticated user.
    
    Extracts and verifies the JWT from the Authorization header,
    returns the Clerk user ID (sub claim).
    
    Args:
        credentials: HTTP Bearer credentials from request.
        
    Returns:
        The Clerk user ID (sub claim from JWT).
        
    Raises:
        HTTPException: 401 if not authenticated or token invalid.
    """
    # Development bypass mode
    if AUTH_BYPASS:
        return AUTH_BYPASS_USER_ID
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    claims = verify_clerk_token(token)
    
    if not claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract user ID from claims
    # Clerk uses 'sub' for user ID
    user_id = claims.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    FastAPI dependency to optionally get the current user.
    
    Unlike get_current_user, this doesn't raise an error if not authenticated.
    Useful for endpoints that work differently for authenticated vs anonymous users.
    
    Returns:
        The Clerk user ID if authenticated, None otherwise.
    """
    if AUTH_BYPASS:
        return AUTH_BYPASS_USER_ID
    
    if not credentials:
        return None
    
    claims = verify_clerk_token(credentials.credentials)
    if not claims:
        return None
    
    return claims.get("sub")


# =============================================================================
# Utility Functions
# =============================================================================

def is_auth_configured() -> bool:
    """Check if Clerk authentication is configured."""
    return bool(CLERK_FRONTEND_API) or AUTH_BYPASS
