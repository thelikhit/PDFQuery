import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from src.app.db.rdb import get_all_api_keys

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

async def check_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    try:
        valid_keys: set = set(get_all_api_keys())
    except Exception:
        logger.exception("Failed to fetch API keys from database")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable. Please try again later.",
        )

    if not valid_keys:
        logger.warning("API key table is empty — all requests will be rejected")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable. Please try again later.",
        )

    if api_key not in valid_keys:
        logger.warning("Rejected invalid API key: %.8s...", api_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key