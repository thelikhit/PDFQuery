import logging
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from src.app.core.rag_service import rag
from src.app.api.routes.auth import check_api_key

from fastapi import Request


logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100)

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt must not be empty or whitespace only")
        return stripped

class ChatResponse(BaseModel):
    response: str

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    request: ChatRequest,
    req: Request,
    _: str = Depends(check_api_key),
):
    print(dict(req.headers))
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(rag, request.prompt),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        logger.error("RAG call timed out for prompt: %.100s", request.prompt)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The request timed out. Please try again.",
        )
    except ValueError as e:
        logger.warning("Invalid input to RAG: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {e}",
        )
    except Exception as e:
        logger.exception("Unexpected error during RAG execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later.",
        )

    if not response or not response.strip():
        logger.warning("RAG returned empty response for prompt: %.100s", request.prompt)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No response was generated. Please rephrase your prompt.",
        )

    return ChatResponse(response=response.strip())
