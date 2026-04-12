import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.app.api.routes import upload, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting up API...")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "API is running"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}