from fastapi import FastAPI
from src.app.api import lifespan, register_routers

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

register_routers(app)