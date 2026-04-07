from fastapi import FastAPI
from src.app.api.routes import upload, chat

app = FastAPI()
app.include_router(chat.router)
app.include_router(upload.router)

@app.get("/")
async def root():
    return {"message": "API is running"}
