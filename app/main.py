from fastapi import FastAPI
from app.api.routes import chat, upload

app = FastAPI()
app.include_router(chat.router)
app.include_router(upload.router)

@app.get("/")
async def root():
    return {"message": "API is running"}
