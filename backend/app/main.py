from fastapi import FastAPI
from app.routes import chat

app = FastAPI(title="Simple Chatbot API")

# Include the chat endpoint under the /api prefix
app.include_router(chat.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
