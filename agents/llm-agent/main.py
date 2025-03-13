from fastapi import FastAPI
from pydantic import BaseModel
from agent import generate_llm_response

app = FastAPI(title="LLM Agent Service")

class LLMRequest(BaseModel):
    question: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/llm")
async def query_llm(request: LLMRequest):
    answer = generate_llm_response(request.question)
    return {"answer": answer}