from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    
router = APIRouter()

def get_response(question: str) -> str:
    url = "http://llm-agent:5080/llm"
    payload = {"question": question}
    try:
        print(f"Sending payload to LLM agent: {payload} at {url}")
        response = requests.post(url, json=payload, timeout=100)
        print(f"Received response: {response.status_code} - {response.text}")
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer provided by LLM agent.")
    except Exception as e:
        print(f"Error calling LLM agent: {str(e)}")
        return f"Error calling LLM agent: {str(e)}"

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    answer = get_response(chat_request.question)
    if answer.startswith("Error calling LLM agent"):
        raise HTTPException(status_code=500, detail=answer)
    return ChatResponse(answer=answer)
