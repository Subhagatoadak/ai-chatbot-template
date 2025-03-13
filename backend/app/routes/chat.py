from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.llm_agent.agent import get_llm_response

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    API endpoint that receives a user's question, fetches the answer
    from the OpenAI LLM, and returns it.
    """
    answer = get_llm_response(chat_request.question)
    if answer.startswith("Error calling OpenAI API"):
        raise HTTPException(status_code=500, detail=answer)
    return ChatResponse(answer=answer)
