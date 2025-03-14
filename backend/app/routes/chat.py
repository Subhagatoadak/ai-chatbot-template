from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

router = APIRouter()

# --- Data Models ---
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

def get_context_for_query(query: str) -> dict:
    """
    Calls the Weaviate Agent's /context endpoint to retrieve context documents.
    Expects a payload with the key "question".
    """
    url = "http://weaviate-agent:8001/context"
    payload = {"question": query}
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        # Log raw response for debugging
        print("Raw context response:", response.json())
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_llm_response(question: str) -> str:
    """
    Retrieves context from the Weaviate Agent, builds a combined prompt,
    and calls the LLM Agent to get the answer.
    """
    context_response = get_context_for_query(question)
    if "error" in context_response:
        combined_prompt = question  # Fallback: use the question only
    else:
        context_text = ""
        if "results" in context_response:
            for item in context_response["results"]:
                q = item.get("question", "")
                a = item.get("answer", "")
                context_text += f"Q: {q}\nA: {a}\n\n"
        else:
            # In case the raw response is a list
            for item in context_response:
                q = item.get("question", "")
                a = item.get("answer", "")
                context_text += f"Q: {q}\nA: {a}\n\n"
        combined_prompt = (
            f"Given the context below, answer the user's question.\n\n"
            f"User's question: {question}\n\n"
            f"Context:\n{context_text}"
        )
    print("Combined prompt:", combined_prompt)
    url = "http://llm-agent:5080/llm"
    payload = {"question": combined_prompt}
    try:
        response = requests.post(url, json=payload, timeout=100)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer provided by LLM agent.")
    except Exception as e:
        return f"Error calling LLM agent: {str(e)}"

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    answer = get_llm_response(chat_request.question)
    if answer.startswith("Error calling LLM agent"):
        raise HTTPException(status_code=500, detail=answer)
    return ChatResponse(answer=answer)
