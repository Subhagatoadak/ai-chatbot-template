from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

router = APIRouter()

# --- Data Models ---
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# --- Helper function to retrieve context from Weaviate ---
def get_context_for_query(query: str) -> dict:
    """
    Calls the Weaviate Agent service to get context documents for a query.
    """
    url = "http://weaviate-agent:8001/context"
    payload = {"query": query}
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- Helper function to call LLM Agent with context ---
def get_response(question: str) -> str:
    context_response = get_context_for_query(question)

    if "error" in context_response:
        combined_prompt = question  # fallback to question only
    else:
        # Formatting context neatly to pass to LLM agent
        context_text = ""
        for item in context_response:
            context_question = item.get('question', '')
            context_answer = item.get('answer', '')
            context_text += f"Q: {context_response}\nA: {context_response}\n\n"

        combined_prompt = f"""
        Given the context below, answer the user's question.

        User's question: {question}

        Context:
        {context_text}
        """
    print("Context",combined_prompt)
    url = "http://llm-agent:5080/llm"
    payload = {"question": combined_prompt}

    try:
        response = requests.post(url, json=payload, timeout=100)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer provided by LLM agent.")
    except Exception as e:
        return f"Error calling LLM agent: {str(e)}"

# --- Route to handle chat requests ---
@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    answer = get_response(chat_request.question)
    if answer.startswith("Error calling LLM agent"):
        raise HTTPException(status_code=500, detail=answer)
    return ChatResponse(answer=answer)
