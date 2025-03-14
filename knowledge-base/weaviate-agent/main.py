import os
import random
import weaviate
import weaviate.classes as wvc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Weaviate Knowledge Base Agent")

# --- Data Models ---
class QueryRequest(BaseModel):
    query: str

class Document(BaseModel):
    question: str
    answer: str = ""
    category: str = ""

# --- Correct Weaviate Client Initialization (V4) ---
client = weaviate.connect_to_custom(
    http_host=os.getenv("WEAVIATE_HOST", "weaviate"),
    http_port=int(os.getenv("WEAVIATE_PORT", "8080")),
    http_secure=False,
    grpc_host=os.getenv("WEAVIATE_HOST", "weaviate"),
    grpc_port=50051,
    grpc_secure=False,
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

def create_question_collection():
    if client.collections.exists("Question"):
        print("Collection 'Question' already exists.")
        return

    questions = client.collections.create(
        name="Question",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.cohere(),
        properties=[
            wvc.config.Property("question", wvc.config.DataType.TEXT),
            wvc.config.Property("answer", wvc.config.DataType.TEXT),
            wvc.config.Property("category", wvc.config.DataType.TEXT),
        ]
    )
    print("Collection 'Question' created.")

@app.on_event("startup")
async def startup_event():
    create_question_collection()

    questions_collection = client.collections.get("Question")
    existing = questions_collection.query.fetch_objects(limit=1)
    if len(existing.objects) == 0:
        sample_docs = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "geography"
            },
            {
                "question": "Who wrote 'Hamlet'?",
                "answer": "William Shakespeare",
                "category": "literature"
            },
            {
                "question": "What is the boiling point of water?",
                "answer": "100°C at sea level",
                "category": "science"
            }
        ]
        random_doc = random.choice(sample_docs)
        questions_collection.data.insert(random_doc)
        print(f"Random document indexed: {random_doc}")

@app.post("/context")
async def get_context(request: QueryRequest):
    try:
        questions_collection = client.collections.get("Question")
        response = questions_collection.query.near_text(
            query=request.query,
            limit=3,
            return_properties=["question", "answer", "category"]
        )
        results = [
            {
                "question": obj.properties["question"],
                "answer": obj.properties["answer"],
                "category": obj.properties.get("category", "")
            }
            for obj in response.objects
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add_document(doc: Document):
    try:
        questions_collection = client.collections.get("Question")
        res = questions_collection.data.insert(doc.dict())
        return {"status": "success", "object_id": res.uuid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
