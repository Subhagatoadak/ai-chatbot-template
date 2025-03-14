import os
import random
import weaviate
import weaviate.classes as wvc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str  # Note: using "question" key to match expected payload

class Document(BaseModel):
    question: str
    answer: str = ""
    category: str = ""

# --- Weaviate Client Initialization (v4) ---
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
    """
    Creates the 'Question' collection if it doesn't already exist.
    """
    if client.collections.exists("Question"):
        print("Collection 'Question' already exists.")
        return

    client.collections.create(
        name="Question",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.cohere(),
        properties=[
            wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
        ]
    )
    print("Collection 'Question' created.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create collection and insert a random sample document if none exist
    create_question_collection()
    questions_collection = client.collections.get("Question")
    existing_docs = questions_collection.query.fetch_objects(limit=1)
    if not existing_docs.objects:
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
        questions_collection.data.insert(properties=random_doc)
        print("Random document indexed:", random_doc)
    yield
    client.close()

app = FastAPI(title="Weaviate Knowledge Base Agent", lifespan=lifespan)

@app.post("/context")
async def get_context(request: QueryRequest):
    """
    Given a question, perform a near_text search in the 'Question' collection
    to retrieve similar documents.
    """
    try:
        questions_collection = client.collections.get("Question")
        response = questions_collection.query.near_text(
            query=request.question,
            limit=3,
            return_properties=["question", "answer", "category"]
        )
        results = [
            {
                "question": obj.properties.get("question", ""),
                "answer": obj.properties.get("answer", ""),
                "category": obj.properties.get("category", "")
            }
            for obj in response.objects
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add_document(doc: Document):
    """
    Adds a new document into the 'Question' collection.
    """
    try:
        questions_collection = client.collections.get("Question")
        res = questions_collection.data.insert(properties=doc.dict())
        return {"status": "success", "object_id": str(res.uuid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
