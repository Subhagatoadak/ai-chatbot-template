import os
import asyncio
import weaviate
import weaviate.classes as wvc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str

class Document(BaseModel):
    question: str
    answer: str = ""
    category: str = ""

# --- Asynchronous Weaviate Client Initialization (v4) ---
# Using the asynchronous client provided by Weaviate.
client = weaviate.use_async_with_custom(
    http_host=os.getenv("WEAVIATE_HOST", "weaviate"),
    http_port=int(os.getenv("WEAVIATE_PORT", "8080")),
    http_secure=False,
    grpc_host=os.getenv("WEAVIATE_HOST", "weaviate"),
    grpc_port=50051,
    grpc_secure=False,
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

async def wait_for_weaviate(client, retries=10, delay=5):
    """
    Connects the client once and then retries checking if Weaviate is ready.
    """
    try:
        await client.connect()
    except Exception as e:
        print(f"Initial client.connect() failed: {e}")
    for i in range(retries):
        try:
            await client.is_ready()
            print("Weaviate is ready.")
            return
        except Exception as e:
            print(f"Attempt {i+1}/{retries}: Weaviate not ready, waiting {delay} seconds. Error: {e}")
            await asyncio.sleep(delay)
    raise Exception("Weaviate did not become ready in time.")

async def create_question_collection():
    """
    Creates the 'Question' collection (class) if it doesn't already exist.
    """
    if await client.collections.exists("Question"):
        print("Collection 'Question' already exists.")
        return

    await client.collections.create(
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
    # Connect the client once and wait until Weaviate is ready.
    await wait_for_weaviate(client, retries=10, delay=5)
    # Create the collection if it doesn't exist.
    await create_question_collection()
    questions_collection = client.collections.get("Question")
    existing_docs = await questions_collection.query.fetch_objects(limit=1)
    if not existing_docs.objects:
        sample_docs = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris is the capital city of France, known for its art, architecture, and culture.",
                "category": "geography"
            },
            {
                "question": "Who wrote 'Hamlet'?",
                "answer": "William Shakespeare, the renowned English playwright, authored 'Hamlet', one of the most famous tragedies.",
                "category": "literature"
            },
            {
                "question": "What is the boiling point of water?",
                "answer": "Under standard atmospheric conditions, water boils at 100°C (212°F), a fundamental fact in science.",
                "category": "science"
            }
        ]
        for doc in sample_docs:
            await questions_collection.data.insert(properties=doc)
        print("Sample documents indexed:", sample_docs)
    yield
    await client.close()

app = FastAPI(title="Weaviate Knowledge Base Agent", lifespan=lifespan)

@app.post("/context")
async def get_context(request: QueryRequest):
    """
    Given a question, performs a near_text search in the 'Question' collection
    to retrieve similar documents.
    """
    try:
        await client.is_ready()
        questions_collection = client.collections.get("Question")
        response = await questions_collection.query.near_text(
            query=request.question,
            limit=1,
            return_properties=["question", "answer", "category"]
        )
        print("Raw near_text response:", response)
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
    Inserts a new document into the 'Question' collection.
    """
    try:
        await client.is_ready()
        questions_collection = client.collections.get("Question")
        res = await questions_collection.data.insert(properties=doc.dict())
        return {"status": "success", "object_id": str(res.uuid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
