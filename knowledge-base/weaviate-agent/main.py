import os
import weaviate
import weaviate.classes as wvc
import pypdf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str

class LabourDocument(BaseModel):
    text: str
    chunk_id: int

# --- Weaviate Client Initialization ---
client = weaviate.connect_to_custom(
    http_host=os.getenv("WEAVIATE_HOST", "weaviate"),
    http_port=int(os.getenv("WEAVIATE_PORT", "8080")),
    http_secure=False,
    grpc_host=os.getenv("WEAVIATE_HOST", "weaviate"),
    grpc_port=50051,
    grpc_secure=False,
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

def create_labour_act_collection():
    """
    Creates the 'LabourAct' collection if it doesn't already exist.
    """
    if client.collections.exists("LabourAct"):
        print("Collection 'LabourAct' already exists.")
        return

    client.collections.create(
        name="LabourAct",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
        ]
    )
    print("Collection 'LabourAct' created.")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000):
    """
    Splits text into chunks of `chunk_size` words.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest_labour_act_pdf():
    """
    Extracts text from the Labour Act PDF and stores it in Weaviate.
    """
    pdf_path = "data/Labour Act.pdf"

    # Extract and chunk text
    document_text = extract_text_from_pdf(pdf_path)
    document_chunks = chunk_text(document_text)

    # Insert chunks into Weaviate
    labour_collection = client.collections.get("LabourAct")
    for i, chunk in enumerate(document_chunks):
        labour_collection.data.insert(properties={"text": chunk, "chunk_id": i})
    
    print(f"Inserted {len(document_chunks)} chunks from Labour Act PDF.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_labour_act_collection()

    # Ingest Labour Act content if not already present
    labour_collection = client.collections.get("LabourAct")
    existing_docs = labour_collection.query.fetch_objects(limit=1)
    if not existing_docs.objects:
        ingest_labour_act_pdf()

    yield
    client.close()

app = FastAPI(title="Weaviate Labour Act Agent", lifespan=lifespan)

@app.post("/context")
async def get_context(request: QueryRequest):
    """
    Given a question, performs a near_text search in the 'LabourAct' collection.
    """
    try:
        labour_collection = client.collections.get("LabourAct")
        response = labour_collection.query.near_text(
            query=request.question,
            limit=3,
            return_properties=["text"]
        )
        print("Raw Weaviate Response:", response)
        results = [{"text": obj.properties.get("text", "")} for obj in response.objects]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
