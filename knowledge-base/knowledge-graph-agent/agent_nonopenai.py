import os
import re
import json
import numpy as np
import networkx as nx
import nltk
import openai
import pypdf
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from transformers import pipeline
import torch
import faiss
from functools import lru_cache

# Set Transformers to offline mode (make sure models are already cached)
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure nltk data is available
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("maxent_ne_chunker")
nltk.download('maxent_ne_chunker_tab')
nltk.download("words")

load_dotenv()

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Retrieve API keys from .env (OpenAI key is not used for relevance here)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_PAASOWRD = os.getenv("NEO4J_PASSWORD")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Set up device for models (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize Hugging Face zero-shot classification pipeline for relevance scoring
hf_relevance_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device=="cuda" else -1)

# --- Pydantic Models ---
class sentencscore(BaseModel):
    score: float

class QueryRequest(BaseModel):
    question: str

# The response now returns a structured dict.
class QueryResponse(BaseModel):
    results: dict

# --- Helper Functions for Adaptive Chunking ---
@lru_cache(maxsize=128)
def determine_optimal_chunk_size(text, target_chunks=10, min_chunk=5, max_chunk=30):
    """
    Determines an optimal chunk size (in words) for a given text,
    aiming to produce approximately target_chunks chunks.
    Clamped between min_chunk and max_chunk.
    Caching is applied.
    """
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return min_chunk
    optimal = n_words // target_chunks
    optimal = max(min_chunk, min(optimal, max_chunk))
    return optimal

def split_into_chunks(text, optimal_size=None, overlap=3):
    """
    Splits text into smaller overlapping chunks.
    If optimal_size is not provided, it is determined dynamically.
    Filters out empty chunks.
    """
    words = text.split()
    if optimal_size is None:
        optimal_size = determine_optimal_chunk_size(text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+optimal_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += optimal_size - overlap
    return chunks

# --- PDF and Text Processing Functions ---
def load_pdf_text(file_path):
    """
    Loads and extracts text from a PDF file.
    """
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_paragraphs(text):
    """
    Splits the text into paragraphs using double newlines.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Extracted {len(paragraphs)} paragraphs.")
    return paragraphs

def extract_sentences(text):
    """
    Uses NLTK's sentence tokenizer to extract sentences.
    """
    sentences = nltk.sent_tokenize(text)
    print(f"Extracted {len(sentences)} sentences.")
    return sentences

def extract_named_entities(text):
    """
    Uses NLTK's ne_chunk to extract named entities.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(pos_tags)
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            phrase = " ".join(word for word, pos in subtree.leaves())
            entities.append(phrase)
    entities = list(set(entities))
    print(f"Extracted {len(entities)} named entities.")
    return entities

def extract_phrases_from_sentence(sentence):
    """
    Extracts candidate phrases (named entities) from a sentence.
    """
    return extract_named_entities(sentence)

def embed_text(model, text_list, precision=6):
    """
    Generate embeddings for a list of text items with controlled precision.
    Returns an empty array if no text is provided.
    Batch processing is used.
    """
    if not text_list:
        return np.array([])
    embeddings = model.encode(text_list, batch_size=32, show_progress_bar=False)
    embeddings = np.array(embeddings)
    return np.round(embeddings, precision)

def adaptive_multiscale_grid(text_list):
    """
    Assigns items to 'short', 'medium', and 'long' categories based on token count percentiles.
    """
    lengths = np.array([len(item.split()) for item in text_list])
    short_threshold = np.percentile(lengths, 25)
    long_threshold = np.percentile(lengths, 75)
    grid = {"short": [0] * len(text_list),
            "medium": [0] * len(text_list),
            "long": [0] * len(text_list)}
    for i, item in enumerate(text_list):
        length = len(item.split())
        if length <= short_threshold:
            grid["short"][i] = 1
        elif length >= long_threshold:
            grid["long"][i] = 1
        else:
            grid["medium"][i] = 1
    return grid

# --- Hugging Face Relevance Scoring ---
def get_hf_relevance_score(phrase, context):
    """
    Uses a Hugging Face zero-shot classification pipeline to compute a relevance score.
    Returns the score for the label "relevant".
    """
    if len(context.split()) > 200:
        context = " ".join(context.split()[:200])
    hypothesis = f"This context is relevant to the phrase: {phrase}"
    result = hf_relevance_pipeline(context, candidate_labels=["relevant", "not relevant"])
    if "relevant" in result["labels"]:
        idx = result["labels"].index("relevant")
        score = result["scores"][idx]
        print(f"HuggingFace relevance score for phrase '{phrase}': {score}")
        return score
    return 0.5

def update_ca(scale, grid, items, embeddings):
    """
    Cellular Automata update for a single scale using cosine similarity and Hugging Face relevance score.
    Batch processes relevance scores for efficiency.
    """
    new_grid = grid[scale].copy()
    indices = list(range(1, len(grid[scale]) - 1))
    contexts = [" | ".join([items[i - 1], items[i], items[i + 1]]) for i in indices]
    batch_results = hf_relevance_pipeline(contexts, candidate_labels=["relevant", "not relevant"])
    for idx, i in enumerate(indices):
        left, center, right = embeddings[i - 1], embeddings[i], embeddings[i + 1]
        sim_left = 1 - cosine(left, center)
        sim_right = 1 - cosine(right, center)
        result = batch_results[idx]
        hf_score = result["scores"][result["labels"].index("relevant")] if "relevant" in result["labels"] else 0.5
        final_score_left = (1 - 0.3) * sim_left + 0.3 * hf_score
        final_score_right = (1 - 0.3) * sim_right + 0.3 * hf_score
        if grid[scale][i] == 0 and (final_score_left > 0.7 or final_score_right > 0.7):
            new_grid[i] = 1
        elif grid[scale][i] == 1 and (final_score_left < 0.4 and final_score_right < 0.4):
            new_grid[i] = 0
    return scale, new_grid

# --- Graph Chunking and Retrieval Classes ---
class MultiscaleCellularAutomataGraphChunker:
    """
    Processes a document into a hierarchical graph and pushes it to Neo4j.
    Hierarchy: Document → Context → Paragraph → Subchunk → Sentence → Phrase
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", max_iters=2,
                 neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j",
                 neo4j_pass="your_password"):
        try:
            self.embedding_model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print("Error loading SentenceTransformer. Ensure the model is cached for offline mode.")
            raise e
        self.max_iters = max_iters
        self.graph = nx.DiGraph()
        np.random.seed(42)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_pass = neo4j_pass

    def process_documents(self, documents):
        """
        Process each document into a hierarchy with fine-grained chunks.
        """
        for doc_idx, doc_text in enumerate(documents):
            print(f"Processing document {doc_idx}...")
            doc_embedding = embed_text(self.embedding_model, [doc_text])[0]
            doc_node_id = f"Document_{doc_idx}"
            self.graph.add_node(doc_node_id, type="document", content=doc_text, embedding=doc_embedding.tolist())
            
            context_node_id = f"Context_{doc_idx}"
            context_text = doc_text
            context_embedding = embed_text(self.embedding_model, [context_text])[0]
            self.graph.add_node(context_node_id, type="context", content=context_text, embedding=context_embedding.tolist())
            self.graph.add_edge(doc_node_id, context_node_id)
            
            paragraphs = extract_paragraphs(doc_text)
            print(f"Document {doc_idx} has {len(paragraphs)} paragraphs.")
            all_chunks = []
            all_embeddings = []
            for p_idx, paragraph_text in enumerate(paragraphs):
                optimal_size = determine_optimal_chunk_size(paragraph_text, target_chunks=10, min_chunk=5, max_chunk=30)
                print(f"Optimal chunk size for paragraph {p_idx}: {optimal_size} words")
                subchunks = split_into_chunks(paragraph_text, optimal_size=optimal_size, overlap=3)
                print(f"Paragraph {p_idx} split into {len(subchunks)} subchunks.")
                
                para_node_id = f"Doc{doc_idx}_Para_{p_idx}"
                self.graph.add_node(para_node_id, type="paragraph", content=paragraph_text)
                self.graph.add_edge(context_node_id, para_node_id)
                
                for c_idx, subchunk in enumerate(subchunks):
                    if not subchunk.strip():
                        continue
                    subchunk_node_id = f"Doc{doc_idx}_Para_{p_idx}_Sub_{c_idx}"
                    subchunk_embedding = embed_text(self.embedding_model, [subchunk])[0]
                    self.graph.add_node(subchunk_node_id, type="subchunk", content=subchunk, embedding=subchunk_embedding.tolist())
                    self.graph.add_edge(para_node_id, subchunk_node_id)
                    all_chunks.append(subchunk)
                    all_embeddings.append(subchunk_embedding)
                    
                    sentences = extract_sentences(subchunk)
                    print(f"Subchunk {c_idx} has {len(sentences)} sentences.")
                    if not sentences:
                        continue
                    sent_embeddings = embed_text(self.embedding_model, sentences)
                    for s_idx, sentence_text in enumerate(sentences):
                        if not sentence_text.strip():
                            continue
                        sent_node_id = f"Doc{doc_idx}_Para_{p_idx}_Sub_{c_idx}_Sent_{s_idx}"
                        sent_emb = sent_embeddings[s_idx]
                        self.graph.add_node(sent_node_id, type="sentence", content=sentence_text, embedding=sent_emb.tolist())
                        self.graph.add_edge(subchunk_node_id, sent_node_id)
                        all_chunks.append(sentence_text)
                        all_embeddings.append(sent_emb)
                        
                        phrases = extract_phrases_from_sentence(sentence_text)
                        print(f"Sentence {s_idx} has {len(phrases)} phrases.")
                        if not phrases:
                            continue
                        phrase_embeddings = embed_text(self.embedding_model, phrases)
                        for ph_idx, phrase_text in enumerate(phrases):
                            if not phrase_text.strip():
                                continue
                            phrase_node_id = f"Doc{doc_idx}_Para_{p_idx}_Sub_{c_idx}_Sent_{s_idx}_Ph_{ph_idx}"
                            ph_emb = phrase_embeddings[ph_idx]
                            self.graph.add_node(phrase_node_id, type="phrase", content=phrase_text, embedding=ph_emb.tolist())
                            self.graph.add_edge(sent_node_id, phrase_node_id)
                            all_chunks.append(phrase_text)
                            all_embeddings.append(ph_emb)
            print(f"Total chunks extracted: {len(all_chunks)}")
            if len(all_chunks) > 2:
                all_embeddings = np.array(all_embeddings)
                grid = adaptive_multiscale_grid(all_chunks)
                with Pool(cpu_count()) as pool:
                    for _ in range(self.max_iters):
                        results = pool.starmap(
                            update_ca,
                            [(scale, grid, all_chunks, all_embeddings) for scale in ["short", "medium", "long"]]
                        )
                        for scale, updated_grid in results:
                            grid[scale] = updated_grid
        print(f"Graph processing complete. Total nodes: {len(self.graph.nodes())}, edges: {len(self.graph.edges())}")

    def save_graph_to_neo4j(self, clear_db=True):
        """
        Pushes the built graph to Neo4j using batch writes.
        """
        graph_data = nx.node_link_data(self.graph)
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_pass))
        with driver.session() as session:
            if clear_db:
                session.run("MATCH (n) DETACH DELETE n")
                print("Cleared existing Neo4j data.")
            session.run("""
                UNWIND $nodes AS node
                MERGE (n:Chunk {id: node.id})
                SET n.type = node.type,
                    n.content = node.content,
                    n.embedding = node.embedding
            """, nodes=graph_data["nodes"])
            session.run("""
                UNWIND $links AS link
                MATCH (a:Chunk {id: link.source}), (b:Chunk {id: link.target})
                MERGE (a)-[:HAS_CHILD]->(b)
            """, links=graph_data["links"])
        print("Graph stored in Neo4j.")
        with driver.session() as session:
            result = session.run("MATCH (n:Chunk) RETURN count(n) AS count")
            count = result.single()["count"]
            print("Number of nodes in Neo4j:", count)

class GraphRetriever:
    """
    Connects to Neo4j and enables vector-based similarity search and context retrieval.
    Only retrieves nodes of type 'subchunk', 'sentence', or 'phrase' (fine-grained chunks).
    Uses FAISS for approximate nearest neighbor search.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j", neo4j_pass="your_password"):
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.index = None
        self.id_map = []  # Mapping from FAISS index to node info
        self._build_faiss_index()

    def _build_faiss_index(self):
        """
        Retrieves all fine-grained nodes from Neo4j and builds a FAISS index.
        """
        with self.driver.session() as session:
            query = """
            MATCH (n:Chunk)
            WHERE n.type IN ['subchunk', 'sentence', 'phrase']
            RETURN n.id as id, n.embedding as embedding, n.type as type, n.content as content
            """
            result = session.run(query)
            records = result.data()
        embeddings = []
        self.id_map = []
        for rec in records:
            emb_data = rec["embedding"]
            if not emb_data:
                continue
            emb_vec = np.array(json.loads(emb_data)).astype('float32')
            if emb_vec.size == 0:
                continue
            embeddings.append(emb_vec)
            self.id_map.append({
                "id": rec["id"],
                "type": rec["type"],
                "content": rec["content"]
            })
        if embeddings:
            embeddings = np.vstack(embeddings)
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)
            print(f"Built FAISS index with {self.index.ntotal} vectors.")
        else:
            print("No fine-grained nodes found to build FAISS index.")

    def retrieve_similar_nodes(self, query_text, top_k=3, sim_threshold=0.5):
        """
        Embeds the query, searches the FAISS index for the top matching nodes,
        and returns only nodes with cosine similarity >= sim_threshold.
        """
        query_embedding = self.embedding_model.encode([query_text])[0].astype('float32')
        if self.index is None or self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([query_embedding]), top_k * 5)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            sim = 1 - (dist / 2)
            if sim < sim_threshold:
                continue
            node_info = self.id_map[idx]
            results.append((node_info["id"], node_info["type"], node_info["content"], sim))
        results.sort(key=lambda x: x[3], reverse=True)
        top_matches = results[:top_k]
        final_results = []
        with self.driver.session() as session:
            for node_id, node_type, node_content, sim in top_matches:
                context_chain = self._get_ancestor_chain(session, node_id)
                final_results.append({
                    "node_id": node_id,
                    "type": node_type,
                    "content": node_content,
                    "similarity": sim,
                    "ancestor_chain": context_chain
                })
        return final_results

    def _get_ancestor_chain(self, session, node_id):
        query = """
        MATCH path = (n:Chunk {id: $node_id})<-[:HAS_CHILD*0..]-(ancestor)
        RETURN nodes(path) as allNodes
        """
        records = session.run(query, node_id=node_id).data()
        ancestor_info = set()
        for rec in records:
            for node in rec["allNodes"]:
                ancestor_info.add((node["id"], node["type"], node["content"]))
        return list(ancestor_info)

    def retrieve_subgraph(self, start_node_id):
        with self.driver.session() as session:
            query = """
            MATCH path=(start:Chunk {id: $start_id})-[:HAS_CHILD*0..]->(descendant)
            RETURN nodes(path) as pathNodes
            """
            records = session.run(query, start_id=start_node_id).data()
        node_map = {}
        edges = set()
        for rec in records:
            path_nodes = rec["pathNodes"]
            for i, node_data in enumerate(path_nodes):
                node_id = node_data["id"]
                node_map[node_id] = {
                    "id": node_id,
                    "type": node_data["type"],
                    "content": node_data["content"]
                }
                if i < len(path_nodes) - 1:
                    src_id = path_nodes[i]["id"]
                    tgt_id = path_nodes[i+1]["id"]
                    edges.add((src_id, tgt_id))
        return {"nodes": list(node_map.values()), "edges": list(edges)}

def extract_context_for_query(query_text, retriever, top_k=3):
    """
    Retrieves the top matching nodes and aggregates their context.
    Returns a dictionary with:
      - 'chunks': a list of chunk objects (id, type, content, similarity)
      - 'ancestors': a mapping from chunk id to its ancestor chain.
    """
    results = retriever.retrieve_similar_nodes(query_text, top_k=top_k)
    output = {"chunks": [], "ancestors": {}}
    for res in results:
        chunk_obj = {
            "id": res["node_id"],
            "type": res["type"],
            "content": res["content"],
            "similarity": res["similarity"]
        }
        output["chunks"].append(chunk_obj)
        output["ancestors"][res["node_id"]] = res["ancestor_chain"]
    return output

# Declare global retriever variable
retriever = None

# --- FastAPI Application with Asynchronous Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    pdf_file = "data/POCSO Circular_2015.pdf"
    if not os.path.exists(pdf_file):
        print(f"File '{pdf_file}' not found! Initializing retriever with an empty graph.")
        retriever = GraphRetriever(
            model_name="all-MiniLM-L6-v2",
            neo4j_uri="bolt://neo4j:7687",
            neo4j_user="neo4j",
            neo4j_pass=NEO4J_PAASOWRD
        )
    else:
        print("Processing PDF and storing graph in Neo4j...")
        document_text = load_pdf_text(pdf_file)
        documents = [document_text]
        chunker = MultiscaleCellularAutomataGraphChunker(
            model_name="all-MiniLM-L6-v2",
            max_iters=1,  # quick test; adjust iterations as needed
            neo4j_uri="bolt://neo4j:7687",
            neo4j_user="neo4j",
            neo4j_pass=NEO4J_PAASOWRD
        )
        chunker.process_documents(documents)
        chunker.save_graph_to_neo4j(clear_db=True)
        retriever = GraphRetriever(
            model_name="all-MiniLM-L6-v2",
            neo4j_uri="bolt://neo4j:7687",
            neo4j_user="neo4j",
            neo4j_pass=NEO4J_PAASOWRD
        )
    yield
    # (Optional shutdown cleanup)

app = FastAPI(title="Constitution Context Agent", lifespan=lifespan)

@app.post("/context", response_model=QueryResponse)
async def get_context(query: QueryRequest):
    global retriever
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized.")
    try:
        context_output = extract_context_for_query(query.question, retriever, top_k=3)
        return QueryResponse(results=context_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
