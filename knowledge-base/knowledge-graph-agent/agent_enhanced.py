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
import hnswlib
from annoy import AnnoyIndex
from functools import lru_cache

# PyTorch Geometric imports for the GNN re-ranker
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# For offline mode if needed
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

print("Current working directory:", os.getcwd())

# Retrieve API keys from .env (OpenAI key is not used for relevance here)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_PAASOWRD = os.getenv("NEO4J_PASSWORD")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Set up device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize Hugging Face zero-shot pipeline for relevance scoring
hf_relevance_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device=="cuda" else -1)

# --- Pydantic Models ---
class sentencscore(BaseModel):
    score: float

class QueryRequest(BaseModel):
    question: str

# Return a structured dict with "chunks" and "ancestors"
class QueryResponse(BaseModel):
    results: dict

# --- Helper Functions for Adaptive Chunking ---
@lru_cache(maxsize=128)
def determine_optimal_chunk_size(text, target_chunks=10, min_chunk=5, max_chunk=30):
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return min_chunk
    optimal = n_words // target_chunks
    return max(min_chunk, min(optimal, max_chunk))

def split_into_chunks(text, optimal_size=None, overlap=3):
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
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Extracted {len(paragraphs)} paragraphs.")
    return paragraphs

def extract_sentences(text):
    sentences = nltk.sent_tokenize(text)
    print(f"Extracted {len(sentences)} sentences.")
    return sentences

def extract_named_entities(text):
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
    return extract_named_entities(sentence)

def embed_text(model, text_list, precision=6):
    if not text_list:
        return np.array([])
    embeddings = model.encode(text_list, batch_size=32, show_progress_bar=False)
    embeddings = np.array(embeddings)
    return np.round(embeddings, precision)

def adaptive_multiscale_grid(text_list):
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

# --- GNN Re-Ranker ---
class GNNReRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(GNNReRanker, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

def rerank_with_gnn(candidate_id, candidate_embedding, ancestor_chain):
    """
    For a given candidate (with its embedding and ancestor chain including embeddings),
    build a small PyG graph and run the GNN re-ranker.
    Returns a refined score.
    Note: This is a proof-of-concept; for real applications, the GNN should be pre-trained.
    """
    # Build node list: candidate and its ancestors that have embeddings
    nodes = []
    node_ids = []
    # Candidate node first
    nodes.append(candidate_embedding)
    node_ids.append(candidate_id)
    # For each ancestor in the chain, expect tuple: (id, type, content, embedding)
    for anc in ancestor_chain:
        # anc is expected to be a tuple (id, type, content, embedding)
        if len(anc) == 4 and anc[3] is not None:
            nodes.append(anc[3])
            node_ids.append(anc[0])
    if len(nodes) == 1:
        # No ancestor embeddings available, return original score
        return None
    x = torch.tensor(np.vstack(nodes), dtype=torch.float)
    # For simplicity, connect every node to every other (fully connected graph)
    num_nodes = x.shape[0]
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    input_dim = x.shape[1]
    gnn_model = GNNReRanker(input_dim)
    gnn_model.eval()
    with torch.no_grad():
        scores = gnn_model(x, edge_index)
    # Find refined score for candidate (first node)
    refined_score = scores[0].item()
    return refined_score

# --- Graph Chunking and Retrieval Classes ---
class MultiscaleCellularAutomataGraphChunker:
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
    Supports multiple indexing backends: "faiss", "hnswlib", or "annoy".
    Optionally uses a GNN re-ranker to refine candidate scores.
    Only retrieves nodes of type 'subchunk', 'sentence', or 'phrase' (fine-grained chunks).
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j", neo4j_pass="your_password", index_backend="faiss", use_gnn=False):
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.index_backend = index_backend.lower()
        self.use_gnn = use_gnn
        self.index = None
        self.id_map = []  # mapping from index to node info
        self._build_index()

    def _build_index(self):
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
        if not embeddings:
            print("No fine-grained nodes found to build index.")
            return
        embeddings = np.vstack(embeddings)
        d = embeddings.shape[1]
        if self.index_backend == "faiss":
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)
            print(f"Built FAISS index with {self.index.ntotal} vectors.")
        elif self.index_backend == "hnswlib":
            self.index = hnswlib.Index(space='cosine', dim=d)
            self.index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=16)
            self.index.add_items(embeddings)
            self.index.set_ef(50)
            print(f"Built hnswlib index with {self.index.get_current_count()} vectors.")
        elif self.index_backend == "annoy":
            self.index = AnnoyIndex(d, 'angular')
            for i, vec in enumerate(embeddings):
                self.index.add_item(i, vec)
            self.index.build(10)
            print(f"Built Annoy index with {len(embeddings)} vectors.")
        else:
            raise ValueError("Unsupported index_backend. Use 'faiss', 'hnswlib', or 'annoy'.")

    def _get_ancestor_chain(self, session, node_id):
        query = """
        MATCH path = (n:Chunk {id: $node_id})<-[:HAS_CHILD*0..]-(ancestor)
        RETURN nodes(path) as allNodes
        """
        records = session.run(query, node_id=node_id).data()
        ancestor_info = []
        for rec in records:
            for node in rec["allNodes"]:
                # Try to parse embedding if available
                emb = None
                if "embedding" in node and node["embedding"]:
                    try:
                        emb = np.array(json.loads(node["embedding"]), dtype='float32')
                    except:
                        pass
                ancestor_info.append((node["id"], node.get("type", ""), node.get("content", ""), emb))
        return ancestor_info

    # Optional GNN re-ranking function for a candidate.
    def rerank_with_gnn(self, candidate):
        """
        For a given candidate (tuple: (id, type, content, initial_sim)),
        retrieve its ancestor chain (with embeddings) and use a simple GNN to compute a refined score.
        Note: In production, the GNN should be pre-trained.
        """
        candidate_id, candidate_type, candidate_content, init_sim = candidate
        with self.driver.session() as session:
            ancestor_chain = self._get_ancestor_chain(session, candidate_id)
        # Build node list: candidate first, then ancestors with valid embeddings
        nodes = []
        node_ids = []
        candidate_emb = None
        # For candidate, look up its embedding in our id_map
        for info in self.id_map:
            if info["id"] == candidate_id:
                # For simplicity, re-embed candidate content
                candidate_emb = self.embedding_model.encode([info["content"]])[0].astype('float32')
                break
        if candidate_emb is None:
            return init_sim
        nodes.append(candidate_emb)
        node_ids.append(candidate_id)
        for anc in ancestor_chain:
            # anc is (id, type, content, emb)
            if anc[3] is not None:
                nodes.append(anc[3])
                node_ids.append(anc[0])
        if len(nodes) == 1:
            return init_sim
        x = torch.tensor(np.vstack(nodes), dtype=torch.float)
        num_nodes = x.shape[0]
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        input_dim = x.shape[1]
        gnn_model = GNNReRanker(input_dim)
        gnn_model.eval()
        with torch.no_grad():
            scores = gnn_model(x, edge_index)
        candidate_idx = node_ids.index(candidate_id)
        refined_score = scores[candidate_idx].item()
        print(f"Refined GNN score for candidate {candidate_id}: {refined_score}")
        return refined_score

    def retrieve_similar_nodes(self, query_text, top_k=3, sim_threshold=0.5):
        query_embedding = self.embedding_model.encode([query_text])[0].astype('float32')
        if self.index is None or self.index.ntotal == 0:
            return []
        if self.index_backend == "faiss":
            distances, indices = self.index.search(np.array([query_embedding]), top_k * 5)
            candidate_results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                sim = 1 - (dist / 2)
                if sim < sim_threshold:
                    continue
                candidate_results.append((self.id_map[idx]["id"], self.id_map[idx]["type"],
                                          self.id_map[idx]["content"], sim))
        elif self.index_backend == "hnswlib":
            labels, distances = self.index.knn_query(query_embedding, k=top_k * 5)
            candidate_results = []
            for idx, dist in zip(labels[0], distances[0]):
                sim = 1 - dist
                if sim < sim_threshold:
                    continue
                candidate_results.append((self.id_map[idx]["id"], self.id_map[idx]["type"],
                                          self.id_map[idx]["content"], sim))
        elif self.index_backend == "annoy":
            indices, dists = self.index.get_nns_by_vector(query_embedding.tolist(), top_k * 5, include_distances=True)
            candidate_results = []
            for idx, dist in zip(indices, dists):
                sim = 1 - (dist / 2)
                if sim < sim_threshold:
                    continue
                candidate_results.append((self.id_map[idx]["id"], self.id_map[idx]["type"],
                                          self.id_map[idx]["content"], sim))
        else:
            raise ValueError("Unsupported index_backend.")
        candidate_results.sort(key=lambda x: x[3], reverse=True)
        # Optionally re-rank candidates using GNN if enabled.
        if self.use_gnn:
            reranked = []
            for cand in candidate_results[:top_k]:
                refined = self.rerank_with_gnn(cand)
                reranked.append((cand[0], cand[1], cand[2], refined))
            reranked.sort(key=lambda x: x[3], reverse=True)
            top_matches = reranked[:top_k]
        else:
            top_matches = candidate_results[:top_k]
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
            neo4j_pass=NEO4J_PAASOWRD,
            index_backend="faiss",
            use_gnn=False  # Set to True to enable GNN re-ranking
        )
    else:
        print("Processing PDF and storing graph in Neo4j...")
        document_text = load_pdf_text(pdf_file)
        documents = [document_text]
        chunker = MultiscaleCellularAutomataGraphChunker(
            model_name="all-MiniLM-L6-v2",
            max_iters=1,
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
            neo4j_pass=NEO4J_PAASOWRD,
            index_backend="faiss",
            use_gnn=True  # Enable GNN re-ranking if desired
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
