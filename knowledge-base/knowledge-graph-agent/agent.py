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

# Ensure nltk data is available
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("maxent_ne_chunker")
nltk.download('maxent_ne_chunker_tab')
nltk.download("words")

load_dotenv()

# Retrieve API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_PAASOWRD = os.getenv("NEO4J_PASSWORD")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Pydantic Models ---
class sentencscore(BaseModel):
    score: float

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    results: str

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
    Splits the text into paragraphs. Adjust the delimiter if necessary.
    """
    # If your PDF uses single newlines, you might use a different splitting strategy.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Extracted {len(paragraphs)} paragraphs.")
    return paragraphs

def extract_sentences(text):
    """
    Uses NLTK's sentence tokenizer to extract sentences.
    """
    sentences = nltk.sent_tokenize(text)
    # Debug: Print number of sentences
    print(f"Extracted {len(sentences)} sentences.")
    return sentences

def extract_named_entities(text):
    """
    Uses NLTK's ne_chunk to extract named entities from text.
    Returns a list of candidate phrases.
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
    # Debug: Print number of entities found in this text snippet
    print(f"Extracted {len(entities)} named entities.")
    return entities

def extract_phrases_from_sentence(sentence):
    """
    Extracts candidate phrases (named entities) from a single sentence.
    """
    return extract_named_entities(sentence)

def embed_text(model, text_list, precision=6):
    """
    Generate embeddings for a list of text items with controlled precision.
    """
    embeddings = np.array(model.encode(text_list))
    return np.round(embeddings, precision)

def adaptive_multiscale_grid(text_list):
    """
    Dynamically assigns items to 'short', 'medium', and 'long' categories
    based on percentile thresholds of token lengths.
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

def get_llm_context_score(phrase=None, context=None):
    """
    Uses OpenAI GPT to compute a similarity score (0 to 1) between a phrase and its context.
    Returns a float score. If parsing fails, defaults to 0.5.
    """
    prompt = (
        "Analyze the relevance of the given phrase within its provided context.\n"
        "Return a JSON object containing a single key 'score' with a float value between 0 and 1.\n"
        "Example output: {\"score\": 0.75}\n\n"
        f"Phrase: {phrase}\nContext: {context}"
    )
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            response_format=sentencscore,
            temperature=0
        )
        response_content = response.choices[0].message.parsed
        print("LLM score response:", response_content)
        return response_content.score
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return 0.5

def update_ca(scale, grid, items, embeddings):
    """
    Runs Cellular Automata updates for a single scale.
    """
    new_grid = grid[scale].copy()
    for i in range(1, len(grid[scale]) - 1):
        left, center, right = embeddings[i - 1], embeddings[i], embeddings[i + 1]
        phrase = items[i]
        context = " | ".join([items[i - 1], items[i], items[i + 1]])
        sim_left = 1 - cosine(left, center)
        sim_right = 1 - cosine(right, center)
        llm_score = get_llm_context_score(phrase, context)
        final_score_left = (1 - 0.3) * sim_left + 0.3 * llm_score
        final_score_right = (1 - 0.3) * sim_right + 0.3 * llm_score
        if grid[scale][i] == 0 and (final_score_left > 0.7 or final_score_right > 0.7):
            new_grid[i] = 1
        elif grid[scale][i] == 1 and (final_score_left < 0.4 and final_score_right < 0.4):
            new_grid[i] = 0
    return scale, new_grid

# --- Graph Chunking and Retrieval Classes ---
class MultiscaleCellularAutomataGraphChunker:
    """
    Pipeline for processing a document and building a hierarchical graph.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", max_iters=2, 
                 neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", 
                 neo4j_pass="your_password"):
        self.embedding_model = SentenceTransformer(model_name)
        self.max_iters = max_iters
        self.graph = nx.DiGraph()
        np.random.seed(42)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_pass = neo4j_pass

    def process_documents(self, documents):
        """
        Process each document into a hierarchy:
        Document -> Context -> Paragraph -> Sentence -> Phrase
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
            paragraphs_embeddings = embed_text(self.embedding_model, paragraphs)
            all_chunks = []
            all_embeddings = []
            for p_idx, paragraph_text in enumerate(paragraphs):
                para_node_id = f"Doc{doc_idx}_Para_{p_idx}"
                para_emb = paragraphs_embeddings[p_idx]
                self.graph.add_node(para_node_id, type="paragraph", content=paragraph_text, embedding=para_emb.tolist())
                self.graph.add_edge(context_node_id, para_node_id)
                all_chunks.append(paragraph_text)
                all_embeddings.append(para_emb)
                # NLTK-based sentence segmentation
                sentences = extract_sentences(paragraph_text)
                print(f"Paragraph {p_idx} has {len(sentences)} sentences.")
                sent_embeddings = embed_text(self.embedding_model, sentences)
                for s_idx, sentence_text in enumerate(sentences):
                    sent_node_id = f"Doc{doc_idx}_Para_{p_idx}_Sent_{s_idx}"
                    sent_emb = sent_embeddings[s_idx]
                    self.graph.add_node(sent_node_id, type="sentence", content=sentence_text, embedding=sent_emb.tolist())
                    self.graph.add_edge(para_node_id, sent_node_id)
                    all_chunks.append(sentence_text)
                    all_embeddings.append(sent_emb)
                    # NLTK-based NE extraction for phrases
                    phrases = extract_phrases_from_sentence(sentence_text)
                    print(f"Sentence {s_idx} has {len(phrases)} phrases (named entities).")
                    phrase_embeddings = embed_text(self.embedding_model, phrases)
                    for ph_idx, phrase_text in enumerate(phrases):
                        phrase_node_id = f"Doc{doc_idx}_Para_{p_idx}_Sent_{s_idx}_Ph_{ph_idx}"
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
        Pushes the built graph to Neo4j.
        """
        graph_data = nx.node_link_data(self.graph)
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_pass))
        with driver.session() as session:
            if clear_db:
                session.run("MATCH (n) DETACH DELETE n")
                print("Cleared existing Neo4j data.")
            for node in graph_data["nodes"]:
                node_id = node["id"]
                node_type = node.get("type", "")
                content = node.get("content", "")
                embedding_list = node.get("embedding", [])
                embedding_str = json.dumps(embedding_list)
                session.run(
                    """
                    MERGE (n:Chunk {id: $id})
                    SET n.type = $type,
                        n.content = $content,
                        n.embedding = $embedding
                    """,
                    id=node_id,
                    type=node_type,
                    content=content,
                    embedding=embedding_str
                )
            for link in graph_data["links"]:
                source = link["source"]
                target = link["target"]
                session.run(
                    """
                    MATCH (a:Chunk {id: $source})
                    MATCH (b:Chunk {id: $target})
                    MERGE (a)-[:HAS_CHILD]->(b)
                    """,
                    source=source,
                    target=target,
                )
        print("Graph stored in Neo4j.")
        # Debug: Query and print count of inserted nodes
        with driver.session() as session:
            result = session.run("MATCH (n:Chunk) RETURN count(n) AS count")
            count = result.single()["count"]
            print("Number of nodes in Neo4j:", count)

class GraphRetriever:
    """
    Connects to Neo4j and enables vector-based similarity search and context retrieval.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j", neo4j_pass="your_password"):
        self.embedding_model = SentenceTransformer(model_name)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    def retrieve_similar_nodes(self, query_text, top_k=3):
        """
        Embeds the query, computes cosine similarities with all nodes, and returns the top matches.
        """
        query_embedding = self._embed_query(query_text)
        with self.driver.session() as session:
            result = session.run("MATCH (n:Chunk) RETURN n.id as id, n.type as type, n.content as content, n.embedding as embedding")
            records = result.data()
        similarities = []
        for rec in records:
            node_id = rec["id"]
            node_type = rec["type"]
            node_content = rec["content"]
            emb_data = rec["embedding"]
            if not emb_data:
                continue
            emb_vec = np.array(json.loads(emb_data))
            sim = 1 - cosine(query_embedding, emb_vec)
            similarities.append((node_id, node_type, node_content, sim))
        similarities.sort(key=lambda x: x[3], reverse=True)
        top_matches = similarities[:top_k]
        results_with_context = []
        with self.driver.session() as session:
            for node_id, node_type, node_content, sim in top_matches:
                context_chain = self._get_ancestor_chain(session, node_id)
                results_with_context.append({
                    "node_id": node_id,
                    "type": node_type,
                    "content": node_content,
                    "similarity": sim,
                    "ancestor_chain": context_chain
                })
        return results_with_context

    def _embed_query(self, query_text):
        return self.embedding_model.encode([query_text])[0]

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
    Given a query, retrieves the top matching nodes and returns an aggregated context.
    """
    results = retriever.retrieve_similar_nodes(query_text, top_k=top_k)
    context = ""
    for res in results:
        context += f"NodeID: {res['node_id']} (Type: {res['type']})\n"
        context += f"Content: {res['content']}\n"
        context += "Ancestor Chain:\n"
        for anc in res["ancestor_chain"]:
            context += f"  - {anc}\n"
        context += "\n"
    return context

# Declare global retriever variable
retriever = None

# --- FastAPI Application with Asynchronous Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    pdf_file = "data/posco.pdf"
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
        context_text = extract_context_for_query(query.question, retriever, top_k=1)
        return QueryResponse(results=context_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
