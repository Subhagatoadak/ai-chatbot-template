
# 📘 Graph-Based Context-Aware Document Retriever

A modular, high-precision document retrieval framework using dynamic hierarchical chunking, Neo4j knowledge graphs, semantic embeddings, and hybrid search with support for FAISS, HNSWLib, and GNN-based re-ranking.

---

## ✨ Key Features

| Feature                              | Description                                                                                     |
|--------------------------------------|-------------------------------------------------------------------------------------------------|
| **Multiscale Chunking**              | Documents are chunked into subchunks, sentences, and phrases using adaptive token window sizes. |
| **Graph-Based Hierarchy**            | Stored in Neo4j with context-preserving parent-child relationships (document → chunk → phrase). |
| **Semantic Embedding**               | Embeddings generated using `sentence-transformers` (MiniLM, BERT, etc.)                         |
| **Hybrid Search**                    | Combines keyword-based filtering with semantic similarity via FAISS/HNSWlib/GNNs                |
| **Fast Similarity Retrieval**        | Support for FAISS (CPU) and HNSWLib (in-memory, ultra-fast), switchable at runtime              |
| **GNN-Based Re-Ranking**             | (Optional) Context-aware Graph Neural Network to re-rank results using document topology        |
| **Dynamic Index Update**             | Incremental updates to graph and search index as documents are added                           |
| **Customizable LLM-Free Relevance**  | Local relevance scores with HuggingFace zero-shot pipeline, no OpenAI dependency               |
| **Dockerized API with FastAPI**      | REST API for query-response and document ingestion                                              |

---

## ⚙️ Architecture Overview

```
PDF/Text Input
     ↓
Adaptive Chunker ──────────┐
     ↓                    │
Embedding → FAISS/HNSWlib │
     ↓                    ↓
Neo4j Graph ← Chunk Metadata, Parent Links
     ↓
Query → Embed + Search → TopK Subchunks
     ↓
Retrieve + Ancestor Chain → JSON Response
```

---

## 🔍 Search Modes

| Mode          | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `faiss`       | Efficient CPU-based flat index for small to mid-sized datasets              |
| `hnswlib`     | High-speed approximate search, scalable to millions of vectors              |
| `gnn`         | [WIP] GraphSAGE/GAT-based re-ranker to refine Top-K results using hierarchy |
| `hybrid`      | Combines TF-IDF/keyword match with semantic scores for high accuracy        |

---

## 🧠 Algorithms and Techniques

### 1. **Multiscale Cellular Automata Chunking**
- Uses adaptive token-based segmentation.
- Scores chunks by local similarity and relevance (zero-shot classifier).
- Applies multi-pass Cellular Automata rules to select meaningful subunits.

### 2. **Neo4j Graph Store**
- Hierarchical representation of document → context → subchunk → sentence → phrase.
- Preserves traceable lineage of every result for explainable AI.

### 3. **Vector Indexing**
- `FAISS`: L2/Inner product search with index type `IndexFlatIP`.
- `HNSWLib`: Memory-mapped HNSW index with 10x+ speedup on large graphs.
- Supports GPU acceleration (if `CUDA_VISIBLE_DEVICES` is set).

### 4. **Hybrid Semantic + Keyword Search**
- TF-IDF or keyword matching boosts semantic score.
- Enables relevance tuning in domain-specific scenarios (e.g., legal, policy docs).

### 5. **Dynamic Indexing**
- Chunk additions trigger embedding + Neo4j updates.
- Vector indexes updated in-memory or persisted depending on backend.

---

## 🏆 Why It’s Better Than Managed Vector DBs

| Benefit                         | Graph-Based Retriever                                | Managed Vector DBs                         |
|----------------------------------|------------------------------------------------------|---------------------------------------------|
| ✅ Full Graph Hierarchy          | Context traceability via parent-child relationships | ❌ Often flat vector chunks only            |
| ✅ Custom Relevance Tuning       | Plug-in scoring (LLM, ZSClassifier, TF-IDF, etc.)    | ❌ Limited to similarity + metadata filters |
| ✅ No Vendor Lock-In             | Local embeddings + storage, no cloud dependency      | ❌ Proprietary systems and pricing          |
| ✅ GNN and LLM-Compatible        | Future-proof for agent workflows and reasoning       | ❌ No support for GNNs or logic graphs      |
| ✅ Cost-Efficient for Scale      | Optimized FAISS/HNSW indexes                         | ❌ Per-query token or storage charges       |
| ✅ Incremental Indexing          | Add documents without re-ingesting everything        | ⚠️ Often requires full re-indexing          |

---

## 🔬 Future Research Directions

- [ ] **Graph Neural Networks for Reasoning** (GraphSAGE/GAT re-rankers)
- [ ] **Contrastive Retrieval Models** fine-tuned for chunk-level relevance.
- [ ] **LLM-based Rewriting or Answer Fusion** from top retrieved chunks.
- [ ] **Agentic Traversal** of graphs for multi-hop question answering.
- [ ] **Federated Graph Indexing** across multiple domains or users.

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Start API (Neo4j must be running on bolt://localhost:7687)
uvicorn main:app --reload

# POST Query
curl -X POST http://localhost:8000/context -H "Content-Type: application/json" \
  -d '{"question": "What steps are taken in schools under POCSO?"}'
```

---

## 📂 Folder Structure

```
.
├── main.py
├── models/
│   └── chunking.py, retriever.py, scorer.py
├── graph/
│   └── neo4j_store.py, gnn_ranker.py
├── data/
│   └── pocso_circular.pdf
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

We welcome contributions for:
- Custom chunking logic (e.g., syntax-based)
- Pluggable re-rankers (BGE, GNNs)
- Index sync and streaming ingestion

---

## 🛡 License

MIT License. Use it, modify it, enhance it.

---

Would you like me to:
- Split this into modular `main.py`, `retriever.py`, `scorer.py`, etc.?
- Add usage examples for each search mode (`/context?mode=faiss`, etc.)?
- Build a minimal Streamlit or Swagger interface?

Let me know how you'd like to structure the repo next 👇