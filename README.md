
# 📚 Neo4j-Based Knowledge Graph with Cellular Automata Chunking

This project builds a **context-aware document understanding system** using a novel **Cellular Automata (CA) chunker**, a **graph-based retriever**, and a **Neo4j-powered knowledge graph**. The core idea is to enable multi-resolution chunking and embedding of documents into a semantically rich and queryable graph.

---

## 🔬 Method Overview

### 📄 1. PDF Ingestion and Preprocessing
Documents (e.g., `constitution.pdf`) are loaded and processed into paragraphs, sentences, and phrases using **NLTK**. Instead of using spaCy (which has bulky dependencies), this version uses named entity extraction and POS tagging with pure NLTK.

### ⚙️ 2. Cellular Automata Chunker
The **Multiscale Cellular Automata (CA)** mechanism categorizes text chunks as `short`, `medium`, or `long` based on their token lengths and applies iterative context-aware updates using:
- Cosine similarity of embeddings
- LLM-based relevance scores (OpenAI GPT-4o)

This dynamic chunking simulates "activations" in a local neighborhood, mimicking how meaning builds across scales in natural language.

### 🧠 3. Knowledge Graph Construction
The resulting hierarchy is:
```
Document → Context → Paragraph → Sentence → Phrase
```
Each node is embedded and stored as a `Chunk` in Neo4j with parent-child relationships.

### 🔍 4. Graph Retriever
A `GraphRetriever` allows semantic search by embedding queries and comparing them to all stored chunks. It also reconstructs ancestor chains to provide contextual explanations.

---

## 🚀 Usage

Run the app using Docker:

```bash
docker-compose build
docker-compose up
```

Then hit the FastAPI endpoint:

```bash
POST /context
{
  "question": "What are the duties of the President?"
}
```

---

## 🧠 Uniqueness & Advantages

- ✅ **Multiscale CA-Based Chunking**: Combines local cosine similarity with global context awareness via LLM scoring.
- ✅ **Graph-Based Semantics**: Hierarchical graph structure enables both retrieval and reasoning.
- ✅ **Flexible Preprocessing**: Pure NLTK-based chunker avoids heavyweight NLP dependencies like spaCy.

---

## ⚠️ Limitations

> ⚡ **Data Preparation is Currently Slow**

Yes — the graph construction step is noticeably slow, especially on large documents. This is primarily due to:
- LLM scoring for every phrase-context window
- Multiprocessing + I/O latency
- Graph serialization to Neo4j

**⚙️ Work is actively ongoing to make this more efficient**, including:
- Memoization of LLM scores
- Batch embedding optimizations
- Lazy graph writes

---

## 📁 agent.py Review & USP

In reviewing `agent.py`, we identified the following:

### ✅ Unique Selling Proposition (USP)
- Seamless retrieval of fine-grained, multi-level context for any input query
- Simple plug-and-play design for swapping in new documents
- Potential for cross-document reasoning by linking multiple context trees

### ❌ Current Gaps
- No ranking logic for ancestor chain relevance
- Query-time performance may degrade with large graphs
- Lack of streaming context generation for large answer chains

We’re actively working to plug these gaps with a new version of `agent.py` that supports:
- Context prioritization
- Query cache with embeddings
- Streaming token responses

---

## 🧪 In Development
- [ ] Streaming response support with token buffering
- [ ] Multiple-document context merging
- [ ] Graph pruning for performance
- [ ] Visual dashboard for Neo4j graphs

---

## 🤝 Contributing
If you’re interested in improving the CA chunker, optimizing Neo4j interactions, or enhancing retrieval — feel free to open an issue or submit a PR!

---

## 📘 License
MIT License — open for commercial and academic use.

---

Let me know if you'd like this turned into a live GitHub repo structure or if you want badges, architecture diagrams, or deployment guides added!