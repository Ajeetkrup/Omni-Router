# Omni-Router

Omni-Router is a blazing-fast, enterprise-grade AI Gateway and Semantic Router designed to optimize LLM interactions. It intelligently routes user prompts, caches responses semantically to avoid redundant LLM calls, and dynamically selects the most efficient underlying LLM based on task complexity.

## 🚀 Key Features

*   **Ultra-Fast Local Embeddings**: Utilizes an optimized ONNX Runtime and pure NumPy pipeline (for mean pooling and L2 normalization) to generate embeddings locally, completely avoiding PyTorch tensor overhead.
*   **Semantic Caching Layer**: Employs FAISS (Facebook AI Similarity Search) and Redis to detect semantically similar prompts. If a match is found (above a configurable threshold), it serves the cached response instantly.
*   **Intelligent Dynamic Routing**: When a cache miss occurs, the router evaluates the complexity of the prompt. Simple tasks are routed to ultra-fast models (`Llama-3-8B`), while complex reasoning tasks (e.g., coding, architecting) are routed to capable frontier models (`Qwen-3-32B`).
*   **Groq & LiteLLM Integration**: Seamlessly leverages Groq's LPUs for lightning-fast inference on fallback LLM calls via the standard LiteLLM interface.
*   **Detailed Telemetry**: Built-in granular latency tracking for embedding generation, FAISS search times, LLM inference times, and cache update latency.

## 🧠 Architecture Overview

```mermaid
graph TD
    Client[Client Request] --> FastAPI[FastAPI Gateway \n/v1/chat/completions]
    FastAPI --> Router[Semantic Router]
    
    subgraph Semantic Layer
        Router --> Embedding[Local ONNX Embedding Model \n+ NumPy Pooling]
        Embedding --> FAISS[(FAISS Vector Index)]
    end
    
    FAISS -- "Hit (Similarity > 0.95)" --> Redis[(Redis Cache)]
    Redis --> Response[Return Cached Response]
    
    FAISS -- "Miss" --> ComplexityLogic{Complexity Analysis}
    
    subgraph Fallback LLMs (via Groq & LiteLLM)
        ComplexityLogic -- "Simple Task" --> Llama3[Llama-3-8B]
        ComplexityLogic -- "Complex Task \n(>300 chars or keywords)" --> Qwen3[Qwen-3-32B]
    end
    
    Llama3 --> CacheUpdate[Update FAISS & Redis]
    Qwen3 --> CacheUpdate
    CacheUpdate --> Response
```

### Components
1.  **FastAPI Gateway (`gateway.py`)**: Provides a clean, RESTful endpoint mirroring standard chat completion APIs, while injecting custom telemetry metadata (latency, route taken).
2.  **Semantic Router (`semantic_router.py`)**: 
    *   **ONNX Encoder**: Runs the MiniLM-v2 (or similar) feature extraction model using `CPUExecutionProvider` with tuned intra/inter-op threads.
    *   **FAISS Index**: Stores normalized float32 embedding vectors for high-speed nearest-neighbor lookups (`IndexFlatIP`).
    *   **Redis KV Store**: Backs the semantic cache by mapping prompts/IDs to their corresponding string responses.
3.  **LLM Callers**: Uses `litellm` to call external APIs dynamically based on routing decisions.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- Redis server running locally or accessible remotely.

### Environment Variables
Create a `.env` file in the root directory:
```env
# Example .env file
REDIS_URI="redis://localhost:6379/0"
GROQ_API_KEY="your-groq-api-key"
```

### Running the Server
The gateway uses `uvicorn` to serve the FastAPI application.

```bash
uvicorn src.gateway:app --reload --port 8002
```

## 📊 Example Response

```json
{
  "status": "success",
  "gateway_metrics": {
    "model_used": "Qwen-3-32B",
    "latency_ms": 450.2,
    "route_taken": "llm route"
  },
  "message": "Here is the architectural breakdown you requested..."
}
```

When a subsequent semantically similar request is sent, the `route_taken` will change to `"cache hit"`, and `latency_ms` will drop drastically.
