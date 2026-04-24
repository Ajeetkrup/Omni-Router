from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import hashlib
from src.semantic_router import SemanticRouter
from arq import create_pool
from arq.connections import RedisSettings
from contextlib import asynccontextmanager
from dotenv import load_dotenv 
load_dotenv()

try:
    import mlflow
except ImportError:
    mlflow = None

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

MLFLOW_ENABLED = False


def _initialize_mlflow() -> bool:
    if mlflow is None:
        print("MLflow not installed. Gateway MLflow logging is disabled.")
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_GATEWAY", "omni-router-gateway")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(
        f"MLflow gateway tracking enabled at '{tracking_uri}' "
        f"(experiment: '{experiment_name}')"
    )
    return True


def _log_gateway_run(
    user_id: str,
    prompt: str,
    model_used: str,
    route: str,
    latency_ms: float,
) -> None:
    if not MLFLOW_ENABLED:
        return

    try:
        with mlflow.start_run(run_name="gateway_request"):
            mlflow.log_param("model_used", model_used)
            mlflow.log_param("route_taken", route)
            mlflow.log_metric("latency_ms", float(latency_ms))
            mlflow.log_metric("prompt_chars", len(prompt))
            mlflow.set_tag("component", "gateway")
            mlflow.set_tag(
                "user_hash",
                hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12],
            )
    except Exception as e:
        print(f"MLflow gateway logging failed: {e}")

# 1. Manage the ARQ Redis Pool Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Get the URI
    REDIS_URI = os.getenv("REDIS_URI")
    
    # 2. Parse the base settings from your URI
    settings = RedisSettings.from_dsn(REDIS_URI)
    
    # 3. Fix the timeout using arq's native attributes
    settings.conn_timeout = 5.0       # Increase timeout from 1s to 5s
    settings.conn_retries = 5         # Number of times to retry connecting
    settings.conn_retry_delay = 1.0   # Wait 1 second between retries
    
    # 4. Connect
    app.state.redis_pool = await create_pool(settings)
    global MLFLOW_ENABLED
    MLFLOW_ENABLED = _initialize_mlflow()
    print("Successfully connected to Redis pool!")
    
    yield
    
    # 5. Shutdown cleanly
    if hasattr(app.state, "redis_pool"):
        await app.state.redis_pool.close()

app = FastAPI(title="Omni-Route Enterprise LLM Gateway", lifespan=lifespan)

# --- 1. Define the User Request ---
class UserRequest(BaseModel):
    user_id: str
    prompt: str

sem_router = SemanticRouter()

# --- 3. The Gateway Endpoint with LiteLLM ---
@app.post("/v1/chat/completions")
async def handle_chat_request(request: UserRequest):
    start_time = time.time()
    print(f"--> Incoming request from {request.user_id}: '{request.prompt}'")

    try:
        ai_message, route, model_used = sem_router.route_request(request.prompt)

        await app.state.redis_pool.enqueue_job(
            'run_evaluation',    
            request.prompt, 
            ai_message, 
            route, 
            model_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM routing failed: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)
    print(f"--> Request completed in {latency_ms}ms")
    _log_gateway_run(
        request.user_id,
        request.prompt,
        model_used,
        route,
        latency_ms,
    )

    return {
        "status": "success",
        "gateway_metrics": {
            "model_used": model_used,
            "latency_ms": latency_ms,
            "route_taken": route
        },
        "message": ai_message
    }
