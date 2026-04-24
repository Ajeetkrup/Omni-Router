from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
from src.semantic_router import SemanticRouter
from arq import create_pool
from arq.connections import RedisSettings
from contextlib import asynccontextmanager
from dotenv import load_dotenv 
load_dotenv()

# 1. Manage the ARQ Redis Pool Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    REDIS_URI = os.getenv("REDIS_URI")
    # Connect to the Redis queue on startup
    app.state.redis_pool = await create_pool(RedisSettings.from_dsn(REDIS_URI))
    yield
    # Close connection on shutdown
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

    return {
        "status": "success",
        "gateway_metrics": {
            "model_used": model_used,
            "latency_ms": latency_ms,
            "route_taken": route
        },
        "message": ai_message
    }
