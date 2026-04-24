import os
from arq.connections import RedisSettings
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ToxicityMetric

# 1. Define the asynchronous evaluation task
async def run_evaluation(ctx, prompt: str, actual_answer: str, route_type: str, model_name: str):
    print(f"\n[EVAL WORKER] Starting DeepEval for route: {route_type} ({model_name})")
    
    # Initialize your metrics (you can customize thresholds here)
    relevancy = AnswerRelevancyMetric(threshold=0.8)
    toxicity = ToxicityMetric(threshold=0.1)

    # Package the interaction into a DeepEval test case
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_answer,
        additional_metadata={
            "route_type": route_type,
            "model_name": model_name
        }
    )

    try:
        # 2. Run the measurements asynchronously
        # a_measure ensures the worker isn't blocked by network calls to the evaluator LLM
        await relevancy.a_measure(test_case)
        await toxicity.a_measure(test_case)

        print(f"[EVAL COMPLETE] Relevancy: {relevancy.score:.2f} | Toxicity: {toxicity.score:.2f}")

        # 3. Handle Business Logic (e.g., Log to DB or MLflow)
        if toxicity.score > 0.1:
            print(f"⚠️ ALERT: Toxic response detected from {model_name}!")
            # In production: trigger Slack alert or log to MLflow here
            
    except Exception as e:
        print(f"[EVAL ERROR] Failed to evaluate prompt: {e}")

# 4. Configure the ARQ Worker
class WorkerSettings:
    # Register the function so ARQ knows what to execute
    functions = [run_evaluation]
    
    # Default Redis settings connect to localhost:6379
    # If using a cloud Redis, pass url="redis://..." here
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URI"))