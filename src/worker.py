import os
import time
import hashlib
from arq.connections import RedisSettings
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ToxicityMetric
from deepeval.models import LiteLLMModel
from dotenv import load_dotenv

try:
    import mlflow
except ImportError:
    mlflow = None

load_dotenv()

MLFLOW_ENABLED = False


def _initialize_mlflow() -> bool:
    if mlflow is None:
        print("MLflow not installed. Worker MLflow logging is disabled.")
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_EVAL", "omni-router-eval")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(
        f"MLflow eval tracking enabled at '{tracking_uri}' "
        f"(experiment: '{experiment_name}')"
    )
    return True


def _log_eval_run(
    prompt: str,
    route_type: str,
    model_name: str,
    judge_model_name: str,
    relevancy_score,
    toxicity_score,
    eval_duration_ms: float,
    status: str,
) -> None:
    if not MLFLOW_ENABLED:
        return

    try:
        with mlflow.start_run(run_name="eval_request"):
            mlflow.log_param("route_type", route_type)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("judge_model", judge_model_name)
            mlflow.log_param("status", status)
            mlflow.log_metric("eval_duration_ms", float(eval_duration_ms))
            mlflow.log_metric("prompt_chars", len(prompt))
            if relevancy_score is not None:
                mlflow.log_metric("relevancy_score", float(relevancy_score))
            if toxicity_score is not None:
                mlflow.log_metric("toxicity_score", float(toxicity_score))
                mlflow.log_metric(
                    "toxicity_alert",
                    1.0 if float(toxicity_score) > 0.1 else 0.0,
                )
            mlflow.set_tag("component", "worker")
            mlflow.set_tag(
                "prompt_hash",
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
            )
    except Exception as e:
        print(f"MLflow worker logging failed: {e}")


class LenientLiteLLMModel(LiteLLMModel):
    """
    Avoid provider-enforced JSON tool schema validation by requesting plain output.
    DeepEval will still parse JSON from the returned text.
    """

    def generate_with_schema(self, *args, schema=None, **kwargs):
        return self.generate(*args, schema=None, **kwargs)

    async def a_generate_with_schema(self, *args, schema=None, **kwargs):
        return await self.a_generate(*args, schema=None, **kwargs)


def _build_judge_model() -> LiteLLMModel:
    """
    Build the LLM-as-a-judge model for DeepEval.
    Defaults to Groq via LiteLLM, but can be overridden with env vars.
    """
    judge_model_name = os.getenv(
        "DEEPEVAL_JUDGE_MODEL", "groq/llama-3.1-8b-instant"
    )
    judge_api_key = os.getenv("GROQ_API_KEY")
    judge_api_base = os.getenv("DEEPEVAL_JUDGE_API_BASE")

    kwargs = {}
    if judge_api_key:
        kwargs["api_key"] = judge_api_key
    if judge_api_base:
        kwargs["base_url"] = judge_api_base

    return LenientLiteLLMModel(model=judge_model_name, **kwargs)

# 1. Define the asynchronous evaluation task
async def run_evaluation(ctx, prompt: str, actual_answer: str, route_type: str, model_name: str):
    print(f"\n[EVAL WORKER] Starting DeepEval for route: {route_type} ({model_name})")
    start_time = time.time()
    
    # Initialize your metrics (you can customize thresholds here)
    judge_model = _build_judge_model()
    judge_model_name = judge_model.name
    relevancy = AnswerRelevancyMetric(threshold=0.8, model=judge_model)
    toxicity = ToxicityMetric(threshold=0.1, model=judge_model)

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
    except Exception as e:
        relevancy = None
        print(f"[EVAL WARN] Relevancy metric failed: {e}")

    try:
        await toxicity.a_measure(test_case)
    except Exception as e:
        toxicity = None
        print(f"[EVAL WARN] Toxicity metric failed: {e}")

    if relevancy is None and toxicity is None:
        print("[EVAL ERROR] All metrics failed.")
        _log_eval_run(
            prompt=prompt,
            route_type=route_type,
            model_name=model_name,
            judge_model_name=judge_model_name,
            relevancy_score=None,
            toxicity_score=None,
            eval_duration_ms=round((time.time() - start_time) * 1000, 2),
            status="all_metrics_failed",
        )
        return

    relevancy_score = f"{relevancy.score:.2f}" if relevancy is not None else "N/A"
    toxicity_score = f"{toxicity.score:.2f}" if toxicity is not None else "N/A"
    print(
        f"[EVAL COMPLETE] Relevancy: {relevancy_score} | Toxicity: {toxicity_score}"
    )

    # 3. Handle Business Logic (e.g., Log to DB or MLflow)
    if toxicity is not None and toxicity.score > 0.1:
        print(f"⚠️ ALERT: Toxic response detected from {model_name}!")
        # In production: trigger Slack alert or log to MLflow here
    _log_eval_run(
        prompt=prompt,
        route_type=route_type,
        model_name=model_name,
        judge_model_name=judge_model_name,
        relevancy_score=relevancy.score if relevancy is not None else None,
        toxicity_score=toxicity.score if toxicity is not None else None,
        eval_duration_ms=round((time.time() - start_time) * 1000, 2),
        status="success",
    )

# 4. Configure the ARQ Worker
class WorkerSettings:
    # Register the function so ARQ knows what to execute
    functions = [run_evaluation]
    
    # Default Redis settings connect to localhost:6379
    # If using a cloud Redis, pass url="redis://..." here
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URI"))


MLFLOW_ENABLED = _initialize_mlflow()