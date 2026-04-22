import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.responses import Response

from evaluator import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_RUNS = Counter("evaluation_runs_total", "Total evaluation runs", ["status"])
EVAL_FAITHFULNESS = Gauge("eval_faithfulness_score", "Latest faithfulness score")
EVAL_RELEVANCY = Gauge("eval_answer_relevancy_score", "Latest answer relevancy score")

_evaluation_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Evaluation service started")
    yield


app = FastAPI(title="Omni-Analyst Evaluation Service", version="1.0.0", lifespan=lifespan)


class EvalRequest(BaseModel):
    max_questions: int = 10


class EvalResponse(BaseModel):
    status: str
    message: str


@app.post("/evaluate/run", response_model=EvalResponse)
async def trigger_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    global _evaluation_running
    if _evaluation_running:
        return EvalResponse(status="running", message="Evaluation already in progress")

    def run_and_update():
        global _evaluation_running
        _evaluation_running = True
        try:
            result = run_evaluation(max_questions=request.max_questions)
            scores = result.get("scores", {})
            if scores:
                EVAL_FAITHFULNESS.set(scores.get("faithfulness", 0))
                EVAL_RELEVANCY.set(scores.get("answer_relevancy", 0))
            EVAL_RUNS.labels(status="success").inc()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            EVAL_RUNS.labels(status="error").inc()
        finally:
            _evaluation_running = False

    background_tasks.add_task(run_and_update)
    return EvalResponse(status="accepted", message=f"Evaluation started for {request.max_questions} questions")


@app.get("/evaluate/results")
async def get_results():
    results_path = Path("/app/eval_results.json")
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results yet. Run POST /evaluate/run first.")
    import json
    with open(results_path) as f:
        return json.load(f)


@app.get("/evaluate/status")
async def get_status():
    return {"running": _evaluation_running}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "evaluation-service"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
