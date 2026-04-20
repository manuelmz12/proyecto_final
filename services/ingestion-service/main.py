import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel

from pipeline.arxiv_loader import fetch_arxiv_papers
from pipeline.semantic_scholar import enrich_with_citations
from pipeline.embedder import generate_embeddings
from pipeline.vector_store import upsert_to_chroma
from pipeline.graph_store import upsert_to_neo4j
from pipeline.bm25_index import update_bm25_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INGESTIONS_TOTAL = Counter("ingestion_total", "Total ingestion runs", ["status"])
INGESTION_DURATION = Histogram("ingestion_duration_seconds", "Ingestion pipeline duration")
PAPERS_INGESTED = Counter("papers_ingested_total", "Total papers ingested")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Ingestion service started")
    yield
    logger.info("Ingestion service stopped")


app = FastAPI(title="Omni-Analyst Ingestion Service", version="1.0.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    categories: list[str] = ["cs.AI", "cs.LG", "cs.CL"]
    max_papers: int = 50
    days_back: int = 7


class IngestResponse(BaseModel):
    status: str
    papers_processed: int
    message: str


async def run_ingestion_pipeline(categories: list[str], max_papers: int, days_back: int) -> dict:
    with INGESTION_DURATION.time():
        try:
            logger.info(f"Starting ingestion: categories={categories}, max={max_papers}")

            papers = await fetch_arxiv_papers(categories, max_papers, days_back)
            if not papers:
                return {"papers_processed": 0, "status": "no_data"}

            papers = await enrich_with_citations(papers)
            papers_with_embeddings = await generate_embeddings(papers)
            await upsert_to_chroma(papers_with_embeddings)
            await upsert_to_neo4j(papers)
            update_bm25_index(papers)

            PAPERS_INGESTED.inc(len(papers))
            INGESTIONS_TOTAL.labels(status="success").inc()
            logger.info(f"Ingestion complete: {len(papers)} papers processed")
            return {"papers_processed": len(papers), "status": "success"}
        except Exception as e:
            INGESTIONS_TOTAL.labels(status="error").inc()
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise


@app.post("/ingest/trigger", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        asyncio.create_task,
        run_ingestion_pipeline(request.categories, request.max_papers, request.days_back),
    )
    return IngestResponse(
        status="accepted",
        papers_processed=0,
        message=f"Ingestion triggered for categories: {request.categories}",
    )


@app.post("/ingest/trigger/sync", response_model=IngestResponse)
async def trigger_ingestion_sync(request: IngestRequest):
    """Synchronous version for Airflow DAG triggers."""
    result = await run_ingestion_pipeline(request.categories, request.max_papers, request.days_back)
    return IngestResponse(
        status=result["status"],
        papers_processed=result["papers_processed"],
        message=f"Ingestion complete: {result['papers_processed']} papers",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ingestion-service"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
