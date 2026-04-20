"""
Omni-Analyst Ingestion DAG

Runs daily to ingest new ArXiv papers (cs.AI, cs.LG, cs.CL) and enrich them
with Semantic Scholar citation data, then loads them into ChromaDB and Neo4j.
"""

from datetime import datetime, timedelta

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator

INGESTION_SERVICE_URL = "http://ingestion-service:8001"

DEFAULT_ARGS = {
    "owner": "omni-analyst",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR", "stat.ML"]
MAX_PAPERS = 100
DAYS_BACK = 2


def trigger_ingestion_sync(**context) -> dict:
    """Call the ingestion-service synchronous endpoint."""
    payload = {
        "categories": CATEGORIES,
        "max_papers": MAX_PAPERS,
        "days_back": DAYS_BACK,
    }
    with httpx.Client(timeout=300) as client:
        resp = client.post(f"{INGESTION_SERVICE_URL}/ingest/trigger/sync", json=payload)
        resp.raise_for_status()
        result = resp.json()

    print(f"Ingestion result: {result}")
    context["task_instance"].xcom_push(key="papers_processed", value=result["papers_processed"])
    return result


def verify_ingestion(**context) -> None:
    """Verify health of ingestion service after pipeline run."""
    papers_count = context["task_instance"].xcom_pull(
        task_ids="run_ingestion_pipeline", key="papers_processed"
    )
    print(f"Verification: {papers_count} papers were processed")

    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{INGESTION_SERVICE_URL}/health")
        resp.raise_for_status()
        print(f"Ingestion service health: {resp.json()}")


with DAG(
    dag_id="omni_analyst_ingestion",
    default_args=DEFAULT_ARGS,
    description="Daily ingestion of ArXiv papers into ChromaDB and Neo4j",
    schedule_interval="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["omni-analyst", "ingestion", "arxiv"],
) as dag:

    run_pipeline = PythonOperator(
        task_id="run_ingestion_pipeline",
        python_callable=trigger_ingestion_sync,
    )

    verify = PythonOperator(
        task_id="verify_ingestion",
        python_callable=verify_ingestion,
    )

    run_pipeline >> verify
