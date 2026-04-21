import json
import logging
import os
from datetime import datetime, timezone

import httpx
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

CHAT_SERVICE_URL = os.environ.get("CHAT_SERVICE_URL", "http://chat-service:8002")
RESULTS_PATH = "/app/eval_results.json"
TEST_QUESTIONS_PATH = "/app/test_questions.json"

# Use guest token for evaluation calls
EVAL_USERNAME = "demo"
EVAL_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo123")


class DirectAzureEmbeddings(Embeddings):
    """Wraps openai.AzureOpenAI to avoid LangChain pre-tokenization."""

    def __init__(self):
        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
        self._model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS", "devlab-text-embedding-3-large")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_CHAT", "devlab-gpt-4o-mini"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0,
    )


def _get_embeddings() -> DirectAzureEmbeddings:
    return DirectAzureEmbeddings()


def _get_auth_token(client: httpx.Client) -> str:
    return f"{EVAL_USERNAME}:{EVAL_PASSWORD}"


def _query_chat_service(
    client: httpx.Client, question: str, token: str, session_id: str
) -> dict:
    resp = client.post(
        f"{CHAT_SERVICE_URL}/chat",
        json={"message": question, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def run_evaluation(max_questions: int = 10) -> dict:
    """
    Run RAGAS evaluation against the chat service.

    Returns a dict with metrics and per-question results.
    """
    with open(TEST_QUESTIONS_PATH) as f:
        test_data = json.load(f)

    test_data = test_data[:max_questions]
    logger.info(f"Running evaluation on {len(test_data)} questions")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    with httpx.Client(timeout=120) as client:
        token = _get_auth_token(client)

        for i, item in enumerate(test_data):
            session_id = f"eval_session_{i}"
            try:
                response = _query_chat_service(client, item["question"], token, session_id)
                questions.append(item["question"])
                answers.append(response["answer"])
                contexts.append(
                    [s.get("title", "") + ": " + s.get("url", "") for s in response.get("sources", [])]
                    or ["No context retrieved"]
                )
                ground_truths.append(item["ground_truth"])
                logger.info(f"Evaluated Q{i+1}/{len(test_data)}: {item['question'][:50]}...")
            except Exception as e:
                logger.warning(f"Failed to evaluate Q{i+1}: {e}")

    if not questions:
        return {"error": "No questions could be evaluated", "scores": {}}

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    llm = _get_llm()
    embeddings = _get_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
    }

    eval_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(questions),
        "scores": scores,
        "per_question": [
            {
                "question": q,
                "answer": a[:200],
                "num_contexts": len(c),
            }
            for q, a, c in zip(questions, answers, contexts)
        ],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(eval_result, f, indent=2)

    logger.info(f"Evaluation complete: {scores}")
    return eval_result
