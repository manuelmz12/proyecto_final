"""Unit tests for evaluation-service."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "evaluation-service"))


class TestTestQuestions:
    """Validate the test questions dataset structure."""

    def test_questions_file_exists(self):
        questions_path = os.path.join(
            os.path.dirname(__file__), "..", "services", "evaluation-service", "test_questions.json"
        )
        assert os.path.exists(questions_path), "test_questions.json must exist"

    def test_questions_schema(self):
        questions_path = os.path.join(
            os.path.dirname(__file__), "..", "services", "evaluation-service", "test_questions.json"
        )
        with open(questions_path) as f:
            data = json.load(f)

        assert isinstance(data, list), "Root must be a list"
        assert len(data) >= 10, "Must have at least 10 questions"

        for item in data:
            assert "question" in item, "Each item must have 'question'"
            assert "ground_truth" in item, "Each item must have 'ground_truth'"
            assert len(item["question"]) > 10, "Question must be non-trivial"
            assert len(item["ground_truth"]) > 20, "Ground truth must be non-trivial"

    def test_questions_are_unique(self):
        questions_path = os.path.join(
            os.path.dirname(__file__), "..", "services", "evaluation-service", "test_questions.json"
        )
        with open(questions_path) as f:
            data = json.load(f)

        questions = [item["question"] for item in data]
        assert len(questions) == len(set(questions)), "Questions must be unique"


class TestEvalResultsSchema:
    """Validate evaluation result output structure."""

    def test_results_schema(self):
        sample_result = {
            "timestamp": "2026-04-20T10:00:00+00:00",
            "num_questions": 10,
            "scores": {
                "faithfulness": 0.82,
                "answer_relevancy": 0.91,
                "context_recall": 0.75,
            },
            "per_question": [
                {"question": "What is RAG?", "answer": "RAG is...", "num_contexts": 3}
            ],
        }
        assert "scores" in sample_result
        assert "faithfulness" in sample_result["scores"]
        assert 0.0 <= sample_result["scores"]["faithfulness"] <= 1.0
        assert "answer_relevancy" in sample_result["scores"]
        assert "context_recall" in sample_result["scores"]
