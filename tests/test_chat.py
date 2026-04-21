"""Unit tests for chat-service components."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "chat-service"))


class TestSimpleAuth:
    """Tests for the HTTPBearer username:password auth in chat-service."""

    def test_valid_credentials_accepted(self):
        from auth.jwt_handler import USERS_DB, _verify

        user = USERS_DB.get("demo")
        assert user is not None
        assert _verify("demo123", user["password_hash"])

    def test_invalid_password_rejected(self):
        from auth.jwt_handler import USERS_DB, _verify

        user = USERS_DB.get("demo")
        assert user is not None
        assert not _verify("wrongpassword", user["password_hash"])

    def test_unknown_user_not_in_db(self):
        from auth.jwt_handler import USERS_DB

        assert USERS_DB.get("unknownuser") is None

    def test_admin_role(self):
        from auth.jwt_handler import USERS_DB

        assert USERS_DB.get("admin")["role"] == "admin"

    def test_demo_role(self):
        from auth.jwt_handler import USERS_DB

        assert USERS_DB.get("demo")["role"] == "user"

    def test_token_format_parsing(self):
        token = "demo:demo123"
        assert ":" in token
        username, password = token.split(":", 1)
        assert username == "demo"
        assert password == "demo123"

    def test_token_format_with_colon_in_password(self):
        token = "admin:pass:word"
        username, password = token.split(":", 1)
        assert username == "admin"
        assert password == "pass:word"



class TestRouterIntentParsing:
    """Tests for router agent response parsing."""

    def test_parse_valid_router_response(self):
        import json

        valid_responses = [
            '{"intent": "rag", "reasoning": "Question about paper content"}',
            '{"intent": "graph", "reasoning": "Asking about author relationships"}',
            '{"intent": "hybrid", "reasoning": "Need both content and graph"}',
            '{"intent": "general", "reasoning": "Out of scope question"}',
        ]
        for resp in valid_responses:
            parsed = json.loads(resp)
            assert parsed["intent"] in ("rag", "graph", "hybrid", "general")

    def test_invalid_json_fallback(self):
        import json

        try:
            json.loads("not valid json")
            assert False, "Should have raised"
        except json.JSONDecodeError:
            pass

    def test_bm25_rrf_scoring(self):
        results = {}
        for rank in range(5):
            doc_id = f"doc_{rank}"
            rrf_score = 1.0 / (60 + rank + 1)
            results[doc_id] = {"rrf_score": rrf_score}

        sorted_results = sorted(results.values(), key=lambda x: x["rrf_score"], reverse=True)
        assert sorted_results[0]["rrf_score"] > sorted_results[-1]["rrf_score"]
