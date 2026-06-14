from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers import make_chunk, rag


class PermissionFilterTest(unittest.TestCase):
    def test_filter_chunks_for_access_removes_out_of_scope_and_expired_docs(self) -> None:
        chunks = [
            make_chunk("internal", "internal"),
            make_chunk("support", "support_only"),
            make_chunk("expired", "internal", effective_to="2026-01-31"),
        ]

        visible, rejected = rag.filter_chunks_for_access(
            chunks,
            allowed_scopes={"internal"},
            today="2026-06-07",
        )

        self.assertEqual([chunk.chunk_id for chunk in visible], ["internal"])
        self.assertEqual(
            {item["chunk_id"]: item["reason"] for item in rejected},
            {"support": "permission_scope", "expired": "expired"},
        )

    def test_default_scopes_include_public_but_exclude_restricted_docs(self) -> None:
        chunks = [
            make_chunk("internal", "internal"),
            make_chunk("public", "public"),
            make_chunk("finance", "finance_restricted"),
            make_chunk("security", "security_restricted"),
        ]

        visible, rejected = rag.filter_chunks_for_access(
            chunks,
            allowed_scopes=rag.DEFAULT_ALLOWED_SCOPES,
            today="2026-06-07",
        )

        self.assertEqual([chunk.chunk_id for chunk in visible], ["internal", "public"])
        self.assertEqual(
            {item["chunk_id"]: item["reason"] for item in rejected},
            {"finance": "permission_scope", "security": "permission_scope"},
        )
        self.assertEqual(rag.DEFAULT_ALLOWED_SCOPES, {"internal", "public"})

    def test_sufficiency_refuses_when_best_match_is_permission_blocked(self) -> None:
        visible = make_chunk("visible", "internal")
        visible.terms = rag.tokenize("普通处理说明")
        blocked = make_chunk("blocked", "finance_restricted")
        blocked.doc_id = "finance_refund_reconciliation_v1"
        blocked.terms = rag.tokenize("FR-21 差异工单 处理")
        rejected = [{"chunk_id": blocked.chunk_id, "doc_id": blocked.doc_id, "reason": "permission_scope"}]

        blocked_matches = rag.find_permission_blocked_matches(
            "FR-21 差异工单怎么处理？",
            [visible, blocked],
            rejected,
        )
        result = rag.sufficiency_check(
            "FR-21 差异工单怎么处理？",
            [rag.Candidate(chunk_id=visible.chunk_id, rerank_score=0.7, reason="visible")],
            {visible.chunk_id: visible},
            permission_blocked_matches=blocked_matches,
        )
        answer = rag.generate_answer(
            {
                "query": "FR-21 差异工单怎么处理？",
                "sufficiency": result,
                "evidence": [],
            }
        )

        self.assertEqual(result["reason"], "permission_denied")
        self.assertEqual(answer["answer"], "当前权限不足，不能可靠回答这个问题。")


if __name__ == "__main__":
    unittest.main()
