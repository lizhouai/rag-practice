from __future__ import annotations

from datetime import datetime

from rag.chunking import date_to_sortable_day

__all__ = [
    "qdrant_filter_by_doc_id",
    "qdrant_scope_filter",
    "qdrant_effective_filter",
    "qdrant_access_filter",
    "qdrant_permission_blocked_filter",
    "qdrant_not_yet_effective_filter",
    "qdrant_expired_filter",
]


def qdrant_filter_by_doc_id(doc_id: str) -> dict:
    return {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}


def qdrant_scope_filter(allowed_scopes: set[str]) -> dict:
    scopes = sorted(scope for scope in allowed_scopes if scope)
    if not scopes:
        return {"must": [{"key": "__empty_scope__", "match": {"value": "__never__"}}]}
    return {"should": [{"key": "permission_scopes", "match": {"value": scope}} for scope in scopes]}


def qdrant_effective_filter(today: str | None = None) -> list[dict]:
    today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
    return [
        {"key": "effective_from_day", "range": {"lte": today_day}},
        {"key": "effective_to_day", "range": {"gte": today_day}},
    ]


def qdrant_access_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    return {
        "must": [qdrant_scope_filter(allowed_scopes), *qdrant_effective_filter(today)],
    }


def qdrant_permission_blocked_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    scopes = sorted(scope for scope in allowed_scopes if scope)
    return {
        "must": qdrant_effective_filter(today),
        "must_not": [{"key": "permission_scopes", "match": {"value": scope}} for scope in scopes],
    }


def qdrant_not_yet_effective_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
    return {
        "must": [
            qdrant_scope_filter(allowed_scopes),
            {"key": "effective_from_day", "range": {"gt": today_day}},
        ],
    }


def qdrant_expired_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
    return {
        "must": [
            qdrant_scope_filter(allowed_scopes),
            {"key": "effective_to_day", "range": {"lt": today_day}},
        ],
    }
