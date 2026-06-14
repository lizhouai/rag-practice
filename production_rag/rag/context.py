from __future__ import annotations

import re
import sys

from rag.chunking import tokenize
from rag.config import MIN_RERANK_SCORE, SCORE_POLICY_EXTERNAL_RERANK, SCORE_POLICY_RRF_ONLY
from rag.models import Candidate, Chunk, SelectedEvidence
from rag.selection import expand_parent_context

__all__ = [
    "is_out_of_domain_query",
    "sufficiency_check",
    "assemble_context",
]


def _shim_value(name: str, default):
    shim = sys.modules.get("run_pipeline")
    return getattr(shim, name, default) if shim is not None else default


def is_out_of_domain_query(query: str) -> bool:
    out_of_domain_patterns = (
        r"天气|气温|下雨|空气质量",
        r"股票|股价|汇率|彩票",
        r"新闻|热搜|比赛|比分",
    )
    in_domain_patterns = (
        r"订单|退款|退货|换货|物流|快递|发货|配送|签收|发票|税号|优惠券|积分|会员|保修|维修|召回|商品|商家|客服|账号|登录|验证码|地址",
        r"SKU|COD|CoD|invoice|refund|return|delivery|warranty|account|login",
    )
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in out_of_domain_patterns):
        return not any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in in_domain_patterns)
    return False


def sufficiency_check(
    query: str,
    selected: list[Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    permission_blocked_matches: list[dict] | None = None,
    score_policy: str = SCORE_POLICY_EXTERNAL_RERANK,
    min_score: float = MIN_RERANK_SCORE,
    high_confidence_score: float | None = 0.35,
) -> dict:
    is_out_of_domain = _shim_value("is_out_of_domain_query", is_out_of_domain_query)
    if is_out_of_domain(query):
        return {"enough": False, "reason": "out_of_domain_query"}
    query_terms = set(tokenize(query))
    if not selected:
        if permission_blocked_matches:
            return {
                "enough": False,
                "reason": "permission_denied",
                "blocked_doc_ids": sorted({item["doc_id"] for item in permission_blocked_matches}),
            }
        return {"enough": False, "reason": "no_selected_evidence"}
    coverage = set()
    for candidate in selected:
        coverage.update(query_terms & set(chunks_by_id[candidate.chunk_id].terms))
    coverage_ratio = len(coverage) / max(1, len(query_terms))
    best_score = max(item.rerank_score for item in selected)
    if permission_blocked_matches:
        blocked_ratio = float(permission_blocked_matches[0].get("overlap_ratio", 0.0))
        if blocked_ratio >= 0.45 and blocked_ratio >= coverage_ratio + 0.15:
            return {
                "enough": False,
                "reason": "permission_denied",
                "coverage_ratio": round(coverage_ratio, 4),
                "blocked_overlap_ratio": round(blocked_ratio, 4),
                "blocked_doc_ids": sorted({item["doc_id"] for item in permission_blocked_matches}),
            }
    if score_policy == _shim_value("SCORE_POLICY_RRF_ONLY", SCORE_POLICY_RRF_ONLY):
        lexical_coverage = set()
        for candidate in selected:
            if candidate.bm25_rank is not None:
                lexical_coverage.update(query_terms & set(chunks_by_id[candidate.chunk_id].terms))
        lexical_coverage_ratio = len(lexical_coverage) / max(1, len(query_terms))
        has_lexical_signal = bool(lexical_coverage)
        enough = lexical_coverage_ratio >= 0.18
        if enough:
            reason = "pass"
        elif not has_lexical_signal:
            reason = "rrf_only_missing_lexical_signal"
        else:
            reason = "low_query_evidence_overlap"
        return {
            "enough": enough,
            "coverage_ratio": round(coverage_ratio, 4),
            "best_rerank_score": round(best_score, 4),
            "score_policy": score_policy,
            "score_confidence": False,
            "lexical_signal": has_lexical_signal,
            "lexical_coverage_ratio": round(lexical_coverage_ratio, 4),
            "reason": reason,
        }
    enough = best_score >= min_score and (
        coverage_ratio >= 0.18
        or (high_confidence_score is not None and best_score >= high_confidence_score)
    )
    return {
        "enough": enough,
        "coverage_ratio": round(coverage_ratio, 4),
        "best_rerank_score": round(best_score, 4),
        "score_policy": score_policy,
        "score_confidence": True,
        "reason": "pass" if enough else "low_query_evidence_overlap",
    }


def assemble_context(
    query: str,
    selected: list[Candidate | SelectedEvidence],
    chunks_by_id: dict[str, Chunk],
    sufficiency: dict,
    *,
    chunks_by_parent: dict[str, list[Chunk]] | None = None,
) -> dict:
    expand_context = _shim_value("expand_parent_context", expand_parent_context)
    evidence = []
    estimated_token_total = 0
    for index, selection in enumerate(selected, start=1):
        if isinstance(selection, SelectedEvidence):
            candidate = selection.candidate
            expanded_text = selection.expanded_text
            expanded_ids = selection.expanded_from_chunk_ids
            expanded_tokens = selection.expanded_token_count
        else:
            candidate = selection
            chunk = chunks_by_id[candidate.chunk_id]
            expanded_text, expanded_ids, expanded_tokens = expand_context(chunk, chunks_by_parent)
        chunk = chunks_by_id[candidate.chunk_id]
        estimated_token_total += expanded_tokens
        role = "primary" if index == 1 else "supporting"
        if any(word in chunk.text for word in ("不支持", "不能", "除非")):
            role = "exception" if index > 1 else "primary"
        evidence.append(
            {
                "citation_id": f"E{index}",
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "title_path": chunk.title_path,
                "source_path": chunk.metadata.get("source_path", ""),
                "version": chunk.metadata.get("version", ""),
                "effective_from": chunk.metadata.get("effective_from", ""),
                "rerank_score": round(candidate.rerank_score, 4),
                "mmr_score": round(candidate.mmr_score, 4),
                "evidence_role": role,
                "expanded_from_chunk_ids": expanded_ids,
                "text": expanded_text,
            }
        )
    return {
        "query": query,
        "policy": [
            "只能使用本上下文包中的资料回答",
            "资料不足时必须拒答",
            "关键事实必须就近引用资料编号",
        ],
        "sufficiency": sufficiency,
        "estimated_token_total": estimated_token_total,
        "evidence": evidence,
    }
