from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from rag.chunking import tokenize
from rag.config import DEFAULT_CHAT_MODEL, DEFAULT_LLM_MAX_TOKENS, env_first, is_llm_configured
from rag.config import parse_int_env, resolve_llm_base_url
from rag.http import RetryExhausted, call_with_retries, post_json

__all__ = [
    "AnthropicMessagesClient",
    "extract_anthropic_text",
    "split_sentences",
    "sentence_score",
    "generate_answer",
    "build_prompt_from_context_packet",
    "extract_citation_ids",
    "generate_answer_with_llm",
    "generate_answer_resilient",
    "validate_citations",
    "build_llm_status",
]


def _shim_value(name: str, default):
    shim = sys.modules.get("run_pipeline")
    return getattr(shim, name, default) if shim is not None else default


@dataclass
class AnthropicMessagesClient:
    api_key: str | None
    base_url: str

    def create_message(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        headers = {"anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        payload = post_json(f"{self.base_url.rstrip('/')}/v1/messages", body, headers=headers)
        return extract_anthropic_text(payload)


def extract_anthropic_text(payload: dict) -> str:
    content = payload.get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    return "".join(texts)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)
    return [item.strip() for item in re.split(r"(?<=[。！？!?])", text) if item.strip()]


def sentence_score(sentence: str, query: str, query_terms: set[str]) -> float:
    terms = set(tokenize(sentence))
    score = float(len(query_terms & terms))
    query_code_terms = [term for term in query_terms if re.search(r"[a-z0-9]", term)]
    if any(term in sentence.lower() for term in query_code_terms):
        score += 6
    if re.search(r"多久|几天|到账|时间", query):
        if re.search(r"通常|一般|为\s*\d|[0-9０-９]+\s*到\s*[0-9０-９]+", sentence):
            score += 7
        elif re.search(r"工作日|到账|时间", sentence):
            score += 3
        if "超过" in sentence:
            score -= 2
    if re.search(r"sku|SKU|无理由|退货|退款|支持", query) and re.search(r"SKU-A17|不支持|支持|无理由|质量问题", sentence):
        score += 4
    if "SKU-A17" in query and "SKU-A17" in sentence:
        score += 8
    if re.search(r"预售|48\s*小时|催单", query) and re.search(r"预售|48\s*小时|催单|承诺", sentence):
        score += 4
    if re.search(r"积分|提现|转赠", query) and re.search(r"积分|提现|转赠|兑换", sentence):
        score += 4
    return score


def generate_answer(context_packet: dict) -> dict:
    if not context_packet["sufficiency"]["enough"]:
        if context_packet["sufficiency"].get("reason") == "permission_denied":
            return {
                "answer": "当前权限不足，不能可靠回答这个问题。",
                "citations": [],
                "mode": "refusal",
            }
        return {
            "answer": "资料不足，不能可靠回答这个问题。",
            "citations": [],
            "mode": "refusal",
        }
    split = _shim_value("split_sentences", split_sentences)
    score_sentence = _shim_value("sentence_score", sentence_score)
    query_terms = set(tokenize(context_packet["query"]))
    claims = []
    citations = []
    for evidence in context_packet["evidence"]:
        best_sentence = ""
        best_overlap = -1
        for sentence in split(evidence["text"]):
            overlap = score_sentence(sentence, context_packet["query"], query_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        if best_sentence and evidence["citation_id"] not in citations:
            claims.append(f"{best_sentence} [{evidence['citation_id']}]")
            citations.append(evidence["citation_id"])
        if len(claims) >= 2:
            break
    return {
        "answer": "\n".join(claims) if claims else "资料不足，不能可靠回答这个问题。",
        "citations": citations,
        "mode": "extractive",
    }


def build_prompt_from_context_packet(context_packet: dict) -> str:
    evidence_blocks = []
    for evidence in context_packet["evidence"]:
        title_path = " > ".join(evidence.get("title_path", []))
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{evidence['citation_id']}]",
                    f"doc_id: {evidence.get('doc_id', '')}",
                    f"title_path: {title_path}",
                    f"source_path: {evidence.get('source_path', '')}",
                    f"version: {evidence.get('version', '')}",
                    f"role: {evidence.get('evidence_role', '')}",
                    "text:",
                    evidence.get("text", ""),
                ]
            )
        )
    return "\n\n".join(
        [
            "用户问题：",
            context_packet["query"],
            "",
            "证据包：",
            "\n\n---\n\n".join(evidence_blocks),
            "",
            "回答要求：",
            "1. 只能基于证据包回答，不要使用外部常识补全。",
            "2. 资料不足时直接说资料不足，不能可靠回答。",
            "3. 每个关键事实后面必须带引用，如 [E1]。",
            "4. 如果证据互相冲突，要说明冲突并分别引用。",
        ]
    )


def extract_citation_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[(E\d+)\]", text)))


def generate_answer_with_llm(context_packet: dict) -> dict:
    fallback_answer = _shim_value("generate_answer", generate_answer)
    if not context_packet["sufficiency"]["enough"]:
        return fallback_answer(context_packet)
    api_key = env_first("LLM_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")
    if not is_llm_configured():
        raise RuntimeError(
            "Missing LLM configuration. Set LLM_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, "
            "or an explicit LLM_BASE_URL/ANTHROPIC_BASE_URL/DEEPSEEK_BASE_URL."
        )
    model_name = env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL
    client_type = _shim_value("AnthropicMessagesClient", AnthropicMessagesClient)
    build_prompt = _shim_value("build_prompt_from_context_packet", build_prompt_from_context_packet)
    extract_ids = _shim_value("extract_citation_ids", extract_citation_ids)
    client = client_type(api_key=api_key, base_url=resolve_llm_base_url())
    answer_text = client.create_message(
        model=model_name,
        system_prompt="你是严谨的企业知识库 RAG 问答助手，只能基于给定证据回答，并且必须保留引用编号。",
        prompt=build_prompt(context_packet),
        max_tokens=parse_int_env("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),
    )
    return {
        "answer": answer_text,
        "citations": extract_ids(answer_text),
        "mode": f"llm:{model_name}",
    }


def generate_answer_resilient(context_packet: dict, status: dict) -> dict:
    fallback_answer = _shim_value("generate_answer", generate_answer)
    if not is_llm_configured():
        status.update(
            {
                "mode": "extractive_fallback",
                "fallback_used": True,
                "reason": "not_configured",
                "error": "",
                "attempts": 0,
            }
        )
        return fallback_answer(context_packet)
    if not context_packet["sufficiency"]["enough"]:
        status.update(
            {
                "mode": "skipped",
                "fallback_used": False,
                "reason": "insufficient_context",
                "error": "",
                "attempts": 0,
            }
        )
        return fallback_answer(context_packet)
    answer_with_llm = _shim_value("generate_answer_with_llm", generate_answer_with_llm)
    try:
        answer, attempts = call_with_retries(lambda: answer_with_llm(context_packet))
        status.update(
            {
                "mode": "llm",
                "fallback_used": False,
                "reason": "configured_model",
                "error": "",
                "attempts": attempts,
            }
        )
        return answer
    except RetryExhausted as exc:
        status.update(
            {
                "mode": "extractive_fallback",
                "fallback_used": True,
                "reason": "llm_error",
                "error": str(exc),
                "attempts": exc.attempts,
            }
        )
        return fallback_answer(context_packet)


def validate_citations(answer: dict, context_packet: dict) -> dict:
    available = {item["citation_id"] for item in context_packet["evidence"]}
    extract_ids = _shim_value("extract_citation_ids", extract_citation_ids)
    used = set(answer.get("citations") or extract_ids(answer.get("answer", "")))
    return {
        "citation_valid": used.issubset(available),
        "used_citations": sorted(used),
        "missing_citations": sorted(used - available),
        "available_citations": sorted(available),
    }


def build_llm_status() -> dict:
    configured = is_llm_configured()
    return {
        "component": "llm",
        "configured": configured,
        "requested_model": env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL,
        "model": env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL,
        "base_url": resolve_llm_base_url() if configured else "",
        "mode": "pending" if configured else "extractive_fallback",
        "fallback_used": not configured,
        "reason": "configured_model" if configured else "not_configured",
        "error": "",
        "attempts": 0,
    }
