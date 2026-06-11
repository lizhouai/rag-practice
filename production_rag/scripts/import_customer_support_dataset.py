from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw"
DATASET_NAME = "rjac/e-commerce-customer-support-qa"
DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


class MarkdownDocument:
    def __init__(self, filename: str, content: str) -> None:
        self.filename = filename
        self.content = content


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "customer_support"


def parse_qa_payload(raw_value: object) -> list[dict[str, str]]:
    if isinstance(raw_value, dict):
        payload = raw_value
    elif isinstance(raw_value, str) and raw_value.strip():
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return [{"customer_summary_question": "Original QA", "agent_summary_solution": raw_value}]
    else:
        return []

    knowledge = payload.get("knowledge", []) if isinstance(payload, dict) else []
    if not isinstance(knowledge, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in knowledge:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "customer_summary_question": str(item.get("customer_summary_question", "")).strip(),
                "agent_summary_solution": str(item.get("agent_summary_solution", "")).strip(),
            }
        )
    return normalized


def clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def render_case(row: dict, index: int) -> str:
    issue_area = clean_text(row.get("issue_area"))
    issue_category = clean_text(row.get("issue_category"))
    issue_sub_category = clean_text(row.get("issue_sub_category"))
    product_category = clean_text(row.get("product_category"))
    product_sub_category = clean_text(row.get("product_sub_category"))
    issue_complexity = clean_text(row.get("issue_complexity"))
    conversation = clean_text(row.get("conversation"))
    qa_items = parse_qa_payload(row.get("qa"))

    qa_lines = []
    for qa_index, item in enumerate(qa_items, start=1):
        question = clean_text(item.get("customer_summary_question"))
        solution = clean_text(item.get("agent_summary_solution"))
        if question:
            qa_lines.append(f"- Q{qa_index}: {question}")
        if solution:
            qa_lines.append(f"  Resolution: {solution}")

    if not qa_lines:
        qa_lines.append("- No normalized QA item was provided; use the conversation transcript as source evidence.")

    return "\n".join(
        [
            f"## Case {index}: {issue_sub_category or issue_category or issue_area}",
            "",
            f"- Issue area: {issue_area}",
            f"- Issue category: {issue_category}",
            f"- Issue sub-category: {issue_sub_category}",
            f"- Product category: {product_category}",
            f"- Product sub-category: {product_sub_category}",
            f"- Issue complexity: {issue_complexity}",
            "",
            "### Normalized knowledge",
            "",
            *qa_lines,
            "",
            "### Source conversation",
            "",
            conversation,
            "",
            "### RAG notes",
            "",
            "This case should be indexed as support knowledge. Keep the issue taxonomy in metadata-aware payload fields, use the normalized question for recall tests, and keep the conversation as supporting evidence for citation checks. If a generated answer uses this case, it should cite the case section rather than inventing a policy.",
        ]
    )


def rows_to_markdown_documents(rows: Iterable[dict], rows_per_doc: int = 12) -> list[MarkdownDocument]:
    documents: list[MarkdownDocument] = []
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= rows_per_doc:
            documents.append(render_document(batch, len(documents) + 1))
            batch = []
    if batch:
        documents.append(render_document(batch, len(documents) + 1))
    return documents


def render_document(rows: list[dict], doc_index: int) -> MarkdownDocument:
    first = rows[0]
    issue_area = clean_text(first.get("issue_area")) or "Customer Support"
    doc_id = f"brownbox_support_{doc_index:03d}_{slugify(issue_area)}"
    cases = [render_case(row, index) for index, row in enumerate(rows, start=1)]
    title = f"BrownBox Customer Support Cases {doc_index:03d}: {issue_area}"
    content = "\n".join(
        [
            "---",
            f"doc_id: {doc_id}",
            f"title: {title}",
            "business_domain: customer_support",
            "doc_type: open_dataset_casebook",
            "version: dataset_sample_v1",
            "effective_from: 2026-06-01",
            "effective_to: ",
            "permission_scope: internal",
            "owner: support_ops_team",
            f"dataset_source: {DATASET_NAME}",
            "dataset_license: MIT",
            "---",
            "",
            f"# {title}",
            "",
            "This document is generated from an open e-commerce customer support QA dataset. It is intentionally longer than a toy policy file so retrieval, BM25, RRF, rerank, deduplication, dynamic truncation, citation validation, and monitoring traces have enough evidence to operate on.",
            "",
            *cases,
            "",
        ]
    )
    return MarkdownDocument(filename=f"{doc_id}.md", content=content)


def load_huggingface_rows(limit: int) -> list[dict]:
    load_dataset_error: Exception | None = None
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        load_dataset_error = exc
    else:
        try:
            dataset = load_dataset(DATASET_NAME, split="train")
            return [dict(dataset[index]) for index in range(min(limit, len(dataset)))]
        except Exception as exc:
            load_dataset_error = exc

    try:
        return load_dataset_viewer_rows(limit)
    except Exception as viewer_exc:
        raise RuntimeError(format_huggingface_download_error(load_dataset_error, viewer_exc)) from viewer_exc


def load_dataset_viewer_rows(limit: int, page_size: int = 100) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_NAME,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": length,
            }
        )
        url = f"{DATASET_VIEWER_ROWS_URL}?{query}"
        with urllib.request.urlopen(url, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(dict(item.get("row", {})) for item in batch)
        offset += len(batch)
        if len(batch) < length:
            break
    if not rows:
        raise RuntimeError("Dataset Viewer returned no rows.")
    return rows[:limit]


def format_huggingface_download_error(
    load_dataset_error: Exception | None,
    dataset_viewer_error: Exception | None = None,
) -> str:
    error_lines = []
    if load_dataset_error is not None:
        error_lines.extend(["load_dataset error:", str(load_dataset_error)])
    if dataset_viewer_error is not None:
        error_lines.extend(["", "Dataset Viewer fallback error:", str(dataset_viewer_error)])
    return "\n".join(
        [
            f"Failed to download optional Hugging Face dataset `{DATASET_NAME}`.",
            "This does not block the main production RAG experiment because `data/raw/` already contains seed documents.",
            "The importer first tries `datasets.load_dataset`; if that fails, it falls back to the Hugging Face Dataset Viewer rows API.",
            "",
            "Common causes:",
            "- The current network cannot reach huggingface.co.",
            "- The current network can open Hugging Face pages but blocks datasets-server.huggingface.co or parquet/xet file downloads.",
            "- Git Bash/Python is not using the proxy configured in your browser.",
            "- The dataset is not present in the local Hugging Face cache.",
            "",
            "Try one of these in Git Bash, then rerun the importer:",
            "  export HTTPS_PROXY=http://127.0.0.1:7890",
            "  export HTTP_PROXY=http://127.0.0.1:7890",
            "  # or configure a trusted Hugging Face endpoint/mirror if your environment uses one:",
            "  export HF_ENDPOINT=https://your-huggingface-endpoint",
            "",
            "Original error:",
            "\n".join(error_lines) if error_lines else "unknown",
        ]
    )


def write_documents(documents: list[MarkdownDocument], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        (output_dir / document.filename).write_text(document.content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import open e-commerce customer support QA rows as RAG markdown docs.")
    parser.add_argument("--limit", type=int, default=120, help="Rows to import from Hugging Face.")
    parser.add_argument("--rows-per-doc", type=int, default=12, help="How many dataset rows to group into each markdown file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_huggingface_rows(args.limit)
    documents = rows_to_markdown_documents(rows, rows_per_doc=args.rows_per_doc)
    write_documents(documents, args.output_dir)
    print(f"Wrote {len(documents)} markdown documents to {args.output_dir}")


if __name__ == "__main__":
    main()
