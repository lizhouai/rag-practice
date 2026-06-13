# 工业级 RAG 改造 + 模块化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `production_rag/run_pipeline.py`（3208 行单文件）拆成 `rag/` 包，并把 Qdrant 查询路径从 O(全库) 改成 retrieve-then-hydrate，使每次查询的数据量与语料规模解耦。

**Architecture:** Qdrant 退化为纯向量+过滤索引（不再存 text/terms），SQLite 升级为权威 docstore（id→文本、parent_id→兄弟块）。召回返回 id+score+向量 → 本地批量 hydrate 文本 → parent 扩展只取最终选中块的兄弟。权限拆为服务端强制执行（不变）+ count 审计 + query 驱动的 blocked 提示。

**Tech Stack:** Python 3.11，stdlib only（urllib + sqlite3 + concurrent.futures），unittest。Qdrant Cloud（REST）+ 本地 SQLite。

参考 spec：[docs/specs/2026-06-12-industrial-rag-modularization-design.md](../specs/2026-06-12-industrial-rag-modularization-design.md)

---

## 约定

- **工作目录**：所有命令在 `production_rag/` 下运行。
- **Python**：`.venv/Scripts/python.exe`（Windows + bash）。
- **全量测试**：`.venv/Scripts/python.exe -m unittest discover -s tests`（基线 54 通过）。
- **单模块测试**：`.venv/Scripts/python.exe -m unittest tests.test_<name> -v`。
- **提交粒度**：每个 Task 一次提交；不在默认分支直接做，先开 `feat/rag-modularization` 分支或 worktree。

## 提取策略（P1 全程适用）

P1 是**纯搬迁、行为零改动**。每个提取 Task 做四件事：
1. 新建 `rag/<module>.py`，把指定符号从 `run_pipeline.py` **原样剪切**过去（连同它们的 docstring/注释）。
2. 在 `run_pipeline.py` 原位置用 `from rag.<module> import *`（或显式 import）替换，使 `import run_pipeline as rag; rag.<symbol>` 仍然解析——现有测试 `tests/test_production_pipeline.py` 全靠这个 shim 保持绿。
3. 跑全量测试，必须 54 通过。**若有 NameError/ImportError，说明漏搬了某个符号或它的依赖——把缺的符号一起搬过去或补 import。测试套件就是搬迁完整性的校验器。**
4. 提交。

新模块顶部按需 `from __future__ import annotations` + 该模块真正用到的 stdlib import（`json` / `re` / `math` / `hashlib` / `urllib.*` / `sqlite3` / `time` / `uuid` / `dataclasses` / `datetime` / `pathlib` / `collections` / `typing` / `contextlib`）。模块间依赖按 spec §7 的 DAG，只能从叶向根 import，禁止反向（出现循环 import 即违反分层）。

---

## Phase 1 — 纯提取到 `rag/` 包

### Task 1: 包脚手架 + `rag/config.py`

**Files:**
- Create: `rag/__init__.py`（空）
- Create: `rag/config.py`
- Modify: `run_pipeline.py`（顶部常量块 + env/resolve 函数 → 改为 import）

- [ ] **Step 1: 建包目录**

```bash
mkdir -p rag/vectorstore
touch rag/__init__.py rag/vectorstore/__init__.py
```

- [ ] **Step 2: 移动常量与配置函数到 `rag/config.py`**

把以下符号从 `run_pipeline.py` 剪切到 `rag/config.py`（保持定义顺序）：
- 全部模块级常量（`ROOT`、`RAW_DIR`、`DEFAULT_VECTOR_DB_PATH`、`METRICS_PATH`、`DEFAULT_QDRANT_URL`、`DEFAULT_VECTOR_BACKEND`、`DEFAULT_QDRANT_COLLECTION`、`DEFAULT_RETRY_ATTEMPTS`、`DEFAULT_RETRY_BACKOFF_SECONDS`、`QDRANT_DENSE_VECTOR_NAME`、`QDRANT_BM25_VECTOR_NAME`、`QDRANT_SPARSE_HASH_BUCKETS`、`QDRANT_PAYLOAD_INDEXES`、`DENSE_TOP_N`、`BM25_TOP_N`、`RERANK_TOP_N`、`RRF_K`、`DEDUP_THRESHOLD`、`MMR_LAMBDA`、`FINAL_MAX_K`、`MIN_RERANK_SCORE`、`GAP_THRESHOLD`、`CONTEXT_TOKEN_BUDGET`、`PARENT_EXPANSION_MAX_CHARS`、`SCORE_POLICY_*`、全部 embedding/llm/reranker 默认常量、`STOPWORDS`、`LOCAL_EMBEDDING_HOSTS` 等）。
- env/路径/identity helpers：`load_env`、`env_first`、`has_env_value`、`parse_int_env`、`resolve_qdrant_url`、`resolve_qdrant_collection`、`resolve_qdrant_api_key`、`is_qdrant_configured`、`resolve_vector_dimensions`、`resolve_llm_base_url`、`resolve_embedding_base_url`、`embedding_identity`、`local_store_path_for_embedding_identity`、`is_llm_configured`、`is_embedding_configured`、以及其它纯读 env / 纯计算路径的 `resolve_*` / `*_identity` 函数。

`DEFAULT_EMBEDDING_DIMENSIONS` 若是从 `resolve_vector_dimensions()` 派生，连同一起搬。

- [ ] **Step 3: 在 `run_pipeline.py` 顶部替换为 import**

删掉已搬走的定义，在文件顶部（`from __future__` 之后）加：

```python
from rag.config import *  # noqa: F401,F403 - re-export shim during modularization
from rag import config as config  # for fully-qualified access if needed
```

- [ ] **Step 4: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`。若报 `NameError: name 'X' is not defined`，把 X 一并搬进 `config.py`。

- [ ] **Step 5: Commit**

```bash
git add rag/__init__.py rag/config.py run_pipeline.py
git commit -m "refactor(p1): extract config and constants into rag/config.py"
```

### Task 2: `rag/models.py`

**Files:**
- Create: `rag/models.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动 dataclasses**

剪切 `Chunk`、`Candidate`、`ParentSection`、`SelectedEvidence`，以及其它纯数据 dataclass（如 `MonitoringEvent` 之类无行为的结构体）到 `rag/models.py`。顶部加 `from dataclasses import dataclass, field`。

- [ ] **Step 2: 在 `run_pipeline.py` 替换为 import**

```python
from rag.models import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/models.py run_pipeline.py
git commit -m "refactor(p1): extract dataclasses into rag/models.py"
```

### Task 3: `rag/http.py`

**Files:**
- Create: `rag/http.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动网络/重试原语**

剪切 `request_json`、`format_transport_error`、`call_with_retries`、`RetryExhausted`、`ComponentFallback`（异常类），以及重试相关 helper 到 `rag/http.py`。顶部 import `json`、`urllib.request`、`urllib.error`、`urllib.parse`、`time`，并 `from rag.config import DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_BACKOFF_SECONDS`。

- [ ] **Step 2: 替换为 import**

```python
from rag.http import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`（含 `test_request_json_error_includes_target_url`、`test_qdrant_cloud_url_without_rest_port_gets_actionable_error`）

- [ ] **Step 4: Commit**

```bash
git add rag/http.py run_pipeline.py
git commit -m "refactor(p1): extract http/retry primitives into rag/http.py"
```

### Task 4: `rag/chunking.py`

**Files:**
- Create: `rag/chunking.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动文本处理**

剪切 `tokenize`、`vectorize`、`qdrant_sparse_index`、`qdrant_sparse_vector_from_terms`、`sqlite_fts_query`、`split_sections`、`chunk_parent`、`build_document_chunks`、`parse_frontmatter`、`read_documents`、`content_hash`、`safe_relative`、`split_metadata_values`、`date_to_sortable_day`、`allowed_scope_key`、`scope_allowed`（若是纯函数）到 `rag/chunking.py`。顶部 import `re`、`hashlib`、`math`、`json`，`from rag.config import ...`（STOPWORDS、维度等）。

> 注：`tokenize` 是索引侧与查询侧共享的分词器，是 BM25 稀疏向量对齐的关键不变量（spec §3）。它必须只此一份。

- [ ] **Step 2: 替换为 import**

```python
from rag.chunking import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/chunking.py run_pipeline.py
git commit -m "refactor(p1): extract tokenize/chunking/document parsing into rag/chunking.py"
```

### Task 5: `rag/embedding.py`

**Files:**
- Create: `rag/embedding.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动 embedding 层**

剪切 `make_retrying_embedding_function`、`build_embedding_status`、embedding HTTP 调用函数、以及 embedding 相关的 fallback 构造到 `rag/embedding.py`。`from rag.http import request_json, call_with_retries, ComponentFallback`，`from rag.config import ...`，`from rag.chunking import tokenize, vectorize`。

- [ ] **Step 2: 替换为 import**

```python
from rag.embedding import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`（含 `test_external_embedding_success_*`、`test_local_embedding_success_*`）

- [ ] **Step 4: Commit**

```bash
git add rag/embedding.py run_pipeline.py
git commit -m "refactor(p1): extract embedding client into rag/embedding.py"
```

### Task 6: `rag/vectorstore/filters.py`

**Files:**
- Create: `rag/vectorstore/filters.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动过滤构造（纯函数）**

剪切 `qdrant_scope_filter`、`qdrant_effective_filter`、`qdrant_access_filter`、`qdrant_permission_blocked_filter`、`qdrant_not_yet_effective_filter`、`qdrant_expired_filter`、`qdrant_filter_by_doc_id` 到 `rag/vectorstore/filters.py`。`from rag.config import ...`，`from rag.chunking import date_to_sortable_day`。

- [ ] **Step 2: 替换为 import**

```python
from rag.vectorstore.filters import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/vectorstore/filters.py run_pipeline.py
git commit -m "refactor(p1): extract qdrant filter builders into rag/vectorstore/filters.py"
```

### Task 7: `rag/vectorstore/qdrant.py`

**Files:**
- Create: `rag/vectorstore/qdrant.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动 QdrantVectorStore**

剪切 `QdrantVectorStore` 类、`stable_point_id`、`qdrant_point_dense_vector`、`chunk_to_qdrant_payload`、`chunk_from_qdrant_payload` 到 `rag/vectorstore/qdrant.py`。import：`from rag.http import request_json, call_with_retries`、`from rag.config import ...`、`from rag.models import Chunk`、`from rag.chunking import qdrant_sparse_vector_from_terms, tokenize, split_metadata_values`、`from rag.vectorstore.filters import qdrant_access_filter`。

- [ ] **Step 2: 替换为 import**

```python
from rag.vectorstore.qdrant import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`（含 `QdrantVectorStoreTest` 全部，已含上一轮新增的 `test_load_access_chunks_scrolls_all_categories_without_vectors`、`test_run_query_qdrant_recall_vectors_feed_mmr_dedup`）

- [ ] **Step 4: Commit**

```bash
git add rag/vectorstore/qdrant.py run_pipeline.py
git commit -m "refactor(p1): extract QdrantVectorStore into rag/vectorstore/qdrant.py"
```

### Task 8: `rag/vectorstore/sqlite.py`

**Files:**
- Create: `rag/vectorstore/sqlite.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动 LocalVectorStore**

剪切 `LocalVectorStore` 类及其 SQL helper（`chunk_select_columns`、`chunk_from_sql_row`、`fetch_chunks`、schema 建表 SQL、FTS5 相关）到 `rag/vectorstore/sqlite.py`。import：`from rag.config import ...`、`from rag.models import Chunk`、`from rag.chunking import tokenize, sqlite_fts_query, allowed_scope_key, date_to_sortable_day, scope_allowed`、`sqlite3`、`json`、`from contextlib import closing`。

- [ ] **Step 2: 替换为 import**

```python
from rag.vectorstore.sqlite import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`（含 `IncrementalSyncTest`、本地 bm25/access 相关）

- [ ] **Step 4: Commit**

```bash
git add rag/vectorstore/sqlite.py run_pipeline.py
git commit -m "refactor(p1): extract LocalVectorStore into rag/vectorstore/sqlite.py"
```

### Task 9: `rag/vectorstore/mirrored.py`

**Files:**
- Create: `rag/vectorstore/mirrored.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动 MirroredVectorStore**

剪切 `MirroredVectorStore` 到 `rag/vectorstore/mirrored.py`。`from rag.vectorstore.sqlite import LocalVectorStore`、`from rag.models import Chunk`。

- [ ] **Step 2: 替换为 import**

```python
from rag.vectorstore.mirrored import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/vectorstore/mirrored.py run_pipeline.py
git commit -m "refactor(p1): extract MirroredVectorStore into rag/vectorstore/mirrored.py"
```

### Task 10: `rag/retrieval.py`

**Files:**
- Create: `rag/retrieval.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动召回融合**

剪切 `dense_recall`、`bm25_recall`、`rrf_fuse`、`cosine`、`summarize_results` 到 `rag/retrieval.py`。`from rag.models import Chunk, Candidate`、`from rag.config import DENSE_TOP_N, BM25_TOP_N, RRF_K`。

- [ ] **Step 2: 替换为 import**

```python
from rag.retrieval import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/retrieval.py run_pipeline.py
git commit -m "refactor(p1): extract recall/fusion into rag/retrieval.py"
```

### Task 11: `rag/rerank.py`

**Files:**
- Create: `rag/rerank.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动重排**

剪切 `rerank`、`ExternalReranker`、`make_external_reranker`、`make_configured_external_reranker`、`describe_reranker`、`parse_reranker_scores`、score policy 常量相关逻辑到 `rag/rerank.py`。`from rag.http import ...`、`from rag.config import ...`、`from rag.models import Candidate, Chunk`。

- [ ] **Step 2: 替换为 import**

```python
from rag.rerank import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/rerank.py run_pipeline.py
git commit -m "refactor(p1): extract rerank into rag/rerank.py"
```

### Task 12: `rag/selection.py`

**Files:**
- Create: `rag/selection.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动选择/截断/扩展**

剪切 `mmr_select`、`dynamic_truncate`、`expand_parent_context`、`build_chunks_by_parent` 到 `rag/selection.py`。`from rag.models import Candidate, Chunk, SelectedEvidence`、`from rag.config import ...`、`from rag.retrieval import cosine`。

- [ ] **Step 2: 替换为 import**

```python
from rag.selection import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/selection.py run_pipeline.py
git commit -m "refactor(p1): extract mmr/truncate/parent-expansion into rag/selection.py"
```

### Task 13: `rag/context.py`

**Files:**
- Create: `rag/context.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动上下文组装**

剪切 `assemble_context`、`sufficiency_check` 及其 helper 到 `rag/context.py`。`from rag.models import ...`、`from rag.config import CONTEXT_TOKEN_BUDGET`。

- [ ] **Step 2: 替换为 import**

```python
from rag.context import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/context.py run_pipeline.py
git commit -m "refactor(p1): extract context assembly into rag/context.py"
```

### Task 14: `rag/generation.py`

**Files:**
- Create: `rag/generation.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动生成层**

剪切 `generate_answer_resilient`、LLM 客户端（Anthropic messages client）、`build_llm_status`、`validate_citations`、引用解析 helper 到 `rag/generation.py`。`from rag.http import ...`、`from rag.config import ...`、`from rag.context import ...`。

- [ ] **Step 2: 替换为 import**

```python
from rag.generation import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`（含 `ProviderClientTest`、`CitationAndMonitoringTest`）

- [ ] **Step 4: Commit**

```bash
git add rag/generation.py run_pipeline.py
git commit -m "refactor(p1): extract LLM/generation into rag/generation.py"
```

### Task 15: `rag/access.py`

**Files:**
- Create: `rag/access.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动权限提示逻辑**

剪切 `find_permission_blocked_matches`、`fallback_summary`、以及权限相关的纯逻辑 helper 到 `rag/access.py`。`from rag.models import Chunk`、`from rag.chunking import tokenize`、`from rag.vectorstore.filters import ...`。（query 驱动的 blocked 提示在 P3 加入本模块。）

- [ ] **Step 2: 替换为 import**

```python
from rag.access import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/access.py run_pipeline.py
git commit -m "refactor(p1): extract permission-hint logic into rag/access.py"
```

### Task 16: `rag/indexing.py`

**Files:**
- Create: `rag/indexing.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动索引同步**

剪切 `sync_index`、`make_vector_store`、`initialize_vector_store_with_fallback`、`set_vector_store_fallback`、`index_status_without_rebuild`、`sync_index_with_store_fallback` 到 `rag/indexing.py`。import 各 vectorstore + chunking + embedding。

- [ ] **Step 2: 替换为 import**

```python
from rag.indexing import *  # noqa: F401,F403
```

- [ ] **Step 3: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 4: Commit**

```bash
git add rag/indexing.py run_pipeline.py
git commit -m "refactor(p1): extract index sync into rag/indexing.py"
```

### Task 17: `rag/pipeline.py` + monitoring

**Files:**
- Create: `rag/pipeline.py`
- Create: `rag/monitoring.py`
- Modify: `run_pipeline.py`

- [ ] **Step 1: 移动监控**

剪切 `build_monitoring_event`、`persist_monitoring_event` 到 `rag/monitoring.py`。

- [ ] **Step 2: 移动编排**

剪切 `run_query`（含内部 `load_access_state`）到 `rag/pipeline.py`。它 import 几乎所有 rag 模块。这是 shim 反向收口：`run_pipeline.py` 此后只剩 argparse/main/eval 入口 + 顶部的 `from rag.* import *` re-export（暂留，P4 移除）。

- [ ] **Step 3: 替换为 import**

```python
from rag.monitoring import *  # noqa: F401,F403
from rag.pipeline import run_query  # noqa: F401
```

- [ ] **Step 4: 跑全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 54 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add rag/monitoring.py rag/pipeline.py run_pipeline.py
git commit -m "refactor(p1): extract run_query orchestration into rag/pipeline.py"
```

### Task 18: P1 冒烟验证

- [ ] **Step 1: 真实查询端到端跑通**

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe run_pipeline.py --query "跨境订单退款多久到账？" --trace-only --no-monitoring > _p1_smoke.json 2>_p1_err.txt; echo "exit=$?"
```
Expected: `exit=0`，`_p1_smoke.json` 是合法 trace JSON。

- [ ] **Step 2: 确认 backend 与 stage**

Run:
```bash
.venv/Scripts/python.exe -c "import json; t=json.load(open('_p1_smoke.json',encoding='utf-8')); print(t['component_status']['vector_store']['backend'], t['component_status']['vector_store']['fallback_used']); print(list(t['stage_latencies_ms']))"
```
Expected: `qdrant False` + 完整 stage 列表（行为与 P1 前一致）。

- [ ] **Step 3: 清理临时文件 + Commit**

```bash
rm -f _p1_smoke.json _p1_err.txt
git commit --allow-empty -m "chore(p1): smoke-verify behavior unchanged after extraction"
```

---

## Phase 2 — 引入 docstore + retrieve-then-hydrate

### Task 19: SQLite docstore 接口 + parent_id 索引

**Files:**
- Create: `rag/docstore.py`
- Modify: `rag/vectorstore/sqlite.py`（建表加 `parent_id` 索引）
- Test: `tests/test_docstore.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_docstore.py
from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_pipeline as rag  # shim still re-exports during P2
from rag.docstore import SqliteDocstore


def _chunk(cid: str, parent: str, text: str) -> "rag.Chunk":
    return rag.Chunk(
        chunk_id=cid, parent_id=parent, doc_id="doc-1",
        title_path=["doc", "sec"], text=text, metadata={"permission_scope": "internal"},
        token_count=5, dense_vector=[0.1, 0.2], terms=rag.tokenize(text),
    )


class SqliteDocstoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        store_path = Path(self.tmp.name) / "rag.sqlite"
        self.store = rag.LocalVectorStore(store_path)
        self.store.upsert_document("doc-1", "doc.md", "h1", "local:test", [
            _chunk("c1", "p1", "alpha text"),
            _chunk("c2", "p1", "beta text"),
            _chunk("c3", "p2", "gamma text"),
        ])
        self.docstore = SqliteDocstore(store_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_hydrate_returns_only_requested_ids_with_text(self) -> None:
        out = self.docstore.hydrate(["c1", "c3", "missing"])
        self.assertEqual(set(out), {"c1", "c3"})
        self.assertEqual(out["c1"].text, "alpha text")
        self.assertEqual(out["c3"].parent_id, "p2")

    def test_siblings_groups_by_parent_id(self) -> None:
        out = self.docstore.siblings(["p1"])
        self.assertEqual(sorted(c.chunk_id for c in out["p1"]), ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_docstore -v`
Expected: FAIL（`ModuleNotFoundError: rag.docstore` 或 `ImportError: SqliteDocstore`）

- [ ] **Step 3: 在 `sqlite.py` 建表处加 parent_id 索引**

在 `LocalVectorStore` 建表 SQL 之后（`chunks` 表创建后）加：

```python
connection.execute(
    "CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id)"
)
```

- [ ] **Step 4: 实现 `rag/docstore.py`**

```python
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from rag.models import Chunk
from rag.vectorstore.sqlite import LocalVectorStore


class SqliteDocstore:
    """Read-only hydration over the SQLite store: id->text and parent_id->siblings.

    Used by the Qdrant happy path, where recall returns ids + vectors and the
    full chunk text is fetched locally instead of crossing the network.
    """

    def __init__(self, path: Path) -> None:
        self._store = LocalVectorStore(path)

    def _connect(self) -> sqlite3.Connection:
        return self._store.connect()

    def hydrate(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        ids = [cid for cid in dict.fromkeys(chunk_ids) if cid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT {self._store.chunk_select_columns('c')} "
                f"FROM chunks c WHERE c.chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row[0]: self._store.chunk_from_sql_row(row) for row in rows}

    def siblings(self, parent_ids: list[str]) -> dict[str, list[Chunk]]:
        pids = [pid for pid in dict.fromkeys(parent_ids) if pid]
        if not pids:
            return {}
        placeholders = ",".join("?" for _ in pids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT {self._store.chunk_select_columns('c')} "
                f"FROM chunks c WHERE c.parent_id IN ({placeholders}) "
                f"ORDER BY c.chunk_id",
                pids,
            ).fetchall()
        grouped: dict[str, list[Chunk]] = {pid: [] for pid in pids}
        for row in rows:
            chunk = self._store.chunk_from_sql_row(row)
            grouped[chunk.parent_id].append(chunk)
        return grouped
```

> 若 `connect` / `chunk_select_columns` / `chunk_from_sql_row` 当前不是 `LocalVectorStore` 的可访问方法，在 Task 8 提取时确保它们是实例方法/静态方法（已是）。

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.test_docstore -v`
Expected: PASS（2 tests）

- [ ] **Step 6: 全量回归**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: `Ran 56 tests ... OK`

- [ ] **Step 7: Commit**

```bash
git add rag/docstore.py rag/vectorstore/sqlite.py tests/test_docstore.py
git commit -m "feat(p2): add SqliteDocstore hydrate/siblings + parent_id index"
```

### Task 20: Qdrant 写路径停存 text/terms

**Files:**
- Modify: `rag/vectorstore/qdrant.py`（`chunk_to_qdrant_payload`）
- Test: `tests/test_production_pipeline.py`（`test_upsert_document_sends_chunks_to_qdrant_points_api`）

- [ ] **Step 1: 改测试断言 payload 不含 text/terms**

在 `test_upsert_document_sends_chunks_to_qdrant_points_api` 末尾追加：

```python
        self.assertNotIn("text", body["points"][0]["payload"])
        self.assertNotIn("terms", body["points"][0]["payload"])
        self.assertEqual(body["points"][0]["payload"]["parent_id"], chunk.parent_id)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_production_pipeline.QdrantVectorStoreTest.test_upsert_document_sends_chunks_to_qdrant_points_api -v`
Expected: FAIL（payload 当前含 text/terms）

- [ ] **Step 3: 修改 `chunk_to_qdrant_payload`**

从返回 dict 中删除 `"text"` 和 `"terms"` 两个键（保留 `chunk_id` / `parent_id` / `doc_id` / `permission_scope` / `permission_scopes` / `effective_*` / `token_count` / `source_path` / `content_hash` / `embedding_model` / `title_path` / `metadata`）。dense 向量与 bm25 稀疏向量仍在 `upsert_document` 的 `vector` 字段中，不受影响。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.test_production_pipeline.QdrantVectorStoreTest -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add rag/vectorstore/qdrant.py tests/test_production_pipeline.py
git commit -m "feat(p2): stop storing text/terms in qdrant payload (docstore owns them)"
```

### Task 21: pipeline Qdrant 分支改 retrieve-then-hydrate

**Files:**
- Modify: `rag/pipeline.py`（`run_query`：Qdrant 分支与 `load_access_state`）
- Test: `tests/test_pipeline_hydrate.py`

- [ ] **Step 1: 写失败的集成测试（核心回归守卫）**

```python
# tests/test_pipeline_hydrate.py
from __future__ import annotations
import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import run_pipeline as rag


class PipelineHydrateTest(unittest.TestCase):
    def test_qdrant_path_issues_no_full_scroll_and_hydrates_from_docstore(self) -> None:
        seen_urls: list[str] = []

        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            seen_urls.append(url)
            if method == "GET" and url.endswith("/collections/rag_test"):
                return {"result": {
                    "payload_schema": {n: {} for n in rag.QDRANT_PAYLOAD_INDEXES},
                    "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}},
                }}
            if url.endswith("/points/query") and body.get("using") == rag.QDRANT_DENSE_VECTOR_NAME:
                # payload has NO text (docstore owns it); vector is returned
                return {"result": {"points": [
                    {"id": 1, "score": 0.9, "vector": {"dense": [1.0, 0.0]},
                     "payload": {"chunk_id": "c1", "parent_id": "p1", "doc_id": "d1",
                                 "title_path": ["t"], "token_count": 5,
                                 "permission_scopes": ["internal"]}},
                ]}}
            if url.endswith("/points/query") and body.get("using") == rag.QDRANT_BM25_VECTOR_NAME:
                return {"result": {"points": []}}
            raise AssertionError(f"unexpected request {method} {url}")

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            # Seed docstore with the chunk text that hydrate must supply.
            rag.LocalVectorStore(store_path).upsert_document(
                "d1", "d.md", "h", "local:test",
                [rag.Chunk(chunk_id="c1", parent_id="p1", doc_id="d1",
                           title_path=["t"], text="退款 7 到 15 个工作日", metadata={"permission_scope": "internal"},
                           token_count=5, dense_vector=[1.0, 0.0], terms=rag.tokenize("退款 工作日"))],
            )
            with (
                patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test",
                                        "QDRANT_COLLECTION": "rag_test"}, clear=True),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query("退款多久", quiet=True, vector_backend="qdrant",
                                      store_path=store_path, metrics_path=Path(tmp) / "m.jsonl")

        # 1) no full-collection scroll on the qdrant happy path
        self.assertTrue(all(not u.endswith("/points/scroll") for u in seen_urls),
                        f"unexpected scroll in {seen_urls}")
        # 2) bounded request count (probe + dense + bm25, no per-corpus calls)
        self.assertLessEqual(len(seen_urls), 4)
        # 3) text came from docstore hydrate
        self.assertIn("退款", json.dumps(trace["context_packet"], ensure_ascii=False))
        self.assertEqual(trace["component_status"]["vector_store"]["backend"], "qdrant")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate -v`
Expected: FAIL（当前 Qdrant 分支会调 `load_access_chunks` → `/points/scroll`，断言 1 触发；或 hydrate 缺失导致 context 无文本）

- [ ] **Step 3: 改写 `run_query` 的 Qdrant 分支**

把 `load_access_state` 改为 backend 分叉。Qdrant 分支不再调 `vector_store.load_access_chunks`，改为：召回先行 → 用 `SqliteDocstore.hydrate` 取文本 → 召回向量回填。具体：

1. 在 `rag/pipeline.py` 顶部 import：`from rag.docstore import SqliteDocstore`。
2. Qdrant 分支（`resolved_vector_backend == "qdrant"`）：
   - 先做 dense + bm25 召回（已带 `qdrant_access_filter` 服务端过滤），得到 `dense_results` / `bm25_results`（含 id+score+dense_vector，payload 无 text）。
   - 取所有候选 id：`candidate_ids = [c.chunk_id for _, c in dense_results] + [c.chunk_id for _, c in bm25_results]`。
   - `docstore = SqliteDocstore(active_store_path)`；`hydrated = docstore.hydrate(candidate_ids)`。
   - 构建 `chunks_by_id`：对每个召回 chunk，用 `hydrated[id]` 的 text/terms/title_path/token_count 填充，dense_vector 用召回带回的向量（hydrated 里的本地向量与召回向量一致，优先用召回向量保证与查询同源）。
   - `chunks_by_parent`：对最终需要的 parent，用 `docstore.siblings(...)` 在 truncate 后取（见 Task 22），此处先按候选构建一个浅版本供 rerank/mmr。
   - 删除该分支里对 `load_access_chunks` / 全量 visible 的调用。
   - `parents_count` / `chunks_count` 暂以候选数填充（P3 改 count API）。
3. 本地分支（`else`）：保持现有 `lexical_store.load_access_chunks(scopes)` 全量加载（暴力 dense 需要），不动。

> 成员校验 `if chunk.chunk_id in chunks_by_id`（旧第 2921 行）删除——召回已服务端过滤，单一真相在服务端（spec §6）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: OK（旧 `test_run_query_qdrant_recall_vectors_feed_mmr_dedup` 仍绿——它已注入带 text 的 payload，hydrate 也能覆盖）

- [ ] **Step 6: Commit**

```bash
git add rag/pipeline.py tests/test_pipeline_hydrate.py
git commit -m "feat(p2): qdrant path uses retrieve-then-hydrate instead of full scroll"
```

### Task 22: parent 扩展改用 docstore.siblings

**Files:**
- Modify: `rag/pipeline.py`（truncate 后的 parent 扩展接线）
- Test: `tests/test_pipeline_hydrate.py`

- [ ] **Step 1: 加失败测试——兄弟块来自 docstore 而非召回**

在 `PipelineHydrateTest` 增加用例：docstore 里 `p1` 有两个 chunk（`c1` 被召回、`c2` 未召回），断言最终 `context_packet` 的扩展文本包含 `c2` 的文本（证明兄弟块来自 docstore.siblings，而非仅召回集）。

```python
    def test_parent_expansion_pulls_unrecalled_sibling_from_docstore(self) -> None:
        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            if method == "GET":
                return {"result": {"payload_schema": {n: {} for n in rag.QDRANT_PAYLOAD_INDEXES},
                                   "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}}}}
            if body.get("using") == rag.QDRANT_DENSE_VECTOR_NAME:
                return {"result": {"points": [
                    {"id": 1, "score": 0.9, "vector": {"dense": [1.0, 0.0]},
                     "payload": {"chunk_id": "c1", "parent_id": "p1", "doc_id": "d1",
                                 "title_path": ["t"], "token_count": 5, "permission_scopes": ["internal"]}}]}}
            return {"result": {"points": []}}

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            rag.LocalVectorStore(store_path).upsert_document(
                "d1", "d.md", "h", "local:test",
                [rag.Chunk(chunk_id="c1", parent_id="p1", doc_id="d1", title_path=["t"],
                           text="第一段 已召回", metadata={"permission_scope": "internal"}, token_count=5,
                           dense_vector=[1.0, 0.0], terms=rag.tokenize("第一段")),
                 rag.Chunk(chunk_id="c2", parent_id="p1", doc_id="d1", title_path=["t"],
                           text="第二段 未召回兄弟", metadata={"permission_scope": "internal"}, token_count=5,
                           dense_vector=[0.0, 1.0], terms=rag.tokenize("第二段"))],
            )
            with (patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                  patch.object(rag, "request_json", fake_request_json)):
                trace = rag.run_query("第一段", quiet=True, vector_backend="qdrant",
                                      store_path=store_path, metrics_path=Path(tmp) / "m.jsonl")
        self.assertIn("第二段 未召回兄弟", json.dumps(trace["context_packet"], ensure_ascii=False))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate.PipelineHydrateTest.test_parent_expansion_pulls_unrecalled_sibling_from_docstore -v`
Expected: FAIL（兄弟块 c2 不在召回集，当前 chunks_by_parent 不含它）

- [ ] **Step 3: 在 truncate 后用 docstore.siblings 重建最终 parent 上下文**

在 `run_query` 的 Qdrant 分支里，`dynamic_truncate` 选出 `selected`（≤FINAL_MAX_K）后、`expand_parent_context` 之前：

```python
if resolved_vector_backend == "qdrant":
    final_parent_ids = [chunks_by_id[item.candidate.chunk_id].parent_id for item in selected]
    sibling_groups = docstore.siblings(final_parent_ids)
    chunks_by_parent = {pid: chunks for pid, chunks in sibling_groups.items()}
    for chunks in sibling_groups.values():
        for sib in chunks:
            chunks_by_id.setdefault(sib.chunk_id, sib)
```

> 仅对最终 ≤5 个 parent 取兄弟块，请求量与语料无关。`dynamic_truncate` 已接受 `chunks_by_parent`，但当前在 truncate 内部做扩展。若扩展发生在 truncate 内，则把 sibling 取数移到 truncate 之前、按候选的 parent 预取；保持"只取候选+最终 parent"的有界性即可。实现者按 `dynamic_truncate` 实际签名二选一，关键不变量：**不得为兄弟块发起全库请求**。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate -v`
Expected: PASS（3 tests）

- [ ] **Step 5: 全量回归 + 真实查询冒烟**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: OK

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe run_pipeline.py --query "跨境订单退款多久到账？" --trace-only --no-monitoring > _p2.json 2>&1; .venv/Scripts/python.exe -c "import json; t=json.load(open('_p2.json',encoding='utf-8')); print(t['stage_latencies_ms']); print(t['component_status']['vector_store']['backend'])"; rm -f _p2.json
```
Expected: `access_filter` 显著下降（不再全量 scroll），backend=qdrant。

- [ ] **Step 6: Commit**

```bash
git add rag/pipeline.py tests/test_pipeline_hydrate.py
git commit -m "feat(p2): parent expansion fetches siblings from docstore (bounded)"
```

---

## Phase 3 — 权限重做 + trace 演进

### Task 23: Qdrant count API（审计计数）

**Files:**
- Modify: `rag/vectorstore/qdrant.py`（新增 `count`）
- Test: `tests/test_production_pipeline.py`（QdrantVectorStoreTest）

- [ ] **Step 1: 写失败测试**

```python
    def test_count_posts_filtered_count_request(self) -> None:
        captured: dict = {}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return None
            def read(self): return b'{"result":{"count":7}}'

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = request.data
            return FakeResponse()

        store = rag.QdrantVectorStore(base_url="http://qdrant.test", collection_name="rag_test", vector_size=64)
        import urllib.request
        with patch("urllib.request.urlopen", fake_urlopen):
            n = store.count(rag.qdrant_expired_filter({"internal"}, "2026-06-07"))
        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(captured["url"], "http://qdrant.test/collections/rag_test/points/count")
        self.assertEqual(body["filter"], rag.qdrant_expired_filter({"internal"}, "2026-06-07"))
        self.assertEqual(body["exact"], True)
        self.assertEqual(n, 7)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_production_pipeline.QdrantVectorStoreTest.test_count_posts_filtered_count_request -v`
Expected: FAIL（`AttributeError: 'QdrantVectorStore' object has no attribute 'count'`）

- [ ] **Step 3: 实现 `count`**

在 `QdrantVectorStore` 加：

```python
def count(self, access_filter: dict | None = None) -> int:
    body: dict[str, object] = {"exact": True}
    if access_filter is not None:
        body["filter"] = access_filter
    payload = request_json(
        "POST", self.collection_url("/points/count"),
        body=body, headers=self.headers(), ok_statuses=(200,),
    )
    result = payload.get("result", {})
    return int(result.get("count", 0)) if isinstance(result, dict) else 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.test_production_pipeline.QdrantVectorStoreTest -v`
Expected: PASS

- [ ] **Step 5: 全量回归 + Commit**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests` → OK

```bash
git add rag/vectorstore/qdrant.py tests/test_production_pipeline.py
git commit -m "feat(p3): add QdrantVectorStore.count for audit counts"
```

### Task 24: trace 审计字段改 count 语义

**Files:**
- Modify: `rag/pipeline.py`（trace 组装 + Qdrant 分支审计）
- Test: `tests/test_pipeline_hydrate.py`

- [ ] **Step 1: 加失败测试——trace 用 count 而非全量清单**

在 `PipelineHydrateTest` 增加：mock 让 `/points/count` 返回固定数，断言 `trace["permission_filter"]` 含 `rejected_counts`（如 `{"expired": N, "not_yet_effective": M}`），且 `rejected_chunks` 不再是全量清单（长度有界或不存在）。

```python
    def test_trace_uses_count_for_time_audit(self) -> None:
        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            if method == "GET":
                return {"result": {"payload_schema": {n: {} for n in rag.QDRANT_PAYLOAD_INDEXES},
                                   "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}}}}
            if url.endswith("/points/count"):
                # expired filter has range lt; not_yet has range gt
                rendered = json.dumps(body["filter"])
                return {"result": {"count": 3 if '"lt"' in rendered else 2}}
            if body.get("using") == rag.QDRANT_DENSE_VECTOR_NAME:
                return {"result": {"points": [
                    {"id": 1, "score": 0.9, "vector": {"dense": [1.0, 0.0]},
                     "payload": {"chunk_id": "c1", "parent_id": "p1", "doc_id": "d1",
                                 "title_path": ["t"], "token_count": 5, "permission_scopes": ["internal"]}}]}}
            return {"result": {"points": []}}

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            rag.LocalVectorStore(store_path).upsert_document(
                "d1", "d.md", "h", "local:test",
                [rag.Chunk(chunk_id="c1", parent_id="p1", doc_id="d1", title_path=["t"], text="x",
                           metadata={"permission_scope": "internal"}, token_count=5,
                           dense_vector=[1.0, 0.0], terms=rag.tokenize("x"))])
            with (patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                  patch.object(rag, "request_json", fake_request_json)):
                trace = rag.run_query("x", quiet=True, vector_backend="qdrant",
                                      store_path=store_path, metrics_path=Path(tmp) / "m.jsonl")
        self.assertEqual(trace["permission_filter"]["rejected_counts"],
                         {"expired": 3, "not_yet_effective": 2})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate.PipelineHydrateTest.test_trace_uses_count_for_time_audit -v`
Expected: FAIL（无 `rejected_counts`）

- [ ] **Step 3: Qdrant 分支用 count 填审计 + 改 trace schema**

在 Qdrant 分支：
```python
rejected_counts = {
    "expired": vector_store.count(qdrant_expired_filter(scopes)),
    "not_yet_effective": vector_store.count(qdrant_not_yet_effective_filter(scopes)),
}
```
trace 的 `permission_filter` 改为：
```python
"permission_filter": {
    "allowed_scopes": sorted(scopes),
    "visible_chunks": parents_count_or_visible_count,  # 见 Task 25
    "rejected_counts": rejected_counts,                # 替代全量 rejected_chunks
    "blocked_matches": permission_blocked_matches,     # P3 Task 26 定义；默认 []
},
```
本地分支保留旧 `rejected_chunks` 全量清单（成本低、语料小），但同样补一个 `rejected_counts` 派生自清单长度，保持 trace 字段一致。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate -v` → PASS
Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: 旧断言 `permission_filter.rejected_chunks` 的测试需同步更新为 `rejected_counts`（CitationAndMonitoringTest 若涉及）。逐个修绿。

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline.py tests/
git commit -m "feat(p3): time-window audit via count API, trace exposes rejected_counts"
```

### Task 25: chunks_count / parents_count 改 count 语义

**Files:**
- Modify: `rag/pipeline.py`
- Test: `tests/test_pipeline_hydrate.py`

- [ ] **Step 1: 加失败测试**

断言 Qdrant 路径下 `trace["chunks_count"]` 等于 `count(access_filter)` 的返回（"可见集规模"），而非召回数。沿用 count mock：`access_filter`（must scope+effective，无 range lt/gt）返回固定值（如 87），断言 `trace["chunks_count"] == 87`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_pipeline_hydrate.PipelineHydrateTest.test_visible_count_from_count_api -v`
Expected: FAIL

- [ ] **Step 3: 实现**

Qdrant 分支：`visible_count = vector_store.count(qdrant_access_filter(scopes))`；`chunks_count`/`parents_count` 用它填（`parents_count` 无法从 count 直接得到 parent 去重数——改为 trace 注明 `chunks_count` 为可见集规模，`parents_count` 仅统计本次召回+扩展涉及的 parent 数，文档化语义差异）。本地分支沿用现值。

- [ ] **Step 4: 测试通过 + 全量回归 + Commit**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests` → OK

```bash
git add rag/pipeline.py tests/test_pipeline_hydrate.py
git commit -m "feat(p3): chunks_count reflects visible-set size via count API"
```

### Task 26: query 驱动的 blocked 提示 + `--blocked-hint`

**Files:**
- Modify: `rag/access.py`（新增 query 驱动函数）
- Modify: `rag/pipeline.py`（接线 + 开关）
- Modify: `run_pipeline.py`（argparse 加 `--blocked-hint`）
- Test: `tests/test_access_blocked_hint.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_access_blocked_hint.py
from __future__ import annotations
import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import run_pipeline as rag


class BlockedHintTest(unittest.TestCase):
    def _fake(self, blocked_recall_points):
        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            if method == "GET":
                return {"result": {"payload_schema": {n: {} for n in rag.QDRANT_PAYLOAD_INDEXES},
                                   "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}}}}
            if url.endswith("/points/count"):
                return {"result": {"count": 0}}
            if url.endswith("/points/query"):
                f = json.dumps(body.get("filter", {}))
                if "must_not" in f:                       # blocked-hint recall
                    return {"result": {"points": blocked_recall_points}}
                return {"result": {"points": []}}          # primary recall empty
            raise AssertionError(url)
        return fake_request_json

    def test_blocked_hint_off_by_default_no_extra_recall(self) -> None:
        calls: list[str] = []
        def fake(method, url, body=None, headers=None, ok_statuses=(200,)):
            calls.append(json.dumps(body.get("filter", {})) if body else "")
            return self._fake([])(method, url, body, headers, ok_statuses)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            rag.LocalVectorStore(store_path).upsert_document("d1", "d.md", "h", "local:test",
                [rag.Chunk(chunk_id="c1", parent_id="p1", doc_id="d1", title_path=["t"], text="x",
                           metadata={"permission_scope": "internal"}, token_count=5,
                           dense_vector=[1.0, 0.0], terms=rag.tokenize("x"))])
            with (patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                  patch.object(rag, "request_json", self._fake([]))):
                trace = rag.run_query("x", quiet=True, vector_backend="qdrant",
                                      store_path=store_path, metrics_path=Path(tmp) / "m.jsonl")
        self.assertEqual(trace["permission_filter"]["blocked_matches"], [])

    def test_blocked_hint_on_surfaces_relevant_blocked_titles(self) -> None:
        blocked_points = [{"id": 9, "score": 3.0, "vector": {"dense": [0.0, 1.0]},
                           "payload": {"chunk_id": "b1", "parent_id": "pb", "doc_id": "secret",
                                       "title_path": ["机密", "薪酬"], "token_count": 5,
                                       "permission_scopes": ["finance_restricted"]}}]
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            rag.LocalVectorStore(store_path).upsert_document("d1", "d.md", "h", "local:test",
                [rag.Chunk(chunk_id="c1", parent_id="p1", doc_id="d1", title_path=["t"], text="x",
                           metadata={"permission_scope": "internal"}, token_count=5,
                           dense_vector=[1.0, 0.0], terms=rag.tokenize("x"))])
            with (patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                  patch.object(rag, "request_json", self._fake(blocked_points))):
                trace = rag.run_query("薪酬", quiet=True, vector_backend="qdrant", blocked_hint=True,
                                      store_path=store_path, metrics_path=Path(tmp) / "m.jsonl")
        titles = [m["title_path"] for m in trace["permission_filter"]["blocked_matches"]]
        self.assertIn("机密 > 薪酬", titles)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.test_access_blocked_hint -v`
Expected: FAIL（`run_query` 无 `blocked_hint` 参数 / 无 query 驱动逻辑）

- [ ] **Step 3: 实现 query 驱动 blocked 提示**

`rag/access.py` 新增：

```python
def query_driven_blocked_matches(query, vector_store, scopes, docstore, *, top_n=5):
    """Recall blocked-but-effective chunks relevant to THIS query, return title hints.

    Uses the blocked filter (effective range, scope excluded) so only content the
    user cannot see — but that matches the query — is surfaced. Titles only.
    """
    from rag.vectorstore.filters import qdrant_permission_blocked_filter
    sparse = qdrant_sparse_vector_from_terms(tokenize(query))
    if not sparse["indices"]:
        return []
    points = vector_store.bm25_search_with_filter(
        query, qdrant_permission_blocked_filter(scopes), top_n=top_n
    )
    matches = []
    for score, chunk in points:
        matches.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title_path": " > ".join(chunk.title_path),
            "score": round(float(score), 4),
        })
    return matches
```

> `bm25_search_with_filter` = 把 `bm25_search` 的 filter 参数化（当前写死 `qdrant_access_filter`）。在 `QdrantVectorStore.bm25_search` 上加一个可选 `access_filter` 参数（默认沿用 `qdrant_access_filter(scopes)`），blocked 提示传入 blocked filter。title_path 来自召回 payload（payload 仍保留 title_path）。

`rag/pipeline.py`：`run_query` 签名加 `blocked_hint: bool = False`；Qdrant 分支末尾：
```python
permission_blocked_matches = (
    query_driven_blocked_matches(query, vector_store, scopes, docstore)
    if blocked_hint else []
)
```
本地分支：`blocked_hint` 时沿用现有 `find_permission_blocked_matches`（它已有全量 blocked 集），否则 `[]`。

`run_pipeline.py`：argparse 加 `--blocked-hint`（`action="store_true"`），透传给 `run_query(blocked_hint=args.blocked_hint)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.test_access_blocked_hint -v`
Expected: PASS（2 tests）

- [ ] **Step 5: 全量回归 + Commit**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests` → OK

```bash
git add rag/access.py rag/pipeline.py rag/vectorstore/qdrant.py run_pipeline.py tests/test_access_blocked_hint.py
git commit -m "feat(p3): query-driven blocked-relevant hint behind --blocked-hint (default off)"
```

---

## Phase 4 — 收尾：拆测试 + 移除 shim

### Task 27: 按模块拆分测试文件

**Files:**
- Create: `tests/test_chunking.py`、`tests/test_vectorstore_qdrant.py`、`tests/test_vectorstore_sqlite.py`、`tests/test_retrieval.py`、`tests/test_generation.py` 等
- Modify/Delete: `tests/test_production_pipeline.py`

- [ ] **Step 1: 迁移用例**

把 `tests/test_production_pipeline.py` 里的用例按被测模块归位到对应 `tests/test_<module>.py`，import 改为 `from rag.<module> import <symbol>`（不再 `import run_pipeline as rag`）。`make_chunk` 等公共 helper 提到 `tests/helpers.py`。

- [ ] **Step 2: 全量回归**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: 总数不变（≈60+），全绿。

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "refactor(p4): split test suite per module, import from rag package"
```

### Task 28: 移除 re-export shim，`run_pipeline.py` 收为薄 CLI

**Files:**
- Modify: `run_pipeline.py`

- [ ] **Step 1: 删除 `from rag.* import *` shim**

`run_pipeline.py` 只保留：argparse（`parse_args`）、`main`、eval 入口（`run_eval`/读 `eval_cases.csv`），以及精确 import：
```python
from rag.config import load_env, DEFAULT_VECTOR_BACKEND, DEFAULT_VECTOR_DB_PATH, METRICS_PATH, split_metadata_values, DEFAULT_ALLOWED_SCOPES
from rag.pipeline import run_query
```
eval 路径用到的其它符号按需精确 import。

- [ ] **Step 2: CLI 冒烟**

Run:
```bash
.venv/Scripts/python.exe run_pipeline.py --help
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe run_pipeline.py --query "跨境订单退款多久到账？" --trace-only --no-monitoring --blocked-hint > _p4.json 2>&1; .venv/Scripts/python.exe -c "import json;t=json.load(open('_p4.json',encoding='utf-8'));print(t['stage_latencies_ms']);print(t['permission_filter'].keys())"; rm -f _p4.json
```
Expected: `--help` 含 `--blocked-hint`；查询返回合法 trace，`permission_filter` 含 `rejected_counts`。

- [ ] **Step 3: 全量回归**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: 全绿。

- [ ] **Step 4: Commit**

```bash
git add run_pipeline.py
git commit -m "refactor(p4): run_pipeline.py becomes thin CLI entry, shim removed"
```

### Task 29: eval 全跑 + 文档同步

**Files:**
- Modify: `production_rag/README.md`（模块结构 + retrieve-then-hydrate 说明 + `--blocked-hint`）
- Modify: `IMPROVEMENTS.md`（勾掉 "O(全库)查询" 条目，注明已解耦）

- [ ] **Step 1: 跑 eval 套件**

Run:
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe run_pipeline.py --eval --no-monitoring 2>&1 | tail -20
```
Expected: eval 通过率不低于改造前基线（记录数字）。

- [ ] **Step 2: 更新 README/IMPROVEMENTS**

README 增 `rag/` 包结构表（同 spec §7）、查询数据流（spec §4）、`--blocked-hint` 说明。IMPROVEMENTS.md 标注 "O(全库)查询" 已通过 retrieve-then-hydrate 解决（Qdrant 路径）。

- [ ] **Step 3: Commit**

```bash
git add production_rag/README.md IMPROVEMENTS.md
git commit -m "docs(p4): document rag/ package layout and retrieve-then-hydrate"
```

---

## Self-Review 结果

**Spec 覆盖**：§3 职责分离→T20；BM25 不变量→T4(tokenizer 共享)+T20(payload)；§4 查询流→T21/T22；§5 写流→T20；§6 权限(visible/审计/blocked/backend 不对称)→T21/T23/T24/T26；§7 模块拆分→T1-T17；§8 接口演进→T24/T25/T26/T28；§9 测试(无全库 scroll 守卫)→T21；§10 迁移 P1-P4→四个 Phase；§11 决策(blocked 默认关/count 语义/parent_id 索引)→T26/T24-25/T19。无遗漏。

**Placeholder 扫描**：无 TBD/TODO；提取任务用精确符号清单（搬迁的"完整规格"），新逻辑任务给完整 TDD 代码。

**类型/签名一致性**：`SqliteDocstore.hydrate/siblings`（T19）↔ pipeline 调用（T21/T22）一致；`QdrantVectorStore.count`（T23）↔ 审计调用（T24/T25）一致；`bm25_search` 参数化 filter（T26）↔ blocked 提示一致；`run_query(blocked_hint=...)`（T26）↔ CLI 透传（T28）一致。

**已知执行注意**：P1 提取的符号清单基于通读，可能有个别 helper 未列全——以"每次提取后全量测试必须 54 绿"为完整性校验器，缺啥补啥。

---

## 执行交接

见技能末尾的执行方式选择。
