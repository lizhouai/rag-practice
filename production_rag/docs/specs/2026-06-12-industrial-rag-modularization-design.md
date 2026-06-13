# 工业级 RAG 改造 + 模块化设计

- 日期：2026-06-12
- 状态：待评审（brainstorming 产出，下一步转 writing-plans）
- 范围：`production_rag/run_pipeline.py`（3208 行单文件）
- 决策前提：方案 A（SQLite 升级为 docstore）+ 包化拆分；保持 stdlib；允许对外接口演进

---

## 1. 问题与背景

当前 `run_pipeline.py` 是 3208 行单文件，stdlib-only（urllib + sqlite3），Qdrant Cloud 做向量库、SQLite 做镜像兜底。已在本会话定位并修复了 `vector_store_probe` 的 schema-drift 延迟，以及 `access_filter` 的全量 scroll 延迟（21.9s → 3.1s，向量不再无谓回传 + 四类 scroll 并发）。

但**结构性问题仍在**：`load_access_state` 每次查询仍把整个可见语料拉回客户端，用于成员校验、parent 上下文扩展、trace 统计。根因是 **Qdrant 同时承担"向量索引"和"全文仓库"两个职责**，导致查询数据量 = O(全库)，随语料线性恶化（102 chunk 时 3.1s，万级即分钟级）。

详见 [IMPROVEMENTS.md](../../IMPROVEMENTS.md) 的"O(全库)查询"条目。

## 2. 目标 / 非目标

**目标**
1. 查询数据量与语料规模解耦：每查只取 O(召回数 + 最终选中数)，不再拉全库。
2. 把单文件拆成职责清晰、可独立理解和测试的模块（`rag/` 包），`run_pipeline.py` 缩为薄 CLI 入口。
3. 引入工业级 RAG 的核心模式：**vector store ≠ docstore**，检索后再取（retrieve-then-hydrate）。
4. 顺带消除跨境文本传输延迟（文本改由本地 SQLite 提供）。

**非目标（本次不做）**
- 多机/分布式共享 docstore（Redis/Postgres）——单机 SQLite 足够，留作后续。
- service/repository 分层、依赖注入容器、异步化——属"全量分层工业化"档，已排除。
- 引入 qdrant-client / 框架等重依赖——保持 stdlib。
- 连接池消除 TLS 握手底价——属依赖层优化，本次聚焦数据流解耦。

## 3. 架构决策：retrieve-then-hydrate（方案 A）

### 职责分离
- **Qdrant = 纯向量 + 过滤索引**。每个 point 存：dense 向量、bm25 稀疏向量，以及过滤必需的 payload（`chunk_id` / `doc_id` / `parent_id` / `permission_scopes` / `effective_from_day` / `effective_to_day`）。**不再存 `text` 和 `terms`。**
- **SQLite = 权威 docstore**。表存 `chunk_id → {text, terms, title_path, token_count, parent_id, doc_id, metadata}`。现有 `LocalVectorStore` 镜像从"降级兜底"升级为读路径一等公民。按 embedding identity 分文件的现状保留（见 `local_store_path_for_embedding_identity`）。

### 权限过滤位置（保持服务端，不回退）
权限 + 时效过滤继续 100% 由 Qdrant payload filter 在召回时服务端执行（已实现，见 `qdrant_access_filter`）。Qdrant 模式下**不再需要全量 visible scroll 来做客户端二次校验**——召回结果本来就只含可见点，单一真相在服务端。（本地兜底模式仍需全量 visible，原因见 §6 的 backend 不对称。）

### BM25 不变量
BM25 结构上几乎不变，但有两条必须守住：
- **共享 tokenizer**：索引侧（`chunk.terms`）与查询侧（`tokenize(query)`）必须用同一个分词器，否则稀疏向量的桶 id 对不上、BM25 失效。`tokenize` 留在 `chunking.py` 作为两侧共享依赖。
- **稀疏向量留 Qdrant、原始 terms 进 docstore**：Qdrant 仍存 `bm25` 稀疏向量（服务端评分需要，`modifier:idf`），但 payload 不再带原始 `terms` 列表——后者移入 SQLite docstore。注：当前是 IDF 加权的 TF 点积（BM25 风味），非含 k1/b 的严格 BM25；本地兜底路径走 SQLite FTS5 才是真 BM25。

## 4. 查询数据流（新）

```
1.  config         解析 embedding / qdrant / llm 配置
2.  embedding      embed(query) → query_vector（失败 → hash 兜底）
3.  vectorstore    Qdrant 可达性探测（不可达 → 本地 sqlite 兜底）
4.  retrieval      dense_recall: Qdrant /points/query
                     (filter=access_filter, with_vector=[dense]) → top-N [(score,id,vector)]
5.  retrieval      bm25_recall:  Qdrant /points/query (sparse, filter=access_filter) → top-N [(score,id)]
6.  retrieval      rrf_fuse: 按 id 融合 → 有序候选 id（~≤24 个）
7.  docstore       hydrate(候选ids): SQLite 批量 SELECT → {id: Chunk}，并把召回向量回填到 chunk
8.  rerank         打分（外部 reranker 或 rrf_only 策略）
9.  selection      mmr_select: 用召回向量做去重 → diversified
10. selection      dynamic_truncate: token 预算 → 最终 ≤ FINAL_MAX_K
11. selection      expand_parent_context: 仅对最终 ≤5 个 chunk，docstore.siblings(parent_id)
                     SQLite 按 parent_id 批量取兄弟块
12. context        assemble_context + sufficiency_check
13. access         (可选) 相关但无权限提示：见 §6
14. generation     generate_answer_resilient + validate_citations
15. pipeline       组装 trace
```

关键不变量：**步骤 4–15 发出的所有请求数据量都有界**（top-N + 最终 parent 集合），与库中 chunk 总数无关。这是本设计的核心验收点。

## 5. 索引/写数据流

`sync_index` 写路径保持现有 `MirroredVectorStore` 双写语义，但内容分离：
- 写 Qdrant：向量 + 过滤元数据（去掉 text/terms）。
- 写 SQLite docstore：全文 + terms + parent 关系 + 元数据。
- `--rebuild-index` 同时重建两侧，作为两侧失配时的对账手段。

## 6. 权限/访问模型重做

当前 `load_access_chunks` 做四类全量 scroll（visible / permission_blocked / not_yet_effective / expired），其中三类是审计、一类是检索集。把权限拆成两个不同性质的关注点——load-bearing 的**强制执行** vs best-effort 的**审计/提示**——重做后：

- **visible（检索集）**：Qdrant 模式下不再单独物化，带 scope+effective filter 的召回直接给出可见子集。
- **not_yet_effective / expired（纯审计，与 query 无关）**：改用 Qdrant `count` API（带各自 filter），trace 里给数字而非全文清单；因与 query 无关可缓存。
- **permission_blocked（"相关但无权限"提示）**：从"扫全部 blocked chunk 的 terms 做词重合"改为 **query 驱动的二次召回**——发一次 effective-filtered、scope-excluded 的轻量 bm25 召回，取 top-k，减去可见 id，得到"相关但被拦截"的 id，只 hydrate 标题用于提示。语义更准、不拉全文。代价是一次额外的轻量 Qdrant 往返；默认关（`--blocked-hint`）。

### Backend 不对称（关键）
全量 visible 加载在两条路径里性质不同，`load_access_state` 必须**按 backend 分叉**：
- **Qdrant 模式**：有 ANN 索引，召回亚线性，不需要全量 visible → 召回 + hydrate，O(召回+选中)。
- **本地兜底模式**：dense 检索是 Python 对全部可见 chunk 暴力算 cosine（`dense_recall` 吃全量 `chunks`），离不开全量 visible → 维持全量加载，仍 O(全库)，但这是离线降级路径、语料小、可接受。

本地兜底路径的权限继续走现成 SQL 谓词（`scope_allowed()` + effective）+ FTS5 BM25，不变。

## 7. 模块拆分（`run_pipeline.py` → `rag/` 包）

`run_pipeline.py` 缩为薄入口（保留可执行文件名与 `python run_pipeline.py --query ...` 用法），逻辑进 `rag/` 包。依赖关系是无环 DAG（从叶到根）：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `rag/config.py` | env 加载、`resolve_*`、全部常量 | — |
| `rag/models.py` | Chunk / Candidate / ParentSection / SelectedEvidence | — |
| `rag/http.py` | request_json、call_with_retries、重试/超时/错误格式化 | config |
| `rag/chunking.py` | tokenize、vectorize、split_sections、chunk_parent、frontmatter、content_hash | config, models |
| `rag/embedding.py` | embedding 客户端、retrying embedding、identity | http, config, models |
| `rag/vectorstore/filters.py` | qdrant_access_filter 等四个 + access 语义 | config |
| `rag/vectorstore/qdrant.py` | QdrantVectorStore（纯向量+过滤索引） | http, config, models, filters |
| `rag/vectorstore/sqlite.py` | SQLite 存储（本地向量兜底 + docstore 物理层） | config, models |
| `rag/vectorstore/mirrored.py` | MirroredVectorStore 双写 | qdrant, sqlite |
| `rag/docstore.py` | hydrate 接口：id→文本 / parent_id→兄弟块（SQLite 实现） | vectorstore/sqlite, models |
| `rag/retrieval.py` | dense_recall、bm25_recall、rrf_fuse | models, vectorstore, embedding |
| `rag/rerank.py` | rerank、外部 reranker、score policy | models, http, config |
| `rag/selection.py` | mmr_select、dynamic_truncate、expand_parent_context | models, config, docstore |
| `rag/context.py` | assemble_context、sufficiency_check | models, config |
| `rag/generation.py` | LLM 客户端、resilient 生成、引用校验 | http, config, models, context |
| `rag/access.py` | 权限模型、count 审计、query 驱动 blocked 提示 | filters, retrieval, docstore |
| `rag/indexing.py` | read_documents、build_document_chunks、sync_index | chunking, embedding, vectorstore, docstore |
| `rag/pipeline.py` | run_query 编排 | 上面全部 |
| `run_pipeline.py` | argparse、main、eval 入口 | pipeline, config |

测试从单文件 `tests/test_production_pipeline.py` 拆成 `tests/test_<module>.py`，旧用例按模块归位。

## 8. 接口演进（允许破坏性变更）

利用"允许接口演进"，trace 中几个随全库语义的字段重新定义：
- `chunks_count` / `parents_count`：改为"本次召回+hydrate 的候选规模"，或取自 count API；不再是全库数。语义在 trace 里注明。
- `permission_filter.visible_chunks`：改为 count-API 数字，或移除。
- `permission_filter.rejected_chunks`：从全量清单改为审计计数 + blocked-relevant 子集。

CLI 基本不变（`--query` / `--scopes` / `--vector-backend` / `--rebuild-index`）；新增 `--blocked-hint` 开关控制 blocked-relevant 提示，**默认关**（见 §11 推荐决策 1）。

## 9. 测试策略

- **每模块单测**：docstore hydrate / 批量 sibling 取数、retrieval 融合、filters 构造、mmr 去重、truncate token 预算、access count + blocked 提示等。
- **核心回归守卫**：一个 `run_query` 集成测试，用 fake `request_json` 断言查询路径**不发出任何全库 scroll**、且请求数与候选数有界——直接守住"与语料规模解耦"这条验收线。
- 接口保留处沿用旧断言；接口演进处重写对应测试。

## 10. 迁移路径（增量，非大爆炸）

每阶段可独立验证：
- **P1 纯搬迁**：按 §7 抽模块，行为零改动，`run_pipeline.py` 暂留 re-export shim 保持旧 import 路径；旧测试全绿。
- **P2 引入 docstore**：新增 `docstore.py`；Qdrant 停存/停拉 text；pipeline 接 retrieve-then-hydrate；扩充测试。
- **P3 权限重做 + trace 演进**：count 审计 + query 驱动 blocked 提示；演进 trace 字段；重写受影响测试。
- **P4 收尾**：移除 shim，定稿包 API。

## 11. 风险与开放问题

**风险**
- SQLite docstore 与 Qdrant 一致性：写路径靠 MirroredVectorStore 双写，部分写失败靠 `--rebuild-index` 对账。
- Qdrant 不再存 text：若 SQLite 丢失/失配则无法 hydrate。缓解：rebuild 重建两侧。
- blocked-relevant 的额外往返：用开关控制。

**推荐决策（待评审确认）**
1. **blocked-relevant 提示默认关**，靠 `--blocked-hint` 显式开启。理由：它是合规 nice-to-have，但每查多一次 Qdrant 往返（跨境 1~2s），默认开会让我们刚省下的延迟又涨回去；需要合规演示时再开。
2. **`chunks_count` / `parents_count` 改语义保留**，取自 Qdrant count API（带 access filter），trace 注明含义为"可见集规模"而非召回数。理由：count 调用廉价、对可观测性有价值，移除是信息损失。
3. **docstore 复用现有 SQLite 表结构，但在 `parent_id` 上加索引**。理由：sibling 批量取数是新热路径，`parent_id` 索引把它从全表扫降到索引查；不另起表以免和现有 schema/迁移逻辑割裂。

---

下一步：评审通过后转 writing-plans 出分阶段实施计划。
