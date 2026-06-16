# Production RAG 工程化实操手册

`production_rag` 是 `mini_rag` 之后的工程化练习。它不试图一次做成完整平台，而是把一条 RAG 证据链拆成可运行、可观察、可排查、可评测的步骤：从带 metadata 的 Markdown 文档出发，经过权限过滤、混合检索、rerank、上下文组装、资料充足性判断、答案生成、引用校验，最后落到 trace 和 monitoring event。

跑完这个项目，你应该能回答这些更接近生产现场的问题：

- 文档变更后，索引什么时候更新、更新了什么、是否仍能复用旧 embedding？
- 权限受限、未生效或已过期的资料，会不会在生成前被挡住？
- 型号、编号、工单号这类短关键词，为什么不能只依赖向量召回？
- Top-K 里混进重复证据、弱相关证据或相邻 chunk 时，怎么选出可用 context？
- 资料不足、资料越权或问题超出知识库时，系统能不能拒答？
- 线上 bad case 出现后，trace 能不能定位到召回、融合、重排、截断、生成或引用校验中的哪一步？

## 快速导航

| 你想做什么 | 先看哪里 |
| --- | --- |
| 先把链路跑起来 | [1. 准备环境](#1-准备环境)、[2. 先跑一次完整工程链路](#2-先跑一次完整工程链路) |
| 只想离线排障 | [5. 本地兜底模式](#5-本地兜底模式) |
| 理解每一步为什么存在 | [4. 读懂代码里的 12 个环节](#4-读懂代码里的-12-个环节) |
| 看坏 case 怎么定位 | [3. 只看 trace，不急着看答案](#3-只看-trace不急着看答案)、[10. 常见报错](#10-常见报错) |
| 做最小回归评测 | [6. 做一次小型评测](#6-做一次小型评测) |
| 扩展数据或接本地 reranker | [9. 扩充数据集](#9-扩充数据集)、[第九步：rerank](#第九步rerank) |

## 运行模式总览

这个项目有两条常用运行路径：

| 模式 | 什么时候用 | 命令形态 | 实际使用 |
| --- | --- | --- | --- |
| 真实模型模式 | 主学习路径；观察真实 embedding、真实生成模型和 Qdrant 如何协同 | `python run_pipeline.py --query "你的问题" --rebuild-index` | 智谱 `embedding-3` + DeepSeek V4 Pro + Qdrant |
| 本地兜底模式 | API Key、网络或 Qdrant 不可用时排障；验证非模型链路 | `python run_pipeline.py --query "你的问题" --vector-backend local --rebuild-index` | 本地 hash embedding + 抽取式答案 + SQLite |

推荐先跑真实模型模式。本地兜底模式只在 API Key、网络或 Qdrant 暂时不可用时使用，不用它评估真实召回质量、真实生成质量或最终用户体验。

默认命令会自动探测已配置组件。LLM、embedding、Qdrant 或 rerank 不可用时，系统会按组件单独降级，并把原因写入 trace。向量库、embedding、生成模型和 reranker 不是一个总开关：哪一层配置好了就用哪一层，哪一层不可用就回到对应的可排障路径。

- `--vector-backend`：请求向量存储走 `qdrant` 还是 `local`；请求 `qdrant` 失败时会自动降级到 SQLite。

## 5 分钟跑通路径

下面的命令默认在 Git Bash 里执行。如果你已经有模型 API Key，并且本机可以启动 Docker，最短路径是：

```bash
cd ./production_rag
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、EMBEDDING_API_KEY 和 QDRANT_* 配置
docker compose up -d qdrant
python run_pipeline.py --query "跨境订单退款多久到账？" --rebuild-index
```

如果你只想先验证离线路径，不依赖模型服务和 Qdrant：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --vector-backend local --rebuild-index
```

## 项目地图

配套代码都在 `production_rag/` 下：

- `run_pipeline.py`：薄 CLI 入口，只负责参数解析、加载 `.env`、运行单条查询或 eval。
- `rag/`：生产风格 RAG 主链路包，按职责拆分检索、索引、上下文、生成和监控。
- `data/raw/`：样例客服知识库。
- `eval_cases.csv`：小型 golden set。
- `docker-compose.yml`：本地 Qdrant 启动配置。
- `.env.example`：DeepSeek / 智谱 / Qdrant 配置示例。
- `docs/`：模块化设计说明和执行计划留档，帮助理解当前拆分边界。
- `requirements-reranker.txt`：可选本地 reranker 服务依赖。
- `scripts/serve_bge_reranker.py`：`bge-reranker-v2-m3` 的 FlagEmbedding / Transformers HTTP 服务。
- `scripts/import_customer_support_dataset.py`：可选数据集导入脚本。
- `data/DATASET_SOURCES.md`：数据来源和取舍说明。
- `tests/`：权限、手动索引重建、Qdrant、模型请求、引用校验和监控测试。

`rag/` 包的核心模块如下：

| 模块 | 职责 |
| --- | --- |
| `config.py` | 环境变量、路径、默认阈值和 embedding / Qdrant 配置解析 |
| `models.py` | `ParentSection`、`Chunk`、`Candidate` 等数据结构 |
| `chunking.py` | frontmatter 解析、parent section / child chunk 切分、tokenize / vectorize |
| `embedding.py`、`http.py` | embedding / LLM / reranker 请求的 HTTP、重试和 fallback 原语 |
| `vectorstore/` | Qdrant、SQLite、本地镜像存储和 Qdrant filter 构造 |
| `docstore.py` | SQLite 文本仓库，按 chunk id hydrate，并按 parent id 取 sibling chunks |
| `indexing.py` | 原始文档到向量库和 docstore 的索引同步 |
| `retrieval.py`、`rerank.py`、`selection.py` | dense / BM25 召回、RRF、rerank、MMR、动态截断 |
| `context.py`、`generation.py` | context packet、资料充足性判断、答案生成和引用校验 |
| `access.py`、`monitoring.py`、`pipeline.py` | 权限提示、监控事件和查询编排 |

## 0. 项目边界与主链路

先把边界说清楚。

这不是一个完整线上 RAG 平台。它没有 Web 服务层、用户登录、租户体系、异步任务队列、人工审核台、灰度发布、SLO 告警和成本治理。

我们这次只做一件事：把一条生产风格的 RAG 主链路跑通，并把每一步的中间结果暴露出来。

你可以把它理解成 `mini_rag` 的工程化进阶版：

```text
Markdown 文档
  -> frontmatter metadata
  -> 权限 / 生效时间过滤
  -> 手动索引重建
  -> parent section / child chunk
  -> Qdrant 向量库
  -> dense recall
  -> BM25 recall
  -> RRF 融合
  -> rerank
  -> 语义去重
  -> 动态截断
  -> 资料充足性判断
  -> context packet
  -> 答案生成
  -> 引用校验
  -> trace + monitoring event
```

模块化后的 Qdrant 查询主路径是 retrieve-then-hydrate：

```text
run_pipeline.py CLI
  -> rag.pipeline.run_query()
  -> Qdrant dense + bm25 查询，权限和生效期 filter 下推到检索层
  -> Qdrant 只返回候选 chunk metadata，不从 payload 取 text / terms
  -> SqliteDocstore.hydrate(candidate_ids) 补回候选文本和 terms
  -> RRF / rerank / MMR
  -> SqliteDocstore.siblings(parent_ids) 补 parent 邻近 chunk
  -> 动态截断、context packet、生成、引用校验、trace
```

这样 Qdrant 路径不需要在每次查询时全量 scroll collection；文本正文留在本地 docstore，向量库负责候选召回和过滤。

它的重点不是“多堆几个模块”，而是让你看到：生产 RAG 的难点往往不在某一次模型调用，而在证据从原始文档到最终答案的整条链路。

## 1. 准备环境

进入练习目录：

```bash
cd ./production_rag
```

创建虚拟环境：

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

主链路代码只使用 Python 标准库，不需要额外安装运行依赖。后续命令默认都在已激活的 Git Bash 虚拟环境里执行。

主运行路径直接使用真实模型：

- LLM：DeepSeek V4 Pro，Anthropic-compatible Messages API。
- Embedding：智谱 `embedding-3`，OpenAI-compatible embeddings endpoint。
- Vector DB：Qdrant。

先复制环境变量示例：

```bash
cp .env.example .env
```

打开 `.env`，填入生成模型、embedding 模型和 Qdrant 配置：

```text
# 生成模型：DeepSeek V4 Pro
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com/anthropic
LLM_MODEL=deepseek-v4-pro
LLM_MAX_TOKENS=1200
# 兼容旧配置：
# ANTHROPIC_API_KEY=sk-...
# ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
# DEEPSEEK_API_KEY=sk-...
# DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic

# 向量模型：智谱 Embedding-3
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=1024
# 兼容旧配置：
# ZHIPU_API_KEY=...
# ZHIPU_EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# Optional: force index identity provider to local or external.
# EMBEDDING_PROVIDER=external

# 向量数据库
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=production_rag_chunks
# Qdrant Cloud 或开启认证的自托管 Qdrant 才需要：
# QDRANT_API_KEY=your-qdrant-api-key
# 兼容旧配置：
# VECTOR_DB_URL=http://localhost:6333
# VECTOR_DB_COLLECTION=production_rag_chunks
# VECTOR_DB_API_KEY=your-qdrant-api-key
```

如果同时设置了 `LLM_API_KEY`、`ANTHROPIC_API_KEY` 和 `DEEPSEEK_API_KEY`，脚本优先使用 `LLM_API_KEY`，其次使用 `ANTHROPIC_API_KEY`。

如果同时设置了 `EMBEDDING_API_KEY` 和 `ZHIPU_API_KEY`，脚本优先使用 `EMBEDDING_API_KEY`。

不要把真实 API Key 提交到仓库。

默认向量库走 Qdrant，所以你需要一个可访问的 Qdrant HTTP endpoint：

- 有 Docker：用 `docker compose up -d qdrant` 启动本地 Qdrant。
- 没有 Docker：使用 Qdrant Cloud、自托管 Qdrant，或后面“本地兜底模式”里的 SQLite 路径临时排障。

如果使用本地 Qdrant，先启动服务：

```bash
docker compose up -d qdrant
```

确认服务已经起来：

```bash
docker compose ps
```

也可以直接探测 HTTP API：

```bash
curl http://localhost:6333/collections
```

如果你使用 Qdrant Cloud 或自托管 Qdrant，把 `.env` 里的 endpoint 换成可访问地址：

```text
QDRANT_URL=https://your-cluster-url:6333
QDRANT_COLLECTION=production_rag_chunks
QDRANT_API_KEY=your-qdrant-api-key
```

注意：Qdrant Cloud 的 REST API 通常要带 `:6333`。如果只写 `https://your-cluster-url`，客户端会默认连 HTTPS 443，常见报错是 `[SSL: UNEXPECTED_EOF_WHILE_READING]`。

## 2. 先跑一次完整工程链路

第一次运行时加 `--rebuild-index`，脚本会读取 `data/raw/`，切 parent / chunk，调用智谱 `embedding-3`，写入 Qdrant，再调用 DeepSeek V4 Pro 生成答案：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --rebuild-index
```

你会看到几类信息：

- `Trace id`：这次请求的追踪 ID。
- `Monitoring event`：监控事件写入位置。
- `Selected evidence`：最终进入 context packet 的证据。
- `Answer`：真实模型基于 context packet 生成的答案。
- `Validation`：引用是否都能映射回本次 context packet。

这里先直接跑真实模型模式，是为了让你从一开始就观察真实 embedding、真实生成模型、Qdrant 检索和引用校验之间的配合。后面的本地模式只是兜底，不作为推荐学习主线。

如果真实模型服务或 Qdrant 暂时不可用，只想排查代码链路，可以后面再用本地兜底模式：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --vector-backend local --rebuild-index
```

`local` 不是推荐生产路径，只是为了测试和排障不依赖外部服务。

## 3. 只看 trace，不急着看答案

生产 RAG 的坏 case，不能只盯最终答案。先看完整 trace：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？" --trace-only
```

trace 里最值得先看这些字段：

- `index_sync`：哪些文档被重建，哪些文档被删除。
- `permission_filter`：当前用户能看哪些资料，哪些 chunk 被权限或生效时间挡掉。
- `dense_top`：向量召回结果。
- `bm25_top`：关键词召回结果。
- `rrf_top`：双路召回融合后的候选。
- `rerank_top`：重排后的候选和命中原因。
- `dedup_dropped`：哪些重复证据被丢掉。
- `truncation`：为什么最终只保留这些证据。
- `context_packet`：真正交给答案生成器的证据包。
- `validation`：答案里的引用是否有效。

默认 trace 不会额外搜索用户无权访问的标题。如果你想在权限拒答场景里看到“哪些被挡住的标题和 query 有关”，显式加 `--blocked-hint`：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --trace-only --blocked-hint
```

这个开关只用于排障；默认关闭，避免每次查询都额外探测受限资料。

这一步对应真实线上排障：当用户说“答错了”，你要先判断是没召回、召回了但没排上去、排上去了但被截掉、进入上下文了但答案没用，还是引用校验没兜住。

如果需要把完整 trace 保存下来：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？" --save-trace
```

## 4. 读懂代码里的 12 个环节

先从 `rag/pipeline.py` 的 `run_query()` 看整体编排，再按下面顺序跳到对应模块。`run_pipeline.py` 现在只是 CLI 包装层，不再 re-export 包内实现。

### 第一步：读取带 metadata 的文档

`sync_index()` 会遍历 `data/raw/` 下的 Markdown 文件，并解析 frontmatter。

这些 metadata 不只是装饰字段。生产 RAG 里，文档从一开始就应该带上可检索、可过滤、可治理的信息，例如：

- `doc_id`：稳定文档 ID；
- `title`：文档标题；
- `permission_scope`：权限范围；
- `effective_from` / `effective_to`：生效时间；
- `source_path`：来源路径。

`mini_rag` 里只要读到文本就能继续。生产链路里，如果 metadata 丢了，后面的权限过滤、增量更新、溯源和排障都会变得很脆。

### 第二步：parent section / child chunk

`split_sections()` 会先把文档按二级标题切成 parent section，`chunk_parent()` 再把 parent 切成 child chunk。

这样做是为了同时保留两件事：

- child chunk 足够小，方便召回和重排；
- parent / title path 仍然能告诉系统证据属于哪个业务章节。

代码里的默认切块参数是：

```text
CHUNK_CHARS = 280
OVERLAP_CHARS = 60
```

这不是“最佳参数”，只是一个可观察起点。生产系统里，chunk size 应该由评测集、文档形态和上下文预算共同决定。

### 第三步：手动重建索引

索引不是每次查询都会自动同步 raw 文档。默认查询只加载已经存在的索引；只有显式加 `--rebuild-index` 时，脚本才会读取 `data/raw/`、重新切 chunk、计算 embedding 并写入 Qdrant 或 SQLite。embedding identity 会区分本地/外部方式和模型名，例如 `local:local-hash-embedding`、`local:nomic-embed-text` 或 `external:embedding-3`。

本地 SQLite 存储会按 embedding identity 派生独立文件名，避免真实 embedding 索引和 hash fallback 索引互相覆盖。

生产系统里，索引不是临时缓存，而是系统状态。你要知道：

- 哪些文档变了；
- 哪些文档没变，所以复用旧 embedding；
- 哪些文档从原始目录里消失了，所以要从索引删除；
- 当前索引用的是哪个 embedding 模型。

加 `--rebuild-index` 会手动重建整个 collection / SQLite 索引：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --rebuild-index
```

### 第四步：权限和生效时间过滤

权限和生效时间过滤在存储层完成：Qdrant 查询使用 payload filter，SQLite 查询使用表字段条件。`filter_chunks_for_access()` 仍保留为小规模测试和对照用的纯函数。

默认查询 scope 是：

```text
internal,public
```

也就是说，普通用户默认看不到这些受限资料：

- `partner_support`
- `finance_restricted`
- `security_restricted`

试一个财务受限问题：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --trace-only
```

默认权限下，系统应该拒答或提示权限不足。再带上财务 scope：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --scopes finance_restricted --trace-only
```

这一步很关键：权限过滤必须发生在生成答案之前，不能指望 Prompt 告诉模型“不要泄露”就完事。

### 第五步：向量库写入和查询

默认后端是 Qdrant：

```text
DEFAULT_VECTOR_BACKEND = "qdrant"
QDRANT_URL = http://localhost:6333
QDRANT_COLLECTION = production_rag_chunks
```

`QdrantVectorStore` 负责创建 collection、写入 points、查询 points，并为 `doc_id` 创建 payload index。每个 point 会同时写入 named dense vector 和 `bm25` sparse vector；payload 只保留检索和审计需要的 metadata，不再存正文 `text` 或 `terms`。手动重建需要按 `doc_id` 删除旧 chunk，所以这个 payload index 不是可有可无。

如果看到：

```text
Index required but not found for "doc_id" of one of the following types: [keyword]
```

说明 collection 里缺少 `doc_id` payload index。当前代码启动时会自动创建；如果 collection 里已经混了旧实验数据，可以加 `--rebuild-index` 重建。

### 第六步：dense recall

`dense_recall()` 负责语义召回。本地 SQLite 模式会在已过滤的可见 chunk 上计算；Qdrant 模式会把权限和生效期 filter 一起发给向量库，并且 dense 查询结果不再请求返回 vector。

主路径下，文档和 query embedding 都走智谱 `embedding-3`。只有在本地兜底模式里，才会用 hash embedding 代替真实 embedding 做离线排障。

dense recall 擅长处理同义表达。比如用户问：

```bash
python run_pipeline.py --query "跨境退款一般几天能回到卡里？" --trace-only
```

即使文档不完全使用同一句话，也可能命中退款时效相关资料。

但 dense recall 对型号、编号、工单号、短关键词不总是稳定，所以还需要 BM25。

### 第七步：BM25 recall

Qdrant 模式下，关键词召回走 Qdrant 的 `bm25` sparse vector，并在 Qdrant 查询里同时应用权限和生效期 filter。只有 `--vector-backend local` 时才使用 SQLite FTS5 的 `bm25()`。`bm25_recall()` 是本地简易实现，作为纯函数对照。

它对这类问题尤其有用：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？" --trace-only
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --trace-only
```

如果只靠向量，`SKU-A17`、`FR-21` 这类 token 很容易被语义相似度稀释。BM25 的作用就是把这些“精确词”补回来。

### 第八步：RRF 融合

`rrf_fuse()` 会把 dense 和 BM25 两路候选融合。

RRF 的直觉很简单：一个 chunk 如果在两条路里都排得靠前，它就更值得上浮。这样可以避免系统过度相信单一路径。

观察 trace 里的：

```text
dense_top
bm25_top
rrf_top
```

你会看到哪些候选是语义召回来的，哪些候选是关键词召回来的，哪些候选因为双路都命中而上浮。

### 第九步：rerank

`rerank()` 会在融合候选上调用已配置的 rerank 模型再做一次重排。

如果没有配置 rerank 模型，流程会跳过模型 rerank，直接把 RRF 融合后的候选交给后续 MMR 和动态截断。此时 RRF 只被当作同一 query 内的排序信号，不被当作相关性置信分。trace 中会记录 `reranker.mode=skipped`、`reranker.reason=not_configured`、`reranker.score_policy=rrf_only`，便于区分“没有模型重排”和“模型重排成功”。

有条件时，也可以接一个本地 CPU reranker 服务。这个项目使用 `FlagEmbedding` 或 `Transformers` 把 `BAAI/bge-reranker-v2-m3` 包成 HTTP 服务。

先安装可选依赖：

```bash
python -m pip install -r requirements-reranker.txt
```

启动本地 reranker 服务：

```bash
RERANKER_BACKEND=flagembedding python scripts/serve_bge_reranker.py
```

如果本机 `FlagEmbedding` 安装不顺，也可以切到 Transformers 后端：

```bash
RERANKER_BACKEND=transformers python scripts/serve_bge_reranker.py
```

第一次启动时，`FlagEmbedding` / `Transformers` 会从 Hugging Face 下载 `BAAI/bge-reranker-v2-m3`。

如果无法连上`https://huggingface.co/`，可以将`HF_ENDPOINT`设置成`https://hf-mirror.com`。

如果看到类似下面的错误：

```text
OSError: We couldn't connect to 'https://hf-mirror.com' to load the files, and couldn't find them in the cached files.
```

说明当前 Python 进程连不上 `HF_ENDPOINT` 指向的镜像，而且本地缓存里也没有模型。可以换成可访问的 Hugging Face endpoint，或者先把模型下载到本地目录：

```bash
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir D:/models/bge-reranker-v2-m3
```

然后启动服务时指定本地目录：

```bash
RERANKER_MODEL_DIR=D:/models/bge-reranker-v2-m3 RERANKER_BACKEND=flagembedding python scripts/serve_bge_reranker.py
```

`scripts/serve_bge_reranker.py` 启动时会读取当前目录下的 `.env`。如果报错里仍然显示正在加载 `BAAI/bge-reranker-v2-m3`，说明服务没有读到 `RERANKER_MODEL_DIR`；请确认变量写在 `production_rag/.env`，或者在同一个终端里先设置环境变量再启动服务。

然后在 `.env` 里接入这个服务：

```text
RERANKER_PROVIDER=flagembedding
RERANKER_URL=http://127.0.0.1:8008/rerank
RERANKER_MODEL=bge-reranker-v2-m3
# 如果已经预下载模型，也可以保留这个本地目录配置，方便下次启动服务。
# RERANKER_MODEL_DIR=D:/models/bge-reranker-v2-m3
RERANKER_TIMEOUT_SECONDS=30
```

如果已配置的 rerank 服务不可用，production_rag 会跳过模型 rerank，继续使用 RRF 融合顺序，并在 trace 的 `reranker.mode=skipped`、`reranker.reason=reranker_error`、`reranker.score_policy=rrf_only` 和 `reranker.error` 里记录原因。这样练习不会因为本地 reranker 没启动就中断，也不会混用不同打分体系。

真实生产里，这一层通常会替换成专门的 rerank 模型，但它解决的问题不变：召回阶段先多拿候选，rerank 阶段再更细地判断证据相关性。

### 第十步：MMR、多样性和动态截断

`mmr_select()` 会在 rerank 后做一层轻量 MMR 选择。

它同时考虑两件事：

- 相关性：有 rerank 模型时优先保留 rerank 分高的证据；跳过 rerank 时只按 RRF 排序信号保留候选；
- 多样性：避免最终上下文里全是同一个 parent 或同一段话的近重复 chunk。

近重复候选会进入 trace 的 `dedup_dropped`，并标记 `reason=near_duplicate`。

`dynamic_truncate()` 会根据几个因素决定最终证据包：

- 最低 rerank 分数，未配置 rerank 模型时不使用绝对分数门槛；
- 分数断崖，未配置 rerank 模型时不使用 RRF gap 做断崖截断；
- 最大证据条数；
- context token 预算。

这比固定 `top_k=5` 更接近生产思路。Top-K 太小会漏证据，太大又会把噪声塞进上下文。动态截断的目标，是让上下文里留下“够用但不乱”的证据。

### 第十一步：资料充足性判断

`sufficiency_check()` 会在生成答案之前判断资料是否足够。

它会处理几类情况：

- 问题明显超出业务域，比如天气、股票、新闻；
- 没有选中证据；
- 最相关资料被权限挡住；
- query 和证据 overlap 太低；
- 跳过 rerank 时，BM25 召回到的证据没有覆盖 query terms。

试一个知识库外的问题：

```bash
python run_pipeline.py --query "今天北京天气怎么样？" --trace-only
```

理想结果不是硬答天气，而是拒答。

RAG 的生产能力不只是“答得上”，还包括“知道什么时候不该答”。

### 第十二步：context packet、答案和引用校验

`assemble_context()` 会把最终证据变成 context packet。每条证据都有：

- `citation_id`，例如 `E1`；
- `doc_id`；
- `title_path`；
- `source_path`；
- `version`；
- `rerank_score`；
- `mmr_score`；
- `evidence_role`；
- `expanded_from_chunk_ids`；
- `text`。

这里还做了一层 parent expansion：rerank 和 MMR 选中的是 child chunk，但进入 context packet 时会把同一个 parent section 下的 sibling chunk 一并带上，避免答案所需的定义、例外或处理步骤刚好落在相邻 chunk 里。

主路径下，`generate_answer_with_llm()` 会把 context packet 组装成 Prompt，调用配置好的 Anthropic-compatible Messages API，并要求答案保留 `[E1]` 这样的引用。默认配置使用 DeepSeek V4 Pro，也可以替换成其他兼容服务。

本地兜底模式下，`generate_answer()` 才会做抽取式回答。它适合排查检索和上下文链路，不适合代表真实模型效果。

最后 `validate_citations()` 会检查答案里的引用是否都来自本次 context packet。引用校验不能证明答案完全正确，但它能挡住一类很危险的问题：模型编出不存在的证据编号。

## 5. 本地兜底模式

本地兜底只用于临时排障：比如没有 API Key、Qdrant 没启动、网络不可用，或者你只想快速检查权限过滤、BM25、RRF、去重、动态截断和引用校验这些非模型环节。

降级是按组件发生的：

- 没有 LLM 配置时，答案生成会回到抽取式回答；
- 没有 embedding 配置或 embedding 请求失败时，会回到本地 hash embedding；
- Qdrant 未配置、不可达或显式指定 `--vector-backend local` 时，会使用 SQLite 向量存储；
- reranker 未配置或服务报错时，会跳过模型 rerank，继续使用 RRF 排序信号。

如果你想跑完整离线路径，显式使用 SQLite 后端：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --vector-backend local --rebuild-index
```

这条路径会使用 hash embedding 和抽取式答案。它能帮你定位链路问题，但不要用它评估真实 embedding、真实生成质量或最终用户体验。

## 6. 做一次小型评测

运行内置评测集：

```bash
python run_pipeline.py --eval --rebuild-index
```

评测用例在 `eval_cases.csv` 里，当前覆盖：

- 跨境订单退款时效；
- `SKU-A17` 无理由退货；
- 预售商品是否能按 48 小时规则催单；
- 会员积分提现；
- 资料外问题拒答；
- 公开客服规则响应时效。
- 电子发票处理；
- 保修拒保材料；
- 商品召回通知核对；
- 平台商家售后边界。

也可以逐条手工看 trace：

```bash
python run_pipeline.py --query "会员积分可以提现吗？" --trace-only
python run_pipeline.py --query "今天北京天气怎么样？" --trace-only
```

记录时不要只写“答案对 / 错”，建议至少记这几列：

| 问题 | dense 是否命中 | BM25 是否命中 | rerank 是否排前 | context 是否保留 | 答案是否忠实 | 引用是否有效 |
| --- | --- | --- | --- | --- | --- | --- |
| 跨境订单退款多久到账？ | 是 / 否 | 是 / 否 | 是 / 否 | 是 / 否 | 是 / 否 | 是 / 否 |
| SKU-A17 是否支持无理由退货？ | 是 / 否 | 是 / 否 | 是 / 否 | 是 / 否 | 是 / 否 | 是 / 否 |
| 今天北京天气怎么样？ | 不适用 | 不适用 | 不适用 | 不适用 | 应拒答 | 不适用 |

生产 RAG 评测要拆链路看。最终答案只是结果，trace 才是诊断。

如果改了代码或 README 里的行为描述，建议同时跑一次回归测试，确认文档和实现没有脱节：

```bash
python -m unittest discover -s tests
```

## 7. 两个必须亲手试的实验

### 实验一：验证权限拒答

默认权限下问财务受限问题：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --trace-only --blocked-hint
```

再带财务权限：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --scopes finance_restricted --trace-only --blocked-hint
```

观察 `permission_filter.rejected_chunks`、`permission_filter.blocked_matches` 和 `context_packet.sufficiency.reason`。

如果系统在无权限时仍然给出财务处理细节，就说明权限过滤位置错了，或者受限文档的 metadata 没有写对。

## 8. 线上监控事件

每次查询默认会追加一行 JSONL monitoring event：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？"
```

默认路径类似：

```text
C:/Users/<you>/AppData/Local/Temp/production_rag_runtime/traces/online_metrics.jsonl
```

也可以用环境变量覆盖运行目录：

```bash
RAG_RUNTIME_DIR=C:/tmp/production_rag_runtime python run_pipeline.py --query "SKU-A17 是否支持无理由退货？"
```

monitoring event 会记录：

- `trace_id`、`latency_ms`、`status`；
- `vector_backend`、`embedding_model`、`embedding_identity`、`qdrant_collection`；
- `query_hash` 和 `query_chars`，默认不记录完整 query；
- dense / BM25 / rerank 命中数量；
- MMR 近重复丢弃数量；
- 最终 selected 数量；
- context token 估算；
- 每个阶段的 `stage_latencies_ms`；
- `selection_strategy`；
- `reranker_mode`、`reranker_model`、`reranker_score_policy`、`reranker_fallback_used`；没有配置或服务报错时，`reranker_mode=skipped`，具体原因见 trace 的 `reranker.reason` / `reranker.error`；
- `sufficiency_reason`、`permission_denied`；
- `citation_valid`、`missing_citation_count`；
- 最终选中的 `selected_doc_ids`。

如果只想临时跑命令，不写 monitoring event：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？" --no-monitoring
```

生产系统不应该把完整用户问题随手写进监控日志。这个项目默认写 `query_hash`，就是为了保留排障线索，同时减少敏感信息暴露。

## 9. 扩充数据集

`data/raw/` 已经内置中文为主的客服知识库样例，覆盖：

- 配送异常；
- 保修维修；
- 发票税务；
- 促销积分；
- 商品召回；
- 平台商家售后边界；
- 公开服务规则；
- 商家协同；
- 财务对账；
- 账号风控。

英文只保留 3 篇 BrownBox casebook，用于测试跨语言 query、英文客服文本和中文业务资料混排时的召回表现。

如果要额外导入英文开源客服数据行，可以使用 MIT 许可的 Hugging Face 数据集 `rjac/e-commerce-customer-support-qa`：

```bash
python -m pip install -r requirements-dataset.txt
python scripts/import_customer_support_dataset.py --limit 120 --rows-per-doc 12
```

这一步是可选扩展，不是主实验前置。脚本会先尝试 `datasets.load_dataset()`；如果 parquet / xet 下载失败，会自动降级到 Hugging Face Dataset Viewer 的 rows API。

如果 Git Bash 不能访问 Hugging Face 或 `datasets-server.huggingface.co`，先检查网络、代理、`HTTPS_PROXY` / `HTTP_PROXY` 或 `HF_ENDPOINT`。也可以直接跳过，因为内置中文种子文档已经足够跑主链路。

数据源取舍见 `data/DATASET_SOURCES.md`。

## 10. 常见报错

### Qdrant 连接不上

先确认本地服务是否启动：

```bash
docker compose ps
curl http://localhost:6333/collections
```

如果没有 Docker，也没有可用的 Qdrant endpoint，可以临时使用本地兜底模式：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --vector-backend local --rebuild-index
```

### Qdrant Cloud 报 SSL EOF

检查 `QDRANT_URL` 是否带了 REST API 端口：

```text
QDRANT_URL=https://your-cluster-url:6333
```

不要只写：

```text
QDRANT_URL=https://your-cluster-url
```

### 真实模型模式提示缺少 API Key

生成端需要设置其中一个：

```text
LLM_API_KEY
ANTHROPIC_API_KEY
DEEPSEEK_API_KEY
```

embedding 端需要设置其中一个：

```text
EMBEDDING_API_KEY
ZHIPU_API_KEY
```

### 改了文档但答案没变

确认这次查询是否带了 `--rebuild-index`。默认查询不会扫描 `data/raw/`，所以改文档后答案不变是预期行为。

手动重建：

```bash
python run_pipeline.py --query "你的问题" --rebuild-index
```

### 检索结果看起来不相关

先用 `--trace-only` 看链路，而不是直接改 Prompt：

- `dense_top` 是否命中语义相关证据；
- `bm25_top` 是否命中关键词证据；
- `rrf_top` 是否把双路候选融合上来；
- `rerank_top` 是否把正确证据排前；
- `dedup_dropped` 是否误删了关键证据；
- `truncation` 是否把证据截掉；
- `sufficiency.reason` 是否误判资料不足；
- `validation` 是否发现无效引用。

## 11. 你应该带走的感觉

`mini_rag` 让你看到 RAG 的最小闭环。

`production_rag` 想让你看到的是：生产 RAG 是一条可治理的证据链。

```text
原始资料
  -> 带 metadata 的可治理资料
  -> 可手动重建的索引
  -> 受权限约束的候选池
  -> dense + BM25 的召回集合
  -> rerank 后的相关证据
  -> 去重和截断后的 context packet
  -> 资料充足时才生成的答案
  -> 可校验引用
  -> 可回放 trace 和监控事件
```

当 Production RAG 答错时，不要急着调模型温度，也不要第一反应改 Prompt。先问：

```text
文档 metadata 对吗？
索引更新了吗？
权限过滤位置对吗？
dense 找到了吗？
BM25 补回来了吗？
RRF 融合后排前了吗？
rerank 判断对了吗？
重复证据被正确丢掉了吗？
动态截断有没有截掉关键证据？
资料不足时拒答了吗？
答案引用能映射回 context packet 吗？
monitoring event 能支持线上排查吗？
```

这就是从 demo 走向工程系统时最重要的变化：不只是让模型回答，而是让每个答案都有证据、有边界、能追踪、能复盘。
