# Production RAG 工程化实操手册

跑完 `mini_rag` 以后，你已经能看到一条最小 RAG 链路：读文档、切 chunk、做 embedding、召回、组装 Prompt、生成带引用的答案。

但只要把这条链路放进真实业务，新的问题会马上冒出来：文档怎么增量更新？权限受限资料会不会被召回？型号、编号、工单号这种短关键词靠向量能不能找准？Top-K 里混了重复证据怎么办？资料不足时系统能不能拒答？线上 bad case 发生后，能不能知道它坏在哪一步？

这份手册的目标，是把 `mini_rag` 的透明骨架升级成一条可观察、可排查、接近生产形态的 RAG 工程链路。它仍然不是完整线上系统，但它会把真实项目里绕不开的关键模块都跑给你看：

- metadata 和权限过滤；
- 增量索引同步；
- parent section / child chunk；
- Qdrant 向量数据库；
- dense recall + BM25 双路召回；
- RRF 融合；
- rerank；
- 语义去重；
- 动态截断；
- 资料充足性判断；
- context packet；
- 答案生成；
- 引用校验；
- trace 和 monitoring event。

## 运行模式总览

这个项目按用途分成两种模式：

| 模式 | 推荐用途 | 命令形态 | 使用的模型与存储 |
| --- | --- | --- | --- |
| 真实模型模式 | 主学习路径，观察真实 embedding、真实生成模型和 Qdrant 如何协同 | `python run_pipeline.py --query "你的问题" --rebuild-index` | 真实 embedding provider + 真实 LLM + Qdrant |
| 本地兜底模式 | 临时排障，验证非模型链路是否跑通 | `python run_pipeline.py --query "你的问题" --vector-backend local --rebuild-index` | 本地 hash embedding + 抽取式答案 + SQLite |

推荐先跑真实模型模式。本地兜底模式只在 API Key、网络或 Qdrant 暂时不可用时使用，不用它评估真实召回质量、真实生成质量或最终用户体验。

默认命令会自动探测已配置的真实组件；LLM、embedding、Qdrant 或 rerank 不可用时，会分别降级并写入 trace。

- `--vector-backend`：请求向量存储走 `qdrant` 还是 `local`；请求 `qdrant` 失败时会自动降级到 SQLite。

配套代码在本目录下：

- `run_pipeline.py`：完整 production_rag 练习主链路。
- `data/raw/`：样例客服知识库。
- `eval_cases.csv`：小型 golden set。
- `docker-compose.yml`：本地 Qdrant 启动配置。
- `.env.example`：DeepSeek / 智谱 / Qdrant 配置示例。
- `requirements-reranker.txt`：可选本地 reranker 服务依赖。
- `scripts/serve_bge_reranker.py`：`bge-reranker-v2-m3` 的 FlagEmbedding / Transformers HTTP 服务。
- `scripts/import_customer_support_dataset.py`：可选数据集导入脚本。
- `data/DATASET_SOURCES.md`：数据来源和取舍说明。
- `tests/`：权限、增量同步、Qdrant、模型请求、引用校验和监控测试。

## 0. 先说做成什么样

先把边界说清楚。

这不是一个完整线上 RAG 平台。它没有 Web 服务层、用户登录、租户体系、异步任务队列、人工审核台、灰度发布、SLO 告警和成本治理。

我们这次只做一件事：把一条生产风格的 RAG 主链路跑通，并把每一步的中间结果暴露出来。

你可以把它理解成 `mini_rag` 的工程化进阶版：

```text
Markdown 文档
  -> frontmatter metadata
  -> 权限 / 生效时间过滤
  -> 增量同步
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

它的重点不是“多堆几个模块”，而是让你看到：生产 RAG 的难点往往不在某一次模型调用，而在证据从原始文档到最终答案的整条链路。

## 1. 准备环境

在 Windows Git Bash 里进入练习目录：

```bash
cd ./production_rag
```

创建虚拟环境：

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

主链路代码只使用 Python 标准库，不需要额外安装运行依赖。主运行路径直接使用真实模型：

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
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
LLM_MODEL=deepseek-v4-pro
LLM_MAX_TOKENS=1200

# 向量模型：智谱 Embedding-3
ZHIPU_API_KEY=...
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=1024
# Optional: force index identity provider to local or external.
# EMBEDDING_PROVIDER=external

# 向量数据库
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=production_rag_chunks
# Qdrant Cloud 或开启认证的自托管 Qdrant 才需要：
# QDRANT_API_KEY=your-qdrant-api-key
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

这一步对应真实线上排障：当用户说“答错了”，你要先判断是没召回、召回了但没排上去、排上去了但被截掉、进入上下文了但答案没用，还是引用校验没兜住。

如果需要把完整 trace 保存下来：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？" --save-trace
```

## 4. 读懂代码里的 12 个环节

打开 `run_pipeline.py`，按下面顺序看。

### 第一步：读取带 metadata 的文档

`read_documents()` 会读取 `data/raw/` 下的 Markdown 文件，并解析 frontmatter。

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

### 第三步：增量同步索引

`sync_index()` 会先确定实际 embedding identity，再比较文档内容 hash，只重建发生变化的文档。identity 会区分本地/外部方式和模型名，例如 `local:local-hash-embedding`、`local:nomic-embed-text` 或 `external:embedding-3`。

本地 SQLite 存储会按 embedding identity 派生独立文件名，避免真实 embedding 索引和 hash fallback 索引互相覆盖。

这解决的是 `mini_rag` 里很容易遇到的问题：文档改了，但索引还是旧的。

生产系统里，索引不是临时缓存，而是系统状态。你要知道：

- 哪些文档变了；
- 哪些文档没变，所以复用旧 embedding；
- 哪些文档从原始目录里消失了，所以要从索引删除；
- 当前索引用的是哪个 embedding 模型。

加 `--rebuild-index` 会强制重建整个 collection：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --rebuild-index
```

### 第四步：权限和生效时间过滤

`filter_chunks_for_access()` 会根据当前用户的 `--scopes` 过滤 chunk。

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

`QdrantVectorStore` 负责创建 collection、写入 points、查询 points，并为 `doc_id` 创建 payload index。增量同步需要按 `doc_id` 删除旧 chunk，所以这个 payload index 不是可有可无。

如果看到：

```text
Index required but not found for "doc_id" of one of the following types: [keyword]
```

说明 collection 里缺少 `doc_id` payload index。当前代码启动时会自动创建；如果 collection 里已经混了旧实验数据，可以加 `--rebuild-index` 重建。

### 第六步：dense recall

`dense_recall()` 负责语义召回。

主路径下，文档和 query embedding 都走智谱 `embedding-3`。只有在本地兜底模式里，才会用 hash embedding 代替真实 embedding 做离线排障。

dense recall 擅长处理同义表达。比如用户问：

```bash
python run_pipeline.py --query "跨境退款一般几天能回到卡里？" --trace-only
```

即使文档不完全使用同一句话，也可能命中退款时效相关资料。

但 dense recall 对型号、编号、工单号、短关键词不总是稳定，所以还需要 BM25。

### 第七步：BM25 recall

`bm25_recall()` 负责关键词召回。

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

有条件时，也可以接一个本地 CPU reranker 服务。这个项目推荐用 `FlagEmbedding` 或 `Transformers` 把 `BAAI/bge-reranker-v2-m3` 包成 HTTP 服务，而不是假设 Ollama 原生支持 rerank endpoint。

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

如果已配置的 rerank 服务不可用，production_rag 不会回退到规则 rerank；它会跳过模型 rerank，继续使用 RRF 融合顺序，并在 trace 的 `reranker.mode=skipped`、`reranker.reason=reranker_error`、`reranker.score_policy=rrf_only` 和 `reranker.error` 里记录原因。这样练习不会因为本地 reranker 没启动就中断，也不会混用不同打分体系。

真实生产里，这一层通常会替换成专门的 rerank 模型，但它解决的问题不变：召回阶段先多拿候选，rerank 阶段再更细地判断证据相关性。

### 第十步：MMR、多样性和动态截断

`mmr_select()` 会在 rerank 后做一层轻量 MMR 选择。

它同时考虑两件事：

- 相关性：有 rerank 模型时优先保留 rerank 分高的证据；跳过 rerank 时只按 RRF 排序信号保留候选；
- 多样性：避免最终上下文里全是同一个 parent 或同一段话的近重复 chunk。

近重复候选会进入 trace 的 `dedup_dropped`，并标记 `reason=near_duplicate`。

`dynamic_truncate()` 会根据几个因素决定最终证据包：

- 最低 rerank 分数；未配置 rerank 模型时不使用绝对分数门槛；
- 分数断崖；未配置 rerank 模型时不使用 RRF gap 做断崖截断；
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

本地模式只用于临时排障：比如没有 API Key、Qdrant 没启动、网络不可用，或者你只想快速检查权限过滤、BM25、RRF、去重、动态截断和引用校验这些非模型环节。

只是不调用真实模型，但仍然使用 Qdrant：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --rebuild-index
```

既不调用真实模型，也不用 Qdrant，改用 SQLite 兜底向量存储：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --vector-backend local --rebuild-index
```

本地模式会使用 hash embedding 和抽取式答案。它能帮你定位链路问题，但不要用它评估真实 embedding、真实生成质量或最终用户体验。

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

## 7. 三个必须亲手试的实验

### 实验一：改文档，观察增量同步

修改 `data/raw/refund_policy.md` 里的退款时效描述。

然后运行：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --trace-only
```

观察 `index_sync.changed_docs`。如果只改了这一篇，理论上只应该重建对应文档。

再运行一次同样命令：

```bash
python run_pipeline.py --query "跨境订单退款多久到账？" --trace-only
```

第二次 `changed_docs` 应该为空，因为索引已经是最新状态。

这个实验能帮你建立一个重要直觉：生产 RAG 的知识更新，不是“重新跑全量索引”这么粗糙。你要能知道变更范围，并尽量只更新必要部分。

### 实验二：比较 dense 和 BM25 的分工

先问一个偏语义表达的问题：

```bash
python run_pipeline.py --query "跨境退款一般几天能回到卡里？" --trace-only
```

再问一个偏精确标识的问题：

```bash
python run_pipeline.py --query "SKU-A17 是否支持无理由退货？" --trace-only
```

对比 `dense_top` 和 `bm25_top`。你会看到两条召回路径的强项不一样。

这就是为什么生产 RAG 常用混合检索：向量负责语义，BM25 负责精确词，RRF 负责把两边证据合到一个候选池里。

### 实验三：验证权限拒答

默认权限下问财务受限问题：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --trace-only
```

再带财务权限：

```bash
python run_pipeline.py --query "FR-21 差异工单怎么处理？" --scopes finance_restricted --trace-only
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
C:\Users\<you>\AppData\Local\Temp\production_rag_runtime\traces\online_metrics.jsonl
```

也可以用环境变量覆盖运行目录：

```text
RAG_RUNTIME_DIR=C:\tmp\production_rag_runtime
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
pip install -r requirements-dataset.txt
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

先看 trace 里的 `index_sync.changed_docs`。如果为空，说明脚本认为文档内容和 embedding 模型没有变化。

也可以强制重建：

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
  -> 可增量同步的索引
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
