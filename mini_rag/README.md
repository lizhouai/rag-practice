# RAG 从零搭建实操手册

学 RAG 的理论时，很容易有一种错觉：只要知道“切块、向量化、召回、塞进 Prompt”，好像就会了。

真正动手时才会发现，RAG 不是一个概念，而是一条链路。每一环都可能把答案带偏：文档没读进来、chunk 切坏了、向量召回没命中、Top-K 里混了噪声、Prompt 没有约束引用、模型又开始凭常识补全。

这份手册的目标很朴素：不用 LangChain，不上向量数据库，先用最少代码把一条完整 RAG 链路跑通。你跑完以后，应该能清楚看到：

- 文档是怎么变成 chunk 的；
- chunk 是怎么变成 embedding 的；
- 用户问题是怎么找回相关资料的；
- 检索结果是怎么组装进 Prompt 的；
- 模型答案是怎么带引用输出的；
- 为什么 RAG 的坏 case 要拆链路定位。

配套代码在本目录下：

- `mini_rag.py`：最小可运行 RAG 链路。
- `sample_docs/`：练习用的小型业务知识库。
- `eval_set.md`：人工检查用的小评测集。
- `.env.example`：环境变量示例。
- `requirements.txt`：练习所需 Python 依赖。

## 0. 这次先不做什么

先把预期说清楚。

这不是生产级 RAG 系统。它没有权限系统、增量同步、向量数据库、BM25、RRF、Rerank、动态截断、去重、引用校验和线上监控。

我们这次只做一件事：把 RAG 的主链路完整跑通，并把每一步露出来。

你可以把它理解成 RAG 的“透明骨架版”：

```text
Markdown 文档
  -> 清洗
  -> 切块
  -> embedding
  -> 本地 JSON 索引
  -> 问题 embedding
  -> 余弦相似度 Top-K
  -> 上下文组装
  -> LLM API 生成答案
  -> 带引用输出
```

后面再把这条骨架升级成工程系统，才有地方下手。

## 1. 准备环境

在 Windows Git Bash 里进入练习目录：

```bash
cd ./mini_rag
```

创建虚拟环境：

```bash
python -m venv .venv
source .venv/Scripts/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

复制环境变量示例：

```bash
cp .env.example .env
```

打开 `.env`，填入你的 API Key：

```text
# 生成模型：DeepSeek V4 Pro（Anthropic-compatible Messages API）
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
LLM_MODEL=deepseek-v4-pro
LLM_API_STYLE=anthropic
LLM_MAX_TOKENS=1000

# 向量模型：智谱 Embedding-3
ZHIPU_API_KEY=sk-...
EMBEDDING_MODEL=embedding-3
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

这份练习默认把生成端接到 DeepSeek V4 Pro 的 Anthropic 兼容接口，把 embedding 端接到智谱 Embedding-3。两边的 base URL 和 API Key 是分开的：生成端负责回答，向量端负责检索。

注意：RAG 里有两类模型调用：

- `embedding`：负责把文档和问题向量化。
- `LLM`：负责根据检索结果生成答案。

DeepSeek V4 Pro 只负责 `LLM` 生成端；embedding 默认走智谱 `embedding-3`，不要复用聊天模型接口。

也可以用其他兼容 Anthropic Messages API 的模型服务，只要替换：

```text
ANTHROPIC_BASE_URL=兼容服务的 base_url
LLM_MODEL=对应模型名
LLM_API_STYLE=anthropic
LLM_API_KEY=对应 API key
```

如果同时设置了 `LLM_API_KEY`、`ANTHROPIC_API_KEY` 和 `DEEPSEEK_API_KEY`，脚本优先使用 `LLM_API_KEY`，其次使用 `ANTHROPIC_API_KEY`。

如果同时设置了 `EMBEDDING_API_KEY` 和 `ZHIPU_API_KEY`，脚本优先使用 `EMBEDDING_API_KEY`。

不要把真实 API Key 提交到仓库。

## 2. 先跑一次完整链路

第一次运行时加 `--rebuild`，脚本会读取 `sample_docs/`，切 chunk，调用 embedding，并把索引缓存到 `rag_index.json`。

```bash
python mini_rag.py --question "跨境订单退款多久到账？" --rebuild
```

你会先看到检索结果，大概像这样：

```text
#1 score=0.62 source=refund_policy.md chunk=1
跨境订单退款需要经过平台审核、跨境支付渠道确认和发卡行入账三个步骤...
```

然后看到模型答案：

```text
跨境订单退款在审核通过后，通常 7 到 15 个工作日到账 [1]。
如果超过 15 个工作日仍未到账，需要提供订单号、支付流水号和付款银行卡后四位，由客服发起支付渠道核查 [1]。
```

这里最重要的不是答案本身，而是你要注意两件事：

- 正确证据有没有被排在前面；
- 答案里的关键结论有没有引用证据编号。

RAG 的第一层能力，不是“能回答”，而是“知道自己基于什么回答”。

## 3. 只看检索，不让模型回答

很多 RAG 问题看最终答案是看不出来的。先把生成关掉，只检查召回：

```bash
python mini_rag.py --question "会员积分可以提现吗？" --retrieve-only
```

如果 Top-1 命中 `member_policy.md`，说明检索这一步大概率没坏。

再试一个不在知识库里的问题：

```bash
python mini_rag.py --question "今天北京天气怎么样？" --retrieve-only
```

你可能仍然会看到一些相似度不低的 chunk。这就是向量检索的一个常见现象：它总会努力找“相对最像”的内容，但“最像”不等于“足够相关”。

所以生产系统里通常还要做：

- 最低相关性阈值；
- Rerank；
- 答案前的资料充足性判断；
- 答案后的引用校验。

这也是为什么 RAG 不能只写一句 `top_k=5` 就完事。

## 4. 读懂代码里的 8 个环节

打开 `mini_rag.py`，按下面顺序看。

### 第一步：读取文档

`read_documents()` 会读取 `sample_docs/` 下的 Markdown 文件。

这一步在真实系统里对应数据接入：PDF、网页、数据库、客服工单、知识库页面，都要先变成统一文本。

常见坑：

- PDF 解析出来顺序错乱；
- 表格被拆散；
- 页眉页脚重复出现；
- 文档标题和路径丢失；
- HTML 菜单、广告、版权声明混进正文。

所以清洗不是可有可无。脏数据进入 RAG，后面每一环都要替它还债。

### 第二步：切 chunk

`chunk_document()` 用字符数做了一个很朴素的切块：

```text
max_chars=240
overlap_chars=60
```

为什么要 overlap？因为答案证据经常跨段落边界。如果切得太干净，上一段定义和下一段规则可能被分开，检索只拿回来半截。

但 overlap 也不能太大。太大时，Top-K 里会出现很多重复 chunk，看起来召回了 4 条，实际只是在重复同一段话。

你可以试：

```bash
python mini_rag.py --question "跨境订单退款多久到账？" --max-chars 160 --overlap-chars 40 --rebuild --retrieve-only
```

再试：

```bash
python mini_rag.py --question "跨境订单退款多久到账？" --max-chars 1200 --overlap-chars 100 --rebuild --retrieve-only
```

观察 Top-K 的变化。这个实验比背“chunk size 多少合适”有用得多。

### 第三步：生成 embedding

`embed_texts()` 会把 chunk 文本送到 embeddings 接口，返回一组向量。

向量可以理解成“语义坐标”。两个文本的向量越接近，语义越相关。智谱 `embedding-3` 默认输出 2048 维向量，也支持按需要选择 256 到 2048 之间的维度。

这里没有用向量数据库，而是把向量写进 `rag_index.json`。原因很简单：学习阶段先看懂向量是什么，再上数据库。

真实系统里，这一步会换成：

- Milvus；
- pgvector；
- Qdrant；
- Elasticsearch dense vector；
- OpenSearch vector；
- 云厂商托管向量检索。

但向量数据库只是存储和检索设施，不会替你解决 chunk 切坏、召回误命中、证据不足这些问题。

### 第四步：缓存索引

`get_or_create_index()` 会优先读取 `rag_index.json`。

如果你修改了 `sample_docs/`，记得加 `--rebuild`：

```bash
python mini_rag.py --question "预售商品可以催单吗？" --rebuild
```

这是新手很容易踩的坑：文档已经改了，但索引没重建，系统还在用旧知识回答。

生产系统里要处理得更严肃：

- 文档版本号；
- 增量更新；
- 删除同步；
- 重建任务状态；
- 索引与原文的一致性校验。

### 第五步：问题向量化

用户问题也要走 embedding。

`retrieve()` 里会先把问题变成 query embedding，再和所有 chunk embedding 算余弦相似度。

这一步对应“语义召回”。它擅长处理同义表达，比如：

```bash
python mini_rag.py --question "跨境退款一般几天能回到卡里？" --retrieve-only
```

即使文档里没有“回到卡里”这个说法，也可能命中“发卡行入账”“7 到 15 个工作日”。

但纯向量检索也有弱点：型号、编号、金额、专有名词、短关键词，经常不如 BM25 稳。所以工程上常见做法是“向量 + BM25”双路召回，再用 RRF 融合。

### 第六步：Top-K 召回

`retrieve()` 默认取 `top_k=4`。

试着改 Top-K：

```bash
python mini_rag.py --question "虚拟商品支持无理由退款吗？" --top-k 1
python mini_rag.py --question "虚拟商品支持无理由退款吗？" --top-k 6
```

Top-K 小了，可能漏证据。

Top-K 大了，可能引噪声。

这就是动态截断要解决的问题：不是永远拿固定数量，而是根据分数断崖、证据覆盖和 token 预算决定留下哪些 chunk。

### 第七步：组装上下文

`format_context()` 会把检索结果拼成这样：

```text
[1] source=refund_policy.md chunk=1 score=0.6231
跨境订单退款需要经过平台审核...
```

这里有三个细节很关键：

- 引用编号给模型一个可引用的锚点；
- source 和 chunk 方便排查坏 case；
- score 暂时给人看，不一定要给最终用户看。

很多 RAG 系统答案不可信，不是因为模型太差，而是上下文包太乱：没有边界、没有来源、没有优先级、重复内容一堆，模型只能自己猜。

### 第八步：生成答案

`answer_question()` 会按配置选择调用方式：默认 `LLM_API_STYLE=anthropic`，使用 DeepSeek 的 Anthropic-compatible Messages API；如果你显式改成 `chat_completions` 或 `responses`，脚本仍保留对应兼容路径。Prompt 会要求：

- 只使用资料；
- 资料不足就说不足；
- 关键结论后标注引用编号；
- 先短答，再补充说明。

这不是“防幻觉银弹”，但它给模型明确的工作边界。

你可以试这个问题：

```bash
python mini_rag.py --question "今天北京天气怎么样？"
```

理想情况下，系统应该说资料不足，而不是开始编天气。

## 5. 做一次人工评测

打开 `eval_set.md`，逐条运行问题。

每个问题分两步看：

```bash
python mini_rag.py --question "预售商品可以按 48 小时规则催单吗？" --retrieve-only
python mini_rag.py --question "预售商品可以按 48 小时规则催单吗？"
```

记录三列：

| 问题 | 检索是否命中证据 | 答案是否忠实 |
| --- | --- | --- |
| 跨境订单退款多久到账？ | 是 / 否 | 是 / 否 |
| 预售商品可以按 48 小时规则催单吗？ | 是 / 否 | 是 / 否 |
| 会员积分可以提现吗？ | 是 / 否 | 是 / 否 |

如果答案错了，不要第一反应改 Prompt。先定位坏在哪一段：

```text
文档接入错？
chunk 切坏？
召回没命中？
Top-K 太小？
Top-K 引入噪声？
上下文组装顺序错？
Prompt 没约束住？
模型无视证据？
```

这就是 RAG 评测的基本姿势：最终答案只是结果，链路分段才是诊断。

## 6. 三个必须亲手试的实验

### 实验一：改文档但不重建索引

把 `sample_docs/refund_policy.md` 里的“7 到 15 个工作日”改成“5 到 10 个工作日”。

然后不加 `--rebuild` 直接问：

```bash
python mini_rag.py --question "跨境订单退款多久到账？"
```

你会看到系统仍可能按旧索引回答。

再加 `--rebuild`：

```bash
python mini_rag.py --question "跨境订单退款多久到账？" --rebuild
```

这个实验能帮你理解：RAG 的“知识更新”不是改原文就结束了，索引也是系统状态的一部分。

### 实验二：制造冲突知识

新建一个 Markdown 文件，写入一条冲突规则：

```markdown
# 临时活动规则

跨境订单退款审核通过后，通常 3 到 5 个工作日到账。
```

然后重建索引：

```bash
python mini_rag.py --question "跨境订单退款多久到账？" --rebuild
```

观察模型怎么回答。

真实系统里，冲突知识很常见。解决它不能只靠模型“聪明一点”，通常要靠：

- 文档生效时间；
- 业务优先级；
- metadata 过滤；
- source trust level；
- 答案冲突检测。

### 实验三：问一个资料没有覆盖的问题

```bash
python mini_rag.py --question "海外地址能开发票吗？"
```

如果模型回答得很自信，就说明“资料不足”约束还不够。

你可以调整 Prompt，或者在生成前加一道规则：当 Top-K 分数都低于某个阈值时直接拒答。

示例：

```bash
python mini_rag.py --question "海外地址能开发票吗？" --min-score 0.35
```

阈值不是越高越好。高了会漏答，低了会乱答。真正靠谱的阈值来自评测集，而不是拍脑袋。

## 7. 常见报错

### API key is missing

没有创建 `.env`，或者 `.env` 里没有填 key。推荐配置里，生成端需要 `DEEPSEEK_API_KEY` 或 `LLM_API_KEY`；向量端需要 `ZHIPU_API_KEY` 或 `EMBEDDING_API_KEY`。

检查：

```bash
cat .env
```

### 模型不存在或无权限

把 `.env` 里的 `LLM_MODEL` 换成你账号可用的文本模型。

### 改了文档但答案没变

重新构建索引：

```bash
python mini_rag.py --question "你的问题" --rebuild
```

### 检索结果看起来不相关

先用 `--retrieve-only` 看 Top-K，再逐项排查：

- 问题是不是资料没覆盖；
- chunk 是不是切得太碎；
- Top-K 是不是太大；
- 是否需要 metadata filter；
- 是否需要 BM25 补关键词召回。

## 8. 你应该带走的感觉

RAG 不是“把资料塞给模型”。

RAG 是一条证据链：

```text
原始资料 -> 可检索证据 -> 相关证据 -> 可用上下文 -> 忠实答案
```

任何一段断了，最终答案都会漂。

所以这份练习最重要的收获不是跑通 `mini_rag.py`，而是建立一个工程直觉：

当 RAG 答错时，不要急着改 Prompt。先问：

```text
证据找到了吗？
找到了但排前面了吗？
排前面但塞进上下文了吗？
塞进去了但模型用了吗？
模型用了但引用对了吗？
```

