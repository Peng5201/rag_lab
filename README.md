# RAG-Lab (TypeScript)

基于 LangChain.js 构建的模块化 RAG 实验库：实现知识库检索、混合搜索、多模态入库与自动化评测，可作为 Agent 工具链的检索后端。

核心定位：

- **会用 LangChain 抽象**：`Embeddings` / `VectorStore` / `BaseRetriever` / LCEL。
- **核心算法自研**：混合检索融合、差异化分块、评测指标。
- **模型接入解耦**：Chat / VLM / Embedding 三类模型独立配置，统一走 OpenAI 兼容协议。

---

## 目录

- [功能说明](#功能说明)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
  - [1. 安装依赖并启动本地 Embedding 模型](#1-安装依赖并启动本地-embedding-模型)
  - [2. 配置 Chat 与 VLM 模型](#2-配置-chat-与-vlm-模型)
  - [3. 物料入库并问答](#3-物料入库并问答)
- [其他常用命令](#其他常用命令)
- [使用步骤](#使用步骤)
  - [1. 物料入库](#1-物料入库)
  - [2. 直接提问](#2-直接提问)
  - [3. 端到端评测](#3-端到端评测)
  - [4. 接入 MCP](#4-接入-mcp)
- [原理解析](#原理解析)
  - [模型接入层](#模型接入层)
  - [检索策略](#检索策略)
  - [分块策略](#分块策略)
  - [评测体系](#评测体系)
- [CLI 参数](#cli-参数)
- [License](#license)

---

## 功能说明

| 能力 | 说明 |
| --- | --- |
| **多源物料入库** | 支持 docx / pptx / xlsx / pdf / 图片 / 视频 / csv；视频仅保留引用、不入库 |
| **多模态识图** | 图片与扫描版 PDF 通过 VLM 抽取文字与关键事实，LLM 自动归位到对应章节 |
| **混合检索** | 向量语义检索 + BM25 关键词检索融合，兼顾同义表述与精确术语 |
| **多视角查询改写** | 自动生成自然语言、关键词组合、精确术语三轮查询，提升召回 |
| **自动化评测** | 8 项指标（检索质量 5 项 + 生成质量 3 项），LLM 与规则指标成对出现 |
| **MCP 暴露** | 通过 Model Context Protocol 把检索能力暴露给 Claude Code / Cursor 等 Agent |

---

## 目录结构

```text
src/
├── cli/                  # 入口层：参数解析 + 装配
│   ├── ask.ts            # 直接提问（检索 + 生成）
│   ├── evaluate.ts       # 自检 + 评测对比
│   ├── ingest.ts         # 物料入库管线
│   └── mcp.ts            # MCP server
├── ingest/               # 入库层：物料 → 知识单元
│   └── parsers.ts        # 文档 / 图片 / PDF / 视频解析
├── knowledge/            # 知识层
│   └── splitter.ts       # 差异化分块 + 目录加载
├── retrieval/            # 检索层
│   ├── vectorStore.ts    # InMemoryVectorStore（继承 VectorStore）
│   ├── bm25.ts           # 自研 BM25
│   ├── hybridRetriever.ts # HybridRetriever（继承 BaseRetriever）
│   └── search.ts         # 检索策略
├── llm/                  # 模型接入层
│   ├── providers.ts      # Provider 工厂
│   ├── openai.ts         # OpenAI 协议兼容
│   ├── model.ts          # MockChatModel
│   └── embeddings.ts     # HashEmbeddings
├── rag/                  # 生成 + 评测
│   ├── chain.ts          # LCEL 链
│   └── evaluate.ts       # 评测指标
└── types/                # 类型声明
```

依赖单向：`cli → rag / retrieval / ingest → llm / knowledge`。

---

## 快速开始

三步跑起来：安装本地 Embedding 模型 → 配置 Chat/VLM → 入库并问答。

### 1. 安装依赖并启动本地 Embedding 模型

```bash
npm install
ollama pull bge-m3:latest
ollama serve
```

> Embedding 默认走本地 Ollama，免费、中文强、响应稳定。

### 2. 配置 Chat 与 VLM 模型

```powershell
$env:LLM_BASE_URL   = "https://api.deepseek.com"
$env:LLM_API_KEY    = "sk-xxx"
$env:LLM_CHAT_MODEL = "deepseek-v4-flash"

# 识图模型默认跟随 chat，需要时可独立指定
$env:LLM_VLM_MODEL  = "deepseek-v4-flash"
```

### 3. 物料入库

```bash
# 把物料解析成知识库（默认写入 <文档>/RAG-Lab/knowledge）
npm run ingest -- <物料目录>
```

### 4. 问答

```bash
# 基于 Embedding 检索 + Chat 生成回答,
npm run ask -- "你的问题"
```

---

## 其他常用命令

```bash
npm start              # Mock 模式：离线跑通流程
npm run eval           # 端到端评测：建库 → 检索 → 8 项指标
npm run kb:mcp         # 启动 MCP server，把检索暴露给 Agent
npm run typecheck      # 类型检查
```

PowerShell 一键脚本：

```powershell
.\run.ps1 check              # 连通性检查
.\run.ps1 start             # 端到端评测
.\run.ps1 ask "你的问题"
.\run.ps1 mcp               # 启动 MCP server
.\run.ps1 ingest F:\code\temp\data
```

### 配置说明

配置优先级：**CLI 参数 > 环境变量 > 代码默认值**。

完整环境变量：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `RAG_PROVIDER` | `mock` / `openai` | `mock` |
| `LLM_BASE_URL` | Chat / VLM 的 OpenAI 兼容端点 | 必填（openai 模式） |
| `LLM_API_KEY` | API Key，兼容 `OPENAI_API_KEY` | — |
| `LLM_CHAT_MODEL` | Chat 模型 | 必填（openai 模式） |
| `LLM_VLM_MODEL` | 识图模型 | `LLM_CHAT_MODEL` |
| `LLM_EMBED_BASE_URL` | Embedding 端点 | `http://127.0.0.1:11434/v1` |
| `LLM_EMBED_MODEL` | Embedding 模型 | `bge-m3:latest` |
| `LLM_TEMPERATURE` | 采样温度 | `0.2` |
| `RAG_KNOWLEDGE_DIR` | 知识库目录 | `<文档>/RAG-Lab/knowledge` |

---

## 使用步骤

### 1. 物料入库

```bash
npm run ingest -- <物料目录>
```

默认输出到系统「文档」目录下的 `RAG-Lab/knowledge`，知识库与代码仓库分离。
需要换位置用 `--out <目录>` 或 `RAG_KNOWLEDGE_DIR`。

支持素材与处理方式：

| 素材 | 处理方式 |
| --- | --- |
| `.docx` / `.pptx` / `.xlsx` / `.pdf` | officeparser 本地解析 → Markdown |
| 图片 | VLM 抽取文字与关键事实 → LLM 归位 |
| 扫描版 PDF | 渲染成图 → VLM 识别 |
| `.csv` | 原样复制，走行级 KV 分块 |
| 视频 | 只留引用 + 一句话概括 |
| 其他 | 抽取可读文本，抽不出则跳过 |

常用选项：

```bash
npm run ingest -- <物料目录> --no-vlm          # 跳过识图
npm run ingest -- <物料目录> --max-images=5    # 限制识图数量
npm run ingest -- <物料目录> --max-pages=30    # 限制 PDF 页数
npm run ingest -- <物料目录> --vlm-model=gpt-4o # 临时指定识图模型
```

### 2. 直接提问

```bash
npm run ask -- "文档导出 PDF 后字体显示异常，怎么排查？"
```

输出包含：结构化答案 + 引用来源。

### 3. 端到端评测

```bash
npm run eval
```

流程：构建向量库 → 检索 → 自动生成 Gold Dataset → 跑 8 项指标 → 输出对比报告。

### 4. 接入 MCP

把检索能力通过 MCP 暴露给 Agent，只需配置 Embedding，无需 Chat 模型。

```bash
RAG_PROVIDER=openai npm run kb:mcp
```

宿主 `.mcp.json` 示例：

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "npx",
      "args": ["tsx", "src/cli/mcp.ts"],
      "env": {
        "RAG_PROVIDER": "openai",
        "LLM_EMBED_BASE_URL": "http://127.0.0.1:11434/v1",
        "LLM_EMBED_MODEL": "bge-m3:latest"
      }
    }
  }
}
```

暴露工具：`search_knowledge(query, k, strategy)`，策略可选：

| 策略 | 所需模型 | 说明 |
| --- | --- | --- |
| `hybrid` | embedding | 默认，向量 + BM25 |
| `progressive` | embedding + chat | 多轮改写，召回更全 |
| `vector` | embedding | 仅向量检索 |
| `bm25` | embedding | 仅 BM25 检索 |

---

## 原理解析

### 模型接入层

上层只依赖 LangChain 的 `BaseChatModel` / `Embeddings` 两个抽象：

```ts
const cfg = resolveConfig({ kind: "openai" });
const model = createChatModel(cfg);       // BaseChatModel
const embeddings = createEmbeddings(cfg); // Embeddings
```

- Chat、VLM、Embedding 三类模型**独立配置**，统一通过 OpenAI 兼容端点接入。
- `mock` 模式使用 `MockChatModel` + `HashEmbeddings`，离线跑通流程。

### 检索策略

单纯向量或关键词检索各有盲区：

| 检索方式 | 擅长 | 不擅长 |
| --- | --- | --- |
| 向量检索 | 语义相似，如「文档打不开」匹配「文件无法加载」 | 精确错误码、版本号 |
| BM25 | 精确匹配专有术语 | 同义不同表述 |

采用 **Hybrid Search（向量 + BM25 融合）**，配合 **三轮递进搜索**：

1. 完整自然语言描述（hybrid）
2. 核心关键词组合（hybrid）
3. 精确术语兜底（keyword）

### 分块策略

| 文档类型 | 分块方式 |
| --- | --- |
| 表格（xlsx/csv） | 每行一个 chunk，KV 格式，附带来源元信息 |
| 文档（md/otl/docx） | 以段落/章节为自然边界，重叠区保留语义连续 |

### 评测体系

8 项指标，LLM 与规则成对：

| 维度 | 指标 | 实现 |
| --- | --- | --- |
| 检索质量 | Context Precision | LLM 语义匹配 |
| 检索质量 | Context Recall | LLM 语义匹配 |
| 检索质量 | Context Entity Recall | 规则 |
| 检索质量 | NonLLM Precision | 规则（字符 Dice） |
| 检索质量 | NonLLM Recall | 规则（字符 Dice） |
| 生成质量 | Faithfulness | LLM 逐条判断 |
| 生成质量 | Answer Relevancy | 规则 |
| 生成质量 | Answer Correctness | 规则 |

Gold Dataset 借鉴 TREC Pooling：多路检索结果取并集，再由 LLM 逐片判断相关性，避免单一系统偏差。

---

## CLI 参数

全局可用：

| 参数 | 说明 |
| --- | --- |
| `--provider=mock\|openai` | 指定 Provider |
| `--chat-model=<model>` | 指定 Chat 模型 |
| `--embed-model=<model>` | 指定 Embedding 模型 |

`ingest` 专用：

| 参数 | 说明 |
| --- | --- |
| `--out=<dir>` | 输出目录 |
| `--no-vlm` | 跳过视觉解析 |
| `--max-images=<n>` | 最大识图数量 |
| `--max-pages=<n>` | PDF 最大页数 |
| `--vlm-model=<model>` | 临时指定识图模型 |

---

## License

MIT


## 模型选型
embedding选型：https://cloud.tencent.com/developer/article/2696433
