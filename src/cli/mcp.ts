/**
 * MCP Server —— 把知识库检索暴露成 Agent 工具（对应文档 §11）
 *
 * stdio 传输：宿主（Claude Code / Cursor）spawn 这个进程，走 JSON-RPC 调用工具。
 * 注意：stdout 是协议通道，日志一律走 stderr。
 *
 *   RAG_PROVIDER=openai LLM_EMBED_BASE_URL=… LLM_EMBED_MODEL=… npm run kb:mcp
 *
 * 新增工具 = 再 registerTool 一个；新增被测/被调检索策略 = search() 里加一个 case。
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import type { Document } from "@langchain/core/documents";

import { createChatModel, createEmbeddings, describeConfig, resolveConfig } from "../llm/providers.js";
import { findKnowledgeDir, loadAll } from "../knowledge/splitter.js";
import { buildRetriever, type HybridRetriever } from "../retrieval/hybridRetriever.js";
import type { InMemoryVectorStore } from "../retrieval/vectorStore.js";
import { progressiveSearch } from "../retrieval/search.js";
import type { ChatModelType } from "../llm/model.js";

/** stdout 是 MCP 协议通道，日志只能走 stderr */
const log = (...args: unknown[]): void => {
  console.error("[rag-mcp]", ...args);
};

interface Kb {
  retriever: HybridRetriever;
  vectorStore: InMemoryVectorStore;
  chunks: number;
  dir: string;
}

let kbPromise: Promise<Kb> | undefined;
let modelCache: ChatModelType | undefined;

/**
 * chat 模型懒加载：serving 链路只需要 embedding，
 * 只有 progressive 策略（查询改写）才用到 LLM。
 */
function getModel(): ChatModelType {
  modelCache ??= createChatModel(resolveConfig());
  return modelCache;
}

/** 懒加载：索引在第一次工具调用时才建，启动要快，报错也落在工具结果里 */
function ensureKb(): Promise<Kb> {
  if (!kbPromise) {
    kbPromise = (async () => {
      const cfg = resolveConfig();
      const dir = findKnowledgeDir();
      const docs = await loadAll(dir);
      const { retriever, vectorStore } = await buildRetriever(docs, createEmbeddings(cfg));
      log(`索引就绪：${describeConfig(cfg)}  chunk=${docs.length}  dir=${dir}`);
      return { retriever, vectorStore, chunks: docs.length, dir };
    })().catch((e) => {
      kbPromise = undefined; // 失败不缓存，改完配置可直接重试
      throw e;
    });
  }
  return kbPromise;
}

type Strategy = "hybrid" | "progressive" | "vector" | "bm25";

async function search(query: string, k: number, strategy: Strategy): Promise<Document[]> {
  const { retriever, vectorStore } = await ensureKb();
  switch (strategy) {
    case "hybrid":
      return retriever.hybridSearch(query, k);
    case "progressive":
      // 需要 chat 模型做多视角查询改写；agent 自己换角度多问几次也等效
      return progressiveSearch(retriever, query, getModel(), k);
    case "vector":
      return vectorStore.similaritySearch(query, k);
    case "bm25":
      return retriever.keywordSearch(query, k);
  }
}

const server = new McpServer({ name: "rag-knowledge", version: "1.0.0" });

server.registerTool(
  "search_knowledge",
  {
    title: "检索知识库",
    description:
      "检索知识库（工单记录 + 排查手册），返回相关片段原文与来源。" +
      "只做检索、不做回答——生成答案和引用组织由调用方完成。" +
      "回答表格/文档的技术支持问题（转 PDF 跑版、下拉列表、权限、兼容性等）之前先调用它。",
    inputSchema: {
      query: z.string().describe("检索问题，用户原始描述或关键词组合"),
      k: z.number().int().min(1).max(20).default(5).describe("返回片段数，默认 5"),
      strategy: z
        .enum(["hybrid", "progressive", "vector", "bm25"])
        .default("hybrid")
        .describe(
          "检索策略：hybrid=混合检索（默认，纯检索零 LLM）；" +
            "progressive=三轮递进（需 chat 模型做查询改写，召回更全）；vector / bm25=单路，用于对比",
        ),
    },
  },
  async ({ query, k, strategy }) => {
    try {
      const hits = await search(query, k, strategy);
      const text = hits.length
        ? hits
            .map((d, i) => `[${i + 1}]（${d.metadata.source ?? "未知"}）${d.pageContent}`)
            .join("\n\n")
        : "知识库中没有找到相关片段，换种问法或确认知识库是否覆盖该问题";
      return { content: [{ type: "text", text }] };
    } catch (e) {
      return {
        content: [{ type: "text", text: `检索失败：${(e as Error).message}` }],
        isError: true,
      };
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
log("已启动（stdio）。RAG_KNOWLEDGE_DIR 未设置时按 <文档>/RAG-Lab/knowledge 查找");
