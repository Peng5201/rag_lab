/**
 * 模型接入层（Provider）—— 统一工厂
 *
 * 上层（chain.ts / hybridRetriever.ts / evaluate.ts）只认 LangChain 的
 * BaseChatModel 与 Embeddings 两个抽象，换模型 = 换配置，不动业务代码。
 *
 *   mock    HashEmbeddings   + MockChatModel     离线桩，跑通流程不连服务
 *   openai  OpenAIEmbeddings + ChatOpenAI        任何 OpenAI 兼容端点
 *
 * 所有真实模型都走 OpenAI 协议：DeepSeek（chat + 多模态）、
 * 各家向量服务、Ollama 的 /v1 兼容端点——统一一种接入方式，不再维护原生协议。
 */
import type { Embeddings } from "@langchain/core/embeddings";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

import { HashEmbeddings } from "./embeddings.js";
import { MockChatModel, type ChatModelType } from "./model.js";

export type ProviderKind = "mock" | "openai";

export interface ProviderConfig {
  kind: ProviderKind;
  chatModel: string;
  /** 多模态识图模型；默认跟随 chatModel，可独立指定 */
  vlmModel: string;
  embeddingModel: string;
  /** OpenAI 兼容端点 */
  baseUrl?: string;
  /** embedding 可与 chat 走不同端点（chat 用 DeepSeek、向量指向别家时） */
  embedBaseUrl?: string;
  /** openai 通路必填 */
  apiKey?: string;
  temperature: number;
}

/** 向量默认走本地（任何 OpenAI 兼容端点都行），换云向量用 LLM_EMBED_BASE_URL 覆盖 */
const DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:11434/v1";

/** 各 Provider 的默认模型名。chat 不给默认值——它跟着供应商走，必须显式配置 */
const DEFAULTS: Record<ProviderKind, Pick<ProviderConfig, "chatModel" | "vlmModel" | "embeddingModel">> = {
  mock: { chatModel: "mock-chat", vlmModel: "mock-chat", embeddingModel: "hash-512" },
  // chat 示例：deepseek-chat / gpt-4o / glm-4-plus
  openai: { chatModel: "", vlmModel: "", embeddingModel: "bge-m3:latest" },
};

function env(...names: string[]): string | undefined {
  for (const name of names) {
    const v = process.env[name];
    if (v && v.trim()) return v.trim();
  }
  return undefined;
}

/**
 * 配置解析：环境变量 → 默认值，调用方可传 overrides 覆盖。
 *
 *   RAG_PROVIDER        mock | openai              默认 mock
 *   LLM_BASE_URL        OpenAI 兼容端点（DeepSeek 等）
 *   LLM_API_KEY         兼容读取 OPENAI_API_KEY
 *   LLM_CHAT_MODEL      chat / 多模态模型
 *   LLM_EMBED_MODEL     LLM_EMBED_BASE_URL（向量端点可独立）
 *   LLM_TEMPERATURE     RAG_KNOWLEDGE_DIR（知识库目录）
 */
export function resolveConfig(overrides: Partial<ProviderConfig> = {}): ProviderConfig {
  // RAG_PROVIDER 没设时自动推断：配了端点就用 openai，什么都没配才退回 mock
  const inferred = env("LLM_BASE_URL", "OPENAI_BASE_URL") ? "openai" : "mock";
  const kind = overrides.kind ?? (env("RAG_PROVIDER") as ProviderKind | undefined) ?? inferred;
  const base = DEFAULTS[kind] ?? DEFAULTS.mock;
  // mock 是离线桩，没有真实模型，也不读模型名——避免配置显示与实际实现不符
  const read = (name: string, ...alias: string[]) => (kind === "mock" ? undefined : env(name, ...alias));

  const chatModel = overrides.chatModel ?? read("LLM_CHAT_MODEL", "OPENAI_CHAT_MODEL") ?? base.chatModel;
  // 多模态识图模型默认跟随 chatModel，也可独立指定
  const vlmModel =
    (overrides.vlmModel ?? read("LLM_VLM_MODEL", "OPENAI_VLM_MODEL") ?? base.vlmModel) || chatModel;

  return {
    kind,
    chatModel,
    vlmModel,
    embeddingModel:
      overrides.embeddingModel ?? read("LLM_EMBED_MODEL", "OPENAI_EMBED_MODEL") ?? base.embeddingModel,
    baseUrl: overrides.baseUrl ?? read("LLM_BASE_URL", "OPENAI_BASE_URL"),
    // 向量与 chat 分家：chat 端点通常没有 embedding 接口，默认指向本地
    embedBaseUrl:
      overrides.embedBaseUrl ?? read("LLM_EMBED_BASE_URL", "EMBED_BASE_URL") ?? (kind === "mock" ? undefined : DEFAULT_EMBED_BASE_URL),
    apiKey: overrides.apiKey ?? read("LLM_API_KEY", "OPENAI_API_KEY"),
    temperature: overrides.temperature ?? Number(env("LLM_TEMPERATURE") ?? 0.2),
  };
}

/** LLM：mock 与真实端点都是 BaseChatModel，可互换 */
export function createChatModel(cfg: ProviderConfig): ChatModelType {
  switch (cfg.kind) {
    case "mock":
      return new MockChatModel();

    case "openai": {
      // 换个 baseURL + model 就是 DeepSeek / GPT / 通义 / vLLM
      if (!cfg.baseUrl) {
        throw new Error("未配置 chat 端点：请设置 LLM_BASE_URL（OpenAI 兼容端点）");
      }
      if (!cfg.chatModel) {
        throw new Error("未配置 chat 模型：请设置 LLM_CHAT_MODEL（例：deepseek-chat / gpt-4o / glm-4-plus）");
      }
      return new ChatOpenAI({
        model: cfg.chatModel,
        temperature: cfg.temperature,
        configuration: { baseURL: cfg.baseUrl },
        // SDK 要求 key 非空；指向不鉴权端点时用占位符
        apiKey: cfg.apiKey ?? "no-key-required",
      });
    }

    default:
      throw new Error(`未知 Provider：${cfg.kind}`);
  }
}

/** Embeddings：统一 L2 归一化契约 */
export function createEmbeddings(cfg: ProviderConfig): Embeddings {
  switch (cfg.kind) {
    case "mock":
      return new HashEmbeddings();

    case "openai": {
      // chat 端点往往没有 embedding 接口（如 DeepSeek），向量端点要单独给
      const base = cfg.embedBaseUrl ?? cfg.baseUrl;
      if (!base) {
        throw new Error(
          "未配置向量端点：请设置 LLM_EMBED_BASE_URL\n" +
            "（chat 端点通常没有 embedding 接口；向量可指向本地或云端的 OpenAI 兼容端点）",
        );
      }
      if (!cfg.embeddingModel) {
        throw new Error(
          "未配置向量模型：请设置 LLM_EMBED_MODEL（例：mxbai-embed-large-v1 / text-embedding-3-small / embedding-3）",
        );
      }
      return new OpenAIEmbeddings({
        model: cfg.embeddingModel,
        configuration: { baseURL: base },
        apiKey: cfg.apiKey ?? "no-key-required",
      });
    }

    default:
      throw new Error(`未知 Provider：${cfg.kind}`);
  }
}

/** 启动时打一行配置，方便确认打的是哪个模型 */
export function describeConfig(cfg: ProviderConfig): string {
  const target = cfg.baseUrl ? ` @ ${cfg.baseUrl}` : "";
  const embedTarget = cfg.embedBaseUrl && cfg.embedBaseUrl !== cfg.baseUrl ? ` (embed @ ${cfg.embedBaseUrl})` : "";
  const vlmPart = cfg.vlmModel && cfg.vlmModel !== cfg.chatModel ? `  vlm=${cfg.vlmModel}` : "";
  return `provider=${cfg.kind}  chat=${cfg.chatModel}${vlmPart}  embeddings=${cfg.embeddingModel}${target}${embedTarget}`;
}

export type { BaseChatModel, Embeddings };
