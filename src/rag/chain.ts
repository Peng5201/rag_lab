/**
 * RAG 生成链 —— 用 LangChain 的 LCEL（表达式语言）组装
 *
 *   prompt.pipe(model).pipe(parser)
 *
 * 这是 LangChain 的招牌特性：链式组合、可流式、可观测。
 */
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { Document } from "@langchain/core/documents";

import type { ChatModelType } from "../llm/model.js";

export const RAG_PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "[TASK:generate] 你是一个严谨的技术支持助手。严格基于参考资料回答，不要编造任何资料中没有的信息。" +
      "如果资料不足以回答，明确说「资料不足」。",
  ],
  ["human", "参考资料：\n{context}\n\n问题：{question}\n\n回答要求：1. 结论先行 2. 引用资料编号 [1][2] 3. 给出排查步骤"],
]);

export function formatDocs(docs: Document[]): string {
  return docs
    .map((d, i) => `[${i + 1}]（${d.metadata.source ?? "未知"}）${d.pageContent}`)
    .join("\n\n");
}

/** 构建 LCEL 链：prompt → model → parser */
export function createRagChain(model: ChatModelType) {
  return RAG_PROMPT.pipe(model).pipe(new StringOutputParser());
}

/** 单次问答 */
export async function ask(question: string, docs: Document[], model: ChatModelType): Promise<string> {
  const chain = createRagChain(model);
  return chain.invoke({ context: formatDocs(docs), question });
}

/** 三轮递进查询改写：已移到 search.ts，与递进检索放在一起 */

/** Gold Dataset 相关性判断 */
export async function isRelevant(question: string, text: string, model: ChatModelType): Promise<boolean> {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "human",
      "[TASK:relevance] 判断下面的资料片段是否与问题相关，只回答「相关」或「不相关」。\n问题：{question}\n片段：{text}",
    ],
  ]);
  const chain = prompt.pipe(model).pipe(new StringOutputParser());
  const raw: string = await chain.invoke({ question, text: text.slice(0, 500) });
  return raw.trim().startsWith("相关");
}

/**
 * LLM 判别：一条陈述能否被参考资料支撑（faithfulness 用）
 *
 * 别拿 isRelevant 凑：那是「问题 ↔ 片段」的相关性，语义方向完全不同，
 * 混用会把上下文当问题、把陈述当片段，评出来的 faithfulness 是反的。
 */
export async function isSupported(claim: string, contexts: string[], model: ChatModelType): Promise<boolean> {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "human",
      "[TASK:support] 判断下面的陈述能否被参考资料支撑，只回答「支撑」或「不支撑」。\n参考资料：{context}\n陈述：{claim}",
    ],
  ]);
  const chain = prompt.pipe(model).pipe(new StringOutputParser());
  const raw: string = await chain.invoke({
    context: contexts.join("\n").slice(0, 3000),
    claim: claim.slice(0, 300),
  });
  return raw.trim().startsWith("支撑");
}

/**
 * LLM 判别：候选片段是否与参考片段集合表达同一个问题/结论
 * （Context Precision / Recall 用，能吃下「文件打不开」↔「无法加载」这类同义表述）
 */
export async function matchesAny(candidate: string, references: string[], model: ChatModelType): Promise<boolean> {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "human",
      "[TASK:match] 判断「待判断片段」是否与「参考片段」中任意一条表达同一个问题或同一结论，" +
        "只回答「匹配」或「不匹配」。\n参考片段：\n{refs}\n待判断片段：{candidate}",
    ],
  ]);
  const chain = prompt.pipe(model).pipe(new StringOutputParser());
  const raw: string = await chain.invoke({
    refs: references.map((r, i) => `[${i + 1}] ${r}`).join("\n").slice(0, 3000),
    candidate: candidate.slice(0, 500),
  });
  return raw.trim().startsWith("匹配");
}

export type { ChatModelType };
export { AIMessage, HumanMessage, SystemMessage };
