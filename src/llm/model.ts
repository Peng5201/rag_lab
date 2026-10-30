/**
 * LLM 层：继承 LangChain 的 SimpleChatModel
 *
 * - MockChatModel: 离线降级，保证无 API key 也能跑通全流程
 * - 真实模型在 providers.ts 里选：ChatOpenAI（任何 OpenAI 兼容端点）
 *
 * 三者都是 BaseChatModel，对 chain.ts 完全可互换。
 */
import {
  SimpleChatModel,
  type BaseChatModel,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

/** 词面重合（Mock 判别用）：真实模型下这几个 TASK 由语义判断接管 */
function wordOverlap(a: string, b: string): boolean {
  const aw = new Set(a.match(/[一-鿿]{2,}|[a-zA-Z0-9_]{2,}/g) ?? []);
  const bw = new Set(b.match(/[一-鿿]{2,}|[a-zA-Z0-9_]{2,}/g) ?? []);
  return [...aw].some((w) => bw.has(w));
}

export class MockChatModel extends SimpleChatModel {
  lc_namespace = ["rag", "models"];

  constructor(params: BaseChatModelParams = {}) {
    super(params);
  }

  _llmType(): string {
    return "mock";
  }

  /** SimpleChatModel 契约：返回字符串，基类负责包成 ChatResult */
  async _call(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<string> {
    const prompt = messages.map((m) => m.content).join("\n");

    // 与 Python 版一致：按 [TASK:xxx] 标签分类
    const task = prompt.match(/\[TASK:(\w+)\]/)?.[1] ?? "";

    if (task === "rewrite") {
      const q = prompt.split("问题：").pop()?.trim() ?? "";
      const words = q.match(/[一-鿿]{2,}|[a-zA-Z0-9_]{2,}/g) ?? [];
      return JSON.stringify({
        query_1: q,
        query_2: words.slice(0, 6).join(" "),
        query_3: words[words.length - 1] ?? q,
      });
    }

    if (task === "relevance") {
      const q = prompt.split("问题：").pop()?.split("片段：")[0] ?? "";
      const c = prompt.split("片段：").pop() ?? "";
      return wordOverlap(q, c) ? "相关" : "不相关";
    }

    if (task === "support") {
      const claim = prompt.split("陈述：").pop() ?? "";
      const refs = prompt.split("参考资料：").pop()?.split("陈述：")[0] ?? "";
      return wordOverlap(claim, refs) ? "支撑" : "不支撑";
    }

    if (task === "match") {
      const cand = prompt.split("待判断片段：").pop() ?? "";
      const refs = prompt.split("参考片段：").pop()?.split("待判断片段：")[0] ?? "";
      return wordOverlap(cand, refs) ? "匹配" : "不匹配";
    }

    if (task === "generate") {
      const refs = prompt.split("参考资料：").pop()?.split("问题：")[0] ?? "";
      const first = refs.trim().split("\n")[0] ?? "";
      return `根据资料：${first.slice(0, 200)}（Mock 答案，接入真实 LLM 后替换）`;
    }

    return "MockChatModel 未识别的 prompt 类型";
  }
}

/** 三种 Provider 的模型都能塞进 LCEL 链 */
export type ChatModelType = BaseChatModel;
export type ModelInput = BaseLanguageModelInput;
