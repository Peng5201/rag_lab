/**
 * OpenAI 兼容客户端 —— 只用到 chat/completions（含图片输入）
 *
 * 所有真实模型都走这一条路：DeepSeek（chat + 多模态）、各家向量服务、
 * 以及任何 OpenAI 兼容端点。不再维护各家原生协议。
 */

export interface OpenAITextPart {
  type: "text";
  text: string;
}

export interface OpenAIImagePart {
  type: "image_url";
  image_url: { url: string };
}

export type OpenAIContentPart = OpenAITextPart | OpenAIImagePart;

export interface OpenAIMessage {
  role: "system" | "user" | "assistant";
  /** 纯文本直接给字符串；带图就给 content part 数组 */
  content: string | OpenAIContentPart[];
}

export interface OpenAIChatOptions {
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
}

export function toDataUrl(mime: string, base64: string): string {
  return `data:${mime};base64,${base64}`;
}

/** chat/completions：兼容 DeepSeek / GPT / 通义 / GLM / Ollama 的 /v1 */
export async function openAIChat(
  baseUrl: string,
  apiKey: string,
  model: string,
  messages: OpenAIMessage[],
  opts: OpenAIChatOptions = {},
): Promise<string> {
  const url = `${baseUrl.replace(/\/+$/, "")}/chat/completions`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: opts.temperature ?? 0.1,
        ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
      }),
      signal: AbortSignal.timeout(opts.timeoutMs ?? 180_000),
    });
  } catch (e) {
    throw new Error(`OpenAI 兼容端点连不上（${url}）：${(e as Error).message}`);
  }

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    const hint =
      res.status === 404
        ? `\n提示：HTTP 404 通常是端点路径不对——确认 LLM_BASE_URL 是 OpenAI 兼容的根路径` +
          `（如 https://api.deepseek.com，别带 /anthropic 这类别的协议后缀）`
        : `\n提示：确认模型名是否正确（含多模态模型）、key 是否有效、端点是否支持图片输入`;
    throw new Error(`OpenAI 兼容端点调用失败（HTTP ${res.status}）：${detail.slice(0, 300)}${hint}`);
  }

  const data = (await res.json()) as {
    choices?: { message?: { content?: string | OpenAIContentPart[]; reasoning_content?: string } }[];
  };
  const content = data.choices?.[0]?.message?.content ?? "";
  const text =
    typeof content === "string"
      ? content
      : // 少数实现把多模态返回也拆成 parts
        content.map((p) => (p.type === "text" ? p.text : "")).join("");

  if (!text.trim()) {
    // reasoning 模型会把正文写进 reasoning_content，content 留空——不能当成功
    const reasoning = data.choices?.[0]?.message?.reasoning_content;
    throw new Error(
      `OpenAI 兼容端点返回空回答${reasoning ? "（内容写进了 reasoning_content）" : ""}。\n` +
        `提示：换个非 reasoning 模型，或确认端点支持该输入形态（图片需要多模态模型）`,
    );
  }
  return text;
}
