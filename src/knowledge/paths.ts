/**
 * 知识库路径解析 —— 知识库与代码仓库分离
 *
 * 物料是业务私有数据，不该跟源码一起分发；重装、克隆、打包发布都不应影响它。
 * 所以默认落到系统「文档」目录：
 *
 *   <文档>/RAG-Lab/knowledge
 *
 * 解析顺序：RAG_KNOWLEDGE_DIR → <文档>/RAG-Lab/knowledge → 项目内 knowledge/（兼容旧位置）
 */
import { execFileSync } from "node:child_process";
import os from "node:os";
import path from "node:path";

/** 展开注册表里的 %USERPROFILE% 之类的变量 */
function expandEnv(value: string): string {
  return value.replace(/%([^%]+)%/g, (_, name: string) => process.env[name] ?? `%${name}%`);
}

/**
 * 系统「文档」目录。
 * Windows 走注册表读真实位置，能正确处理重定向到其他盘符 / OneDrive 的机器；
 * 其他平台按惯例拼 ~/Documents。
 */
export function documentsDir(): string {
  if (process.platform === "win32") {
    try {
      const out = execFileSync(
        "reg",
        [
          "query",
          String.raw`HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders`,
          "/v",
          "Personal",
        ],
        { encoding: "utf-8", windowsHide: true },
      );
      const hit = out.match(/Personal\s+REG_(?:EXPAND_)?SZ\s+(.+)/);
      if (hit) return expandEnv(hit[1].trim());
    } catch {
      // 读不到注册表就退回默认拼法
    }
  }
  return path.join(os.homedir(), "Documents");
}

/** 默认知识库目录：<文档>/RAG-Lab/knowledge */
export function defaultKnowledgeDir(): string {
  return path.join(documentsDir(), "RAG-Lab", "knowledge");
}
