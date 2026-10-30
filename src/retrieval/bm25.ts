/**
 * 自写 BM25 —— JS 生态没有 rank_bm25 那么顺手的库，30 行搞定
 *
 * BM25 擅长精确匹配（错误码 ERR_500、版本号 v7.0.2510），
 * 这是向量检索的盲区，所以混合检索必须有它。
 */
export interface ScoredDoc<T> {
  doc: T;
  score: number;
}

function tokenize(text: string): string[] {
  // 中英混合切词：中文按字、英文按单词
  return text.toLowerCase().match(/[一-鿿]|[a-zA-Z0-9_]{2,}/g) ?? [];
}

export class BM25Index<T> {
  private docs: T[] = [];
  private tokenized: string[][] = [];
  private avgLen = 0;
  private docFreq = new Map<string, number>();

  constructor(private k1 = 1.5, private b = 0.75) {}

  add(docs: T[], getText: (d: T) => string): void {
    for (const doc of docs) {
      const tokens = tokenize(getText(doc));
      this.docs.push(doc);
      this.tokenized.push(tokens);
      const seen = new Set(tokens);
      for (const t of seen) {
        this.docFreq.set(t, (this.docFreq.get(t) ?? 0) + 1);
      }
    }
    this.avgLen =
      this.tokenized.reduce((s, t) => s + t.length, 0) / Math.max(this.tokenized.length, 1);
  }

  search(query: string, k = 5): ScoredDoc<T>[] {
    const qTokens = tokenize(query);
    const N = this.docs.length;
    const scores = this.tokenized.map((tokens, i) => {
      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);

      let s = 0;
      for (const qt of qTokens) {
        const f = tf.get(qt) ?? 0;
        if (!f) continue;
        const df = this.docFreq.get(qt) ?? 0;
        const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
        const denom = f + this.k1 * (1 - this.b + (this.b * tokens.length) / (this.avgLen || 1));
        s += idf * ((f * (this.k1 + 1)) / denom);
      }
      return { doc: this.docs[i], score: s };
    });

    return scores.sort((a, b) => b.score - a.score).slice(0, k);
  }
}
