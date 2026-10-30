const { OpenAIEmbeddings } = require("@langchain/openai");

(async () => {
  const e = new OpenAIEmbeddings({
    model: "mxbai-embed-large-v1",
    configuration: { baseURL: "http://127.0.0.1:11434/v1" },
    apiKey: "no-key-required",
  });
  console.log("clientConfig.baseURL =", e.clientConfig?.baseURL);
  try {
    const r = await e.embedDocuments(["测试一", "测试二", "测试三"]);
    console.log("OK", r.length, "维", r[0].length);
  } catch (err) {
    console.log("ERR:", err.message.slice(0, 200));
  }
})();
