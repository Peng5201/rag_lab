# 复制为 config.local.ps1 后填写（该文件已在 .gitignore 中，不会入库）
# 也可以不建这个文件，直接在终端里设环境变量。

$env:LLM_BASE_URL   = "https://api.deepseek.com"
$env:LLM_API_KEY    = "<你的 key>"
$env:LLM_CHAT_MODEL = "deepseek-chat"

# 识图模型默认跟随 chat，需要独立指定时取消注释
# $env:LLM_VLM_MODEL = "gpt-4o"

# 向量模型（本地 Ollama）
$env:LLM_EMBED_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LLM_EMBED_MODEL    = "bge-m3:latest"
