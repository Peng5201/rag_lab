<#
  模型接入启动脚本（OpenAI 兼容协议，任意供应商）

  能力分工（在下面配置块里填，或用环境变量覆盖）：
    chat        生成答案 / 查询改写 / LLM 判别 / 图片归位
    多模态识图   入库时解析图片（需要支持图片输入的模型）
    向量        建索引（chat 端点通常没有 embedding 接口，往往要单独指一个端点）

  用法：
    .\run.ps1 check                     # 连通性检查 + 列出各端点可用模型 ID
    .\run.ps1 start                     # 端到端：建库 → 检索 → Gold → 8 指标
    .\run.ps1 mcp                       # MCP server（serving，只需向量）
    .\run.ps1 ingest <物料目录>          # 入库（含多模态识图）
#>
param(
  [Parameter(Position = 0)][string]$Command = "check",
  [Parameter(Position = 1)][string]$Path,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ------------------------- 配置（环境变量优先） -------------------------
# 密钥不写进仓库：本地私有配置放 config.local.ps1（已在 .gitignore 中），
# 或者直接设环境变量。两者都优先于下面的默认值。

$localConfig = Join-Path $PSScriptRoot "config.local.ps1"
if (Test-Path $localConfig) { . $localConfig }

# chat + 多模态：任意 OpenAI 兼容端点
#   https://api.deepseek.com
#   https://api.openai.com/v1
#   https://open.bigmodel.cn/api/paas/v4          （智谱）
#   https://dashscope.aliyuncs.com/compatible-mode/v1   （阿里）
#   http://127.0.0.1:11434/v1                     （本地 OpenAI 兼容端点）
if (-not $env:LLM_BASE_URL)   { $env:LLM_BASE_URL   = "https://api.deepseek.com" }
if (-not $env:LLM_CHAT_MODEL) { $env:LLM_CHAT_MODEL = "deepseek-chat" }   # 例：deepseek-chat / gpt-4o / glm-4-plus
if (-not $env:LLM_VLM_MODEL)  { $env:LLM_VLM_MODEL  = $env:LLM_CHAT_MODEL }  # 需支持图片输入

# 向量：默认指向本地 OpenAI 兼容端点（换云向量改这两行）
if (-not $env:LLM_EMBED_BASE_URL) { $env:LLM_EMBED_BASE_URL = "http://127.0.0.1:11434/v1" }
if (-not $env:LLM_EMBED_MODEL)    { $env:LLM_EMBED_MODEL    = "bge-m3:latest" }

# ------------------------- 配置校验 -------------------------
function Assert-Set([string]$Name, [string]$Example) {
  if (-not [Environment]::GetEnvironmentVariable($Name)) {
    Write-Host "缺少 $Name" -ForegroundColor Yellow
    Write-Host "  设置示例：$Example"
    exit 1
  }
}

$needChat = ($Command -eq "start" -or $Command -eq "ingest" -or $Command -eq "ask")
$needEmbed = ($Command -eq "start" -or $Command -eq "ingest" -or $Command -eq "mcp" -or $Command -eq "ask")

if ($needChat) {
  Assert-Set "LLM_BASE_URL"   '$env:LLM_BASE_URL   = "https://api.deepseek.com"'
  Assert-Set "LLM_API_KEY"    '$env:LLM_API_KEY    = "<你的 key>"'
  Assert-Set "LLM_CHAT_MODEL" '$env:LLM_CHAT_MODEL = "deepseek-chat"'
}
if ($needEmbed) {
  Assert-Set "LLM_EMBED_BASE_URL" '$env:LLM_EMBED_BASE_URL = "http://127.0.0.1:11434/v1"'
  Assert-Set "LLM_EMBED_MODEL"    '$env:LLM_EMBED_MODEL    = "mxbai-embed-large-v1"'
}

# ------------------------- 连通性检查 -------------------------
function Test-Endpoint([string]$Name, [string]$Base, [string]$Model) {
  try {
    $r = Invoke-RestMethod -Uri "$Base/models" `
      -Headers @{ Authorization = "Bearer $($env:LLM_API_KEY)" } -TimeoutSec 15
    Write-Host "$Name OK：$Base  模型=$Model" -ForegroundColor Green
    $ids = @($r.data | ForEach-Object { $_.id })
    if ($ids.Count -gt 0) {
      $hit = $ids | Where-Object { $_ -like "$Model*" }
      if (-not $hit) {
        Write-Host "  端点上没找到该模型名，可用：" -ForegroundColor Yellow
        $ids | ForEach-Object { Write-Host "    - $_" }
      }
    }
  } catch {
    Write-Host "$Name 连不上：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
  }
}

# ------------------------- 分发 -------------------------
switch ($Command) {
  "check" {
    if ($env:LLM_BASE_URL -and $env:LLM_API_KEY) {
      Test-Endpoint "chat 端点" $env:LLM_BASE_URL $env:LLM_CHAT_MODEL
    } else {
      Write-Host "跳过 chat 端点检查（未设 LLM_BASE_URL / LLM_API_KEY）" -ForegroundColor Yellow
    }
    Test-Endpoint "向量端点" $env:LLM_EMBED_BASE_URL $env:LLM_EMBED_MODEL
    if ($env:LLM_VLM_MODEL) {
      Write-Host "多模态识图：$($env:LLM_VLM_MODEL)" -ForegroundColor Green
    }
  }
  "start" {
    Test-Endpoint "chat 端点" $env:LLM_BASE_URL $env:LLM_CHAT_MODEL
    Test-Endpoint "向量端点" $env:LLM_EMBED_BASE_URL $env:LLM_EMBED_MODEL
    npm run eval
  }
  "mcp" {
    # serving 只用向量，不需要 chat
    Test-Endpoint "向量端点" $env:LLM_EMBED_BASE_URL $env:LLM_EMBED_MODEL
    npm run kb:mcp
  }
  "ingest" {
    if (-not $Path) {
      Write-Host "用法：.\run.ps1 ingest <物料目录>" -ForegroundColor Yellow
      exit 1
    }
    Test-Endpoint "chat 端点" $env:LLM_BASE_URL $env:LLM_CHAT_MODEL
    Test-Endpoint "向量端点" $env:LLM_EMBED_BASE_URL $env:LLM_EMBED_MODEL
    npm run ingest -- $Path
  }
  "ask" {
    if (-not $Path) {
      Write-Host '用法：.\run.ps1 ask "你的问题"' -ForegroundColor Yellow
      exit 1
    }
    Test-Endpoint "chat 端点" $env:LLM_BASE_URL $env:LLM_CHAT_MODEL
    Test-Endpoint "向量端点" $env:LLM_EMBED_BASE_URL $env:LLM_EMBED_MODEL
    npm run ask -- $Path @Rest
  }
  default {
    Write-Host "用法：.\run.ps1 [check|start|mcp|ingest <目录>]" -ForegroundColor Yellow
    exit 1
  }
}
