# Run from an extracted VideoSearcher-AgenticAI repository on Windows.
# Requires GitHub CLI (gh) already installed and authenticated: gh auth login.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI (gh) is required. Install it from https://cli.github.com/ and run gh auth login."
}
if (-not (Test-Path ".git")) {
  throw "Git metadata missing. Use the prepared Git repository archive (or initialize and commit manually)."
}
& gh auth status
if ($LASTEXITCODE -ne 0) { throw "Authenticate with gh auth login first." }
& gh repo create "StefanoGarzoni/VideoSearcher-AgenticAI" --private --source . --remote origin --push --description "Agentic video understanding, MCP tools and energy-aware local-LLM benchmarking (academic research)"
if ($LASTEXITCODE -ne 0) { throw "Repository creation or push failed; review the error from GitHub CLI." }
Write-Host "Repository URL: https://github.com/StefanoGarzoni/VideoSearcher-AgenticAI"
