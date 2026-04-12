# Windows PowerShell Wrapper for OpenEnv Push
# This explicitly forces Python to use standard UTF-8 encoding in the CLI
# to prevent fatal "charmap" crashes when printing emojis or complex UI characters.

$env:PYTHONIOENCODING="utf-8"
Write-Host "[INFO] UTF-8 CLI encoding engaged. Starting deployment..." -ForegroundColor Cyan
openenv push $args
