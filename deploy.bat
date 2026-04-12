@echo off
:: Windows CMD Wrapper for OpenEnv Push
:: This explicitly forces Python to use standard UTF-8 encoding in the CLI
:: to prevent fatal "charmap" crashes when printing emojis or complex UI characters.

set PYTHONIOENCODING=utf-8
echo [INFO] UTF-8 CLI encoding engaged. Starting deployment...
openenv push %*
