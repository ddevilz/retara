#!/bin/zsh
cd /Users/devashish/Desktop/personal/deutsche-hackathon/backend
set -a; source ../.env; set +a
export MAGENTA_MODEL_LARGE=llama-3.1-8b-instant
exec uv run magenta ablation -n 10000 --seed 42
