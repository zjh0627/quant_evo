#!/bin/bash
# 启动实时交易监控

cd /Users/keira/project/claude/quant_evo
.venv/bin/python scripts/run_realtime_loop.py >> data/cache/realtime_loop.log 2>&1 &

echo "实时交易监控已启动 (PID: $!)"
echo "日志: data/cache/realtime_loop.log"
