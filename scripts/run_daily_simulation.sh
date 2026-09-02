#!/bin/bash
# 模拟盘每日定时任务
# 添加到 crontab: 0 16 * * 1-5 /Users/keira/project/claude/quant_evo/scripts/run_daily_simulation.sh

cd /Users/keira/project/claude/quant_evo

# 激活虚拟环境并运行
.venv39/bin/python src/simulation/evolving_portfolio.py 2>&1

echo "模拟盘已运行: $(date)"