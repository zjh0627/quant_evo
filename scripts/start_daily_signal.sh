#!/bin/bash
# 每日尾盘信号推送启动器
# 在screen或tmux中运行: bash scripts/start_daily_signal.sh

cd /Users/keira/project/claude/quant_evo

echo "每日尾盘信号推送服务启动"
echo "每天15:50自动推送信号"

while true; do
    NOW=$(date +"%H%M")

    # 周一到周五 15:50-15:55 执行
    if [[ $(date +%u) -le 5 ]] && [[ "$NOW" >= "1540" ]] && [[ "$NOW" < "1555" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 生成尾盘信号..."
        .venv/bin/python scripts/daily_signal_push.py push
        sleep 600  # 10分钟内不重复执行
    fi

    sleep 60  # 每分钟检查一次
done
