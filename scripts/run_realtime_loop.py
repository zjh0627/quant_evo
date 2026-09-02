#!/usr/bin/env python3
"""
实时交易后台运行器
==================
每个交易日 9:25-15:05 每15分钟运行一次实时交易
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import time as time_module
import subprocess
from datetime import datetime, timedelta
from datetime import time as market_time
import os

PYTHON_BIN = '/Users/keira/project/claude/quant_evo/.venv/bin/python'
SCRIPT = '/Users/keira/project/claude/quant_evo/scripts/real_time_trading.py'
LOG_FILE = '/Users/keira/project/claude/quant_evo/data/cache/realtime_loop.log'


def log(msg: str):
    """记录日志"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] {msg}')
    with open(LOG_FILE, 'a') as f:
        f.write(f'[{now}] {msg}\n')


def is_market_time():
    """检查是否在交易时段 (9:25-15:05)"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return market_time(9, 25) <= t <= market_time(15, 5)


def wait_until_next_run(interval=15):
    """等待到下一个运行时间点"""
    now = datetime.now()
    # 计算下一个15分钟节点
    minutes = (now.minute // interval + 1) * interval
    if minutes >= 60:
        next_run = now.replace(minute=0, second=0) + timedelta(hours=1)
    else:
        next_run = now.replace(minute=0, second=0) + timedelta(minutes=minutes)

    # 如果还没到9:25，跳到9:25
    market_start = now.replace(hour=9, minute=25, second=0)
    if now.time() < time(9, 25):
        next_run = market_start

    wait_seconds = (next_run - now).total_seconds()
    if wait_seconds > 0:
        log(f'等待 {wait_seconds:.0f} 秒，下一次运行: {next_run.strftime("%H:%M")}')
        time_module.sleep(min(wait_seconds, 60))  # 最多等60秒


def main():
    log('=' * 60)
    log('实时交易监控启动')
    log('=' * 60)

    consecutive_errors = 0
    max_errors = 3

    while True:
        try:
            if is_market_time():
                log('执行实时交易扫描...')
                result = subprocess.run(
                    [PYTHON_BIN, SCRIPT],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    log('执行成功')
                    consecutive_errors = 0
                else:
                    log(f'执行失败: {result.stderr[:200]}')
                    consecutive_errors += 1

                if consecutive_errors >= max_errors:
                    log(f'连续{max_errors}次失败，休息5分钟')
                    time_module.sleep(300)
                    consecutive_errors = 0
            else:
                now = datetime.now()
                if now.weekday() >= 5:
                    log('周末，休息...')
                    time_module.sleep(3600)  # 1小时后重试
                else:
                    # 非交易时间，等待
                    log(f'非交易时间({now.strftime("%H:%M")})，等待...')
                    time_module.sleep(300)  # 5分钟后重试

            # 等待到下一个15分钟节点
            wait_until_next_run(15)

        except KeyboardInterrupt:
            log('收到中断信号，退出')
            break
        except Exception as e:
            log(f'异常: {e}')
            time_module.sleep(60)


if __name__ == '__main__':
    main()
