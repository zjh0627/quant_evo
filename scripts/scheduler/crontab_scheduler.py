#!/usr/bin/env python3
"""
定时任务调度器
==============

自动化定时任务配置和管理:
1. 数据更新 (每日 15:30 收盘后)
2. 因子计算 (每日 16:00)
3. 信号生成 (每日 16:30)
4. 组合监控 (盘中每30分钟)

使用方法:
    # 查看当前配置
    python crontab_scheduler.py show

    # 生成 crontab 配置
    python crontab_scheduler.py generate

    # 添加到 crontab
    python crontab_scheduler.py install

    # 移除
    python crontab_scheduler.py uninstall
"""

import os
import sys
import subprocess
from datetime import datetime
from typing import List, Dict

# 项目路径
PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
PYTHON_BIN = '/Users/keira/project/claude/quant_evo/.venv39/bin/python'


class CronJob:
    """Cron 任务"""

    def __init__(self, name: str, schedule: str, command: str, description: str = ""):
        self.name = name
        self.schedule = schedule  # cron 表达式
        self.command = command
        self.description = description

    def to_cron_line(self) -> str:
        return f"# {self.name}: {self.description}\n{self.schedule} {self.command}"


# 定义定时任务
CRON_JOBS = [
    # 数据更新 - 收盘后 15:30 (A股收盘)
    CronJob(
        name="Data Update - Daily",
        schedule="30 15 * * 1-5",  # 周一至周五 15:30
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.data.data_updater --mode daily >> logs/data_update.log 2>&1",
        description="每日收盘后更新行情数据"
    ),

    # 因子计算 - 16:00
    CronJob(
        name="Factor Calculation",
        schedule="0 16 * * 1-5",  # 周一至周五 16:00
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.factors.calculate --all >> logs/factor_calc.log 2>&1",
        description="每日计算技术因子"
    ),

    # 信号生成 - 16:30
    CronJob(
        name="Signal Generation",
        schedule="30 16 * * 1-5",  # 周一至周五 16:30
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.trading.signal_service --once >> logs/signals.log 2>&1",
        description="每日生成交易信号"
    ),

    # 全量数据更新 - 每周日 20:00
    CronJob(
        name="Data Update - Weekly",
        schedule="0 20 * * 0",  # 周日 20:00
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.data.data_updater --mode full >> logs/data_update_weekly.log 2>&1",
        description="每周全量更新行情数据"
    ),

    # 组合净值计算 - 盘中每30分钟 (9:30 - 15:00)
    CronJob(
        name="Portfolio Monitor",
        schedule="*/30 9,10,11,12,13,14,15 * * 1-5",  # 周一至周五盘中每30分钟
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.trading.risk_control --check >> logs/portfolio_monitor.log 2>&1",
        description="盘中监控组合风险"
    ),

    # 回测报告生成 - 每周一 09:00
    CronJob(
        name="Backtest Report",
        schedule="0 9 * * 1",  # 周一 09:00
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.backtest.report --weekly >> logs/backtest_report.log 2>&1",
        description="每周生成回测报告"
    ),

    # 数据库维护 - 每周日凌晨 3:00
    CronJob(
        name="Database Maintenance",
        schedule="0 3 * * 0",  # 周日 03:00
        command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.data.maintenance --cleanup >> logs/db_maintenance.log 2>&1",
        description="数据库清理和优化"
    ),
]


def generate_crontab() -> str:
    """生成 crontab 内容"""
    lines = [
        "# ============================================",
        "# quant_evo 定时任务配置",
        f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# ============================================",
        "",
        "# 环境变量",
        f"PATH={os.environ.get('PATH', '')}",
        f"HOME=/Users/keira",
        "",
        "# 项目路径",
        f"PROJECT_ROOT={PROJECT_ROOT}",
        "",
    ]

    lines.append("#" + "=" * 60)
    lines.append("# 任务列表")
    lines.append("#" + "=" * 60)
    lines.append("")

    for job in CRON_JOBS:
        lines.append(job.to_cron_line())
        lines.append("")

    return "\n".join(lines)


def show_current_crontab():
    """显示当前 crontab"""
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            print("当前 crontab:")
            print(result.stdout)
        else:
            print("当前没有 crontab 配置")
    except Exception as e:
        print(f"读取 crontab 失败: {e}")


def install_crontab():
    """安装 crontab"""
    crontab_content = generate_crontab()

    # 创建 logs 目录
    os.makedirs(f"{PROJECT_ROOT}/logs", exist_ok=True)

    # 写入临时文件
    tmp_file = f"{PROJECT_ROOT}/.crontab_temp"
    with open(tmp_file, 'w') as f:
        f.write(crontab_content)

    # 安装
    try:
        subprocess.run(['crontab', tmp_file], check=True)
        print("crontab 安装成功!")
        show_current_crontab()
    except Exception as e:
        print(f"crontab 安装失败: {e}")
    finally:
        os.remove(tmp_file)


def uninstall_crontab():
    """卸载 crontab"""
    try:
        subprocess.run(['crontab', '-r'], check=False)
        print("crontab 已移除")
    except Exception as e:
        print(f"crontab 移除失败: {e}")


def show_jobs():
    """显示任务列表"""
    print("=" * 80)
    print("quant_evo 定时任务列表")
    print("=" * 80)

    for job in CRON_JOBS:
        print(f"\n【{job.name}】")
        print(f"  描述: {job.description}")
        print(f"  时间: {job.schedule}")
        print(f"  命令: {job.command}")

    print("\n" + "=" * 80)
    print("\n使用说明:")
    print("  python crontab_scheduler.py show      # 显示任务列表")
    print("  python crontab_scheduler.py generate   # 生成 crontab 配置")
    print("  python crontab_scheduler.py install   # 安装到 crontab")
    print("  python crontab_scheduler.py uninstall  # 移除 crontab")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_jobs()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'show':
        show_current_crontab()
        show_jobs()

    elif cmd == 'generate':
        print(generate_crontab())

    elif cmd == 'install':
        print("安装 crontab...")
        install_crontab()

    elif cmd == 'uninstall':
        print("移除 crontab...")
        uninstall_crontab()

    else:
        print(f"未知命令: {cmd}")
        show_jobs()
