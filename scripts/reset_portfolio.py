#!/usr/bin/env python3
"""
量化账户重置脚本
================
清空所有账户数据：持仓、交易历史、权益曲线
账户金额恢复初始值：200万
"""

import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
CACHE_DIR = f'{PROJECT_ROOT}/data/cache'

# 文件路径
PORTFOLIO_FILE = f'{CACHE_DIR}/virtual_portfolio.json'
POSITION_FILE = f'{CACHE_DIR}/positions.json'
TRADE_HISTORY_FILE = f'{CACHE_DIR}/real_time_trades.json'
EQUITY_CURVE_FILE = f'{CACHE_DIR}/equity_curve.json'

INITIAL_CASH = 2000000.0  # 初始资金200万


def reset_portfolio():
    """重置量化账户"""
    print("=" * 60)
    print("量化账户重置")
    print("=" * 60)

    # 1. 重置 virtual_portfolio.json
    portfolio_data = {
        "cash": INITIAL_CASH,
        "total_value": INITIAL_CASH,
        "positions": [],
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "equity_curve": []
    }
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 重置 {PORTFOLIO_FILE}")
    print(f"  - 现金: {INITIAL_CASH:,.2f}")
    print(f"  - 总价值: {INITIAL_CASH:,.2f}")
    print(f"  - 持仓: 0")
    print(f"  - 权益曲线: 已清空")

    # 2. 重置 positions.json
    with open(POSITION_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False)
    print(f"✓ 重置 {POSITION_FILE}")
    print(f"  - 持仓数量: 0")

    # 3. 重置 real_time_trades.json (交易历史)
    with open(TRADE_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False)
    print(f"✓ 重置 {TRADE_HISTORY_FILE}")
    print(f"  - 交易记录: 0")

    print("=" * 60)
    print("账户重置完成！")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return True


def verify_reset():
    """验证重置结果"""
    print("\n验证重置结果:")
    print("-" * 40)

    # 检查 virtual_portfolio
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
    print(f"virtual_portfolio.json:")
    print(f"  cash: {portfolio['cash']:,.2f}")
    print(f"  total_value: {portfolio['total_value']:,.2f}")
    print(f"  positions: {len(portfolio['positions'])}")
    print(f"  equity_curve: {len(portfolio['equity_curve'])} 条记录")

    # 检查 positions
    with open(POSITION_FILE, 'r') as f:
        positions = json.load(f)
    print(f"\npositions.json: {len(positions)} 条持仓")

    # 检查交易历史
    with open(TRADE_HISTORY_FILE, 'r') as f:
        trades = json.load(f)
    print(f"real_time_trades.json: {len(trades)} 条交易记录")

    # 返回是否全部正确
    all_correct = (
        portfolio['cash'] == INITIAL_CASH and
        portfolio['total_value'] == INITIAL_CASH and
        len(portfolio['positions']) == 0 and
        len(portfolio['equity_curve']) == 0 and
        len(positions) == 0 and
        len(trades) == 0
    )

    if all_correct:
        print("\n✓ 所有数据已正确重置!")
    else:
        print("\n✗ 重置可能存在问题，请检查!")

    return all_correct


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_reset()
    else:
        reset_portfolio()
        verify_reset()
