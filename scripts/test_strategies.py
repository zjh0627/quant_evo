#!/usr/bin/env python3
"""
策略对比回测脚本
================

对比各策略在2024-07-01到2026-07-10期间的表现
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
from datetime import datetime

from src.data.data_pipeline import DataPipeline
from src.strategies.rsi_strategy import RSIStrategyGenerator, generate_signals_for_stocks as rsi_signals
from src.strategies.ma_crossover_strategy import MACrossoverStrategyGenerator, generate_signals_for_stocks as ma_signals
from src.strategies.mean_reversion_strategy import MeanReversionStrategyGenerator, generate_signals_for_stocks as mr_signals
from src.strategies.momentum_strategy import MomentumStrategyGenerator, generate_signals_for_stocks as momentum_signals
from src.strategies.breakout_strategy import BreakoutStrategyGenerator, generate_signals_for_stocks as breakout_signals
from src.backtest.light_backtest import LightBacktest


def load_stock_data(codes, start_date='2024-07-01', end_date='2026-07-10'):
    """加载股票数据"""
    pipeline = DataPipeline()
    stock_data = {}

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 60:
            # 过滤日期
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if len(df) > 20:
                stock_data[code] = df

    return stock_data


def generate_all_signals(stock_data):
    """生成所有策略信号"""
    strategies = {
        'RSI(平衡)': rsi_signals(stock_data, RSIStrategyGenerator.create_balanced()),
        '均线交叉(中期)': ma_signals(stock_data, MACrossoverStrategyGenerator.create_medium_term()),
        '均值回归(平衡)': mr_signals(stock_data, MeanReversionStrategyGenerator.create_balanced()),
        '动量(中期)': momentum_signals(stock_data, MomentumStrategyGenerator.create_medium_term()),
        '突破(中期)': breakout_signals(stock_data, BreakoutStrategyGenerator.create_medium_term()),
    }

    return strategies


def backtest_strategy(name, stock_data, signals_dict):
    """回测单个策略"""
    print(f"\n{'='*60}")
    print(f"回测: {name}")
    print(f"{'='*60}")

    if not signals_dict:
        print("  无信号数据")
        return None

    bt = LightBacktest(
        initial_capital=1000000,
        commission=0.0003,
        slippage=0.0001,
        position_size=0.1,
        max_positions=5
    )

    try:
        result = bt.run(
            stock_data,
            signals_dict,
            start_date='2024-07-01',
            end_date='2026-07-10'
        )

        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益率: {result.annual_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  交易次数: {result.total_trades}")
        print(f"  胜率: {result.win_rate:.2%}")

        return {
            'name': name,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            'sharpe': result.sharpe_ratio,
            'trades': result.total_trades,
            'win_rate': result.win_rate
        }
    except Exception as e:
        print(f"  回测失败: {e}")
        return None


def main():
    print("=" * 70)
    print("量化策略对比回测")
    print("=" * 70)
    print(f"回测期间: 2024-07-01 至 2026-07-10")
    print(f"初始资金: ¥1,000,000")

    # 加载股票池
    pipeline = DataPipeline()
    pool_codes = pipeline.get_stock_pool()[:30]  # 取前30只

    print(f"\n加载股票: {len(pool_codes)} 只")
    stock_data = load_stock_data(pool_codes)
    print(f"实际加载: {len(stock_data)} 只")

    if len(stock_data) < 5:
        print("股票数据不足,使用默认股票")
        default_codes = ['sh.600519', 'sh.600036', 'sh.601318', 'sh.600276', 'sh.601012']
        stock_data = load_stock_data(default_codes)

    # 生成各策略信号
    print("\n生成策略信号...")
    strategies = generate_all_signals(stock_data)

    # 回测各策略
    results = []
    for name, signals in strategies.items():
        result = backtest_strategy(name, stock_data, signals)
        if result:
            results.append(result)

    # 对比总结
    print("\n" + "=" * 70)
    print("策略对比总结")
    print("=" * 70)
    print(f"{'策略':<20} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'交易次数':>10} {'胜率':>8}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['total_return'], reverse=True):
        print(f"{r['name']:<20} {r['total_return']:>10.2%} {r['annual_return']:>10.2%} "
              f"{r['max_drawdown']:>10.2%} {r['sharpe']:>8.2f} {r['trades']:>10} {r['win_rate']:>8.2%}")

    # 找最优策略
    if results:
        best = max(results, key=lambda x: x['total_return'])
        print(f"\n最优策略: {best['name']} (总收益 {best['total_return']:.2%})")


if __name__ == '__main__':
    main()
