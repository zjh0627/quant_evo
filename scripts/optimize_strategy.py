#!/usr/bin/env python3
"""
策略优化脚本
============

测试不同策略参数，找出最佳组合

使用方法:
    python scripts/optimize_strategy.py
"""

import argparse
import sys
import os
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import DataPipeline
from src.backtest.light_backtest import LightBacktest
from src.factors.technical import FactorCalculator
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rsi_strategy(df, rsi_buy=35, rsi_sell=65):
    """RSI 策略"""
    df = df.copy()
    calc = FactorCalculator()
    factors = calc.calculate_all_factors(df)

    if 'rsi' in factors:
        rsi = factors['rsi']
        df['rsi'] = rsi
        df['signal'] = 'HOLD'
        df.loc[df['rsi'] < rsi_buy, 'signal'] = 'BUY'
        df.loc[df['rsi'] > rsi_sell, 'signal'] = 'SELL'
        df['score'] = (100 - rsi).clip(0, 100)
    else:
        df['signal'] = 'HOLD'
        df['score'] = 50

    return df


def macd_strategy(df, threshold=0.0):
    """MACD 策略 - 使用 MACD histogram 方向作为评分"""
    df = df.copy()
    calc = FactorCalculator()
    factors = calc.calculate_all_factors(df)

    if 'macd' in factors and 'signal' in factors and 'macd_hist' in factors:
        macd = factors['macd']
        macd_signal = factors['signal']  # MACD signal line
        macd_hist = factors['macd_hist']
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['signal'] = 'HOLD'
        # MACD 金叉买入，死叉卖出
        df.loc[(macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), 'signal'] = 'BUY'
        df.loc[(macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), 'signal'] = 'SELL'
        # 分数: MACD histogram 强度，histogram 为正且增大 = 看涨
        # 使用历史百分位来标准化
        hist_pct = macd_hist.rank(pct=True) * 100
        df['score'] = hist_pct.clip(0, 100)
    else:
        df['signal'] = 'HOLD'
        df['score'] = 50

    return df


def bollinger_strategy(df, bb_buy=0.2, bb_sell=0.8):
    """布林带策略"""
    df = df.copy()
    calc = FactorCalculator()
    factors = calc.calculate_all_factors(df)

    if 'bb_position' in factors:
        bb_pos = factors['bb_position']
        df['bb_position'] = bb_pos
        df['signal'] = 'HOLD'
        df.loc[bb_pos < bb_buy, 'signal'] = 'BUY'
        df.loc[bb_pos > bb_sell, 'signal'] = 'SELL'
        # 分数: 接近下轨(低bb_pos) = 高分
        df['score'] = ((1 - bb_pos) * 100).clip(0, 100)
    else:
        df['signal'] = 'HOLD'
        df['score'] = 50

    return df


def test_strategy(codes, strategy_func, start_date, end_date, strategy_name):
    """测试策略"""
    pipeline = DataPipeline()
    results = []

    for code in codes:
        try:
            df = pipeline.load(code)
            if df is None or len(df) < 60:
                continue

            df = strategy_func(df)
            data_dict = {code: df}
            signals_dict = {code: df[['date', 'signal', 'score']]}

            bt = LightBacktest(initial_capital=1000000)
            result = bt.run(data_dict, signals_dict, start_date, end_date)

            if result and result.total_trades > 0:
                results.append({
                    'code': code,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades
                })
        except:
            pass

    if not results:
        return None

    avg_return = sum(r['total_return'] for r in results) / len(results)
    avg_sharpe = sum(r['sharpe_ratio'] for r in results) / len(results)
    winning_rate = sum(1 for r in results if r['total_return'] > 0) / len(results)

    return {
        'strategy': strategy_name,
        'stocks_tested': len(results),
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'winning_rate': winning_rate,
        'best_stock': max(results, key=lambda x: x['total_return']) if results else None
    }


def main():
    parser = argparse.ArgumentParser(description='策略优化')
    parser.add_argument('--quick', action='store_true', help='快速测试（10只）')
    args = parser.parse_args()

    pipeline = DataPipeline()
    codes = pipeline.get_stock_pool()

    if args.quick:
        codes = codes[:10]

    print(f"\n{'='*60}")
    print("策略优化测试")
    print(f"{'='*60}")
    print(f"测试股票数: {len(codes)}")
    print(f"回测区间: 2024-07-01 ~ 2026-07-01")

    start_date = '2024-07-01'
    end_date = '2026-07-01'

    results = []

    # RSI 策略不同参数
    print("\n--- RSI 策略参数测试 ---")
    for rsi_buy, rsi_sell in [(30, 70), (35, 65), (40, 60), (25, 75), (20, 80)]:
        def make_rsi_strategy(b=rsi_buy, s=rsi_sell):
            return lambda df: rsi_strategy(df, rsi_buy=b, rsi_sell=s)

        result = test_strategy(codes, make_rsi_strategy(), start_date, end_date, f'RSI({rsi_buy},{rsi_sell})')
        if result:
            print(f"RSI({rsi_buy},{rsi_sell}): 胜率={result['winning_rate']:.1%}, 均收益={result['avg_return']:.2%}, 夏普={result['avg_sharpe']:.2f}")
            results.append(result)

    # MACD 策略
    print("\n--- MACD 策略 ---")
    result = test_strategy(codes, macd_strategy, start_date, end_date, 'MACD')
    if result:
        print(f"MACD: 胜率={result['winning_rate']:.1%}, 均收益={result['avg_return']:.2%}, 夏普={result['avg_sharpe']:.2f}")
        results.append(result)

    # 布林带策略
    print("\n--- 布林带策略 ---")
    for bb_buy, bb_sell in [(0.2, 0.8), (0.3, 0.7), (0.1, 0.9)]:
        def make_bb_strategy(b=bb_buy, s=bb_sell):
            return lambda df: bollinger_strategy(df, bb_buy=b, bb_sell=s)

        result = test_strategy(codes, make_bb_strategy(), start_date, end_date, f'BB({bb_buy},{bb_sell})')
        if result:
            print(f"BB({bb_buy},{bb_sell}): 胜率={result['winning_rate']:.1%}, 均收益={result['avg_return']:.2%}, 夏普={result['avg_sharpe']:.2f}")
            results.append(result)

    # 找出最佳策略
    if results:
        best = max(results, key=lambda x: x['avg_sharpe'])
        print(f"\n{'='*60}")
        print(f"最佳策略: {best['strategy']}")
        print(f"平均夏普: {best['avg_sharpe']:.2f}")
        print(f"胜率: {best['winning_rate']:.1%}")
        print(f"平均收益: {best['avg_return']:.2%}")

        # 保存结果
        os.makedirs('data', exist_ok=True)
        with open('data/strategy_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存: data/strategy_comparison.json")


if __name__ == '__main__':
    main()
