#!/usr/bin/env python3
"""
批量回测脚本
============

对股票池所有股票进行批量回测

使用方法:
    python scripts/batch_backtest.py --all
    python scripts/batch_backtest.py --codes sh.600519 sz.000858
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import DataPipeline
from src.backtest.light_backtest import LightBacktest
from src.factors.technical import FactorCalculator
from src.data.data_cleaner import DataCleaner
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'backtest_results')


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """根据因子生成信号"""
    df = df.copy()

    calc = FactorCalculator()
    factors = calc.calculate_all_factors(df)

    # RSI 信号
    if 'rsi' in factors:
        rsi = factors['rsi']
        df['rsi'] = rsi
        df['signal'] = 'HOLD'
        df.loc[df['rsi'] < 35, 'signal'] = 'BUY'
        df.loc[df['rsi'] > 65, 'signal'] = 'SELL'
        # score: RSI 低 = 超卖 = 买入机会大 = score 高
        df['score'] = (100 - rsi).clip(0, 100)
    else:
        df['signal'] = 'HOLD'
        df['score'] = 50

    return df


def backtest_stock(code: str, df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """单只股票回测"""
    # 计算因子并生成信号
    df_with_signal = generate_signal(df)

    # 准备数据
    data_dict = {code: df_with_signal}
    signals_dict = {code: df_with_signal[['date', 'signal', 'score']]}

    # 回测
    bt = LightBacktest(initial_capital=1000000)
    result = bt.run(data_dict, signals_dict, start_date, end_date)

    if result:
        return {
            'code': code,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'success': True
        }
    return {'code': code, 'success': False}


def batch_backtest(codes: list, start_date: str, end_date: str, save: bool = True) -> list:
    """批量回测"""
    pipeline = DataPipeline()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    total = len(codes)

    for i, code in enumerate(codes):
        print(f"[{i+1}/{total}] {code}...", end=' ')

        try:
            # 加载数据
            df = pipeline.load(code)
            if df is None or len(df) < 60:
                print(f"数据不足")
                continue

            # 回测
            result = backtest_stock(code, df, start_date, end_date)
            if result['success']:
                results.append(result)
                print(f"收益 {result['total_return']:.2%}, 夏普 {result['sharpe_ratio']:.2f}")
            else:
                print(f"失败")

        except Exception as e:
            print(f"错误: {e}")

    return results


def show_summary(results: list):
    """显示汇总"""
    if not results:
        print("无结果")
        return

    # 计算统计
    returns = [r['total_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]

    positive = sum(1 for r in returns if r > 0)
    winning = sum(1 for r in results if r['win_rate'] > 0.5)

    print(f"\n{'='*60}")
    print("批量回测汇总")
    print(f"{'='*60}")
    print(f"回测股票数: {len(results)}")
    print(f"正收益: {positive} ({positive/len(results):.1%})")
    print(f"夏普>1: {sum(1 for s in sharpes if s > 1)} ({sum(1 for s in sharpes if s > 1)/len(results):.1%})")
    print(f"胜率>50%: {winning} ({winning/len(results):.1%})")
    print(f"\n平均收益: {sum(returns)/len(returns):.2%}")
    print(f"平均夏普: {sum(sharpes)/len(sharpes):.2f}")
    print(f"平均回撤: {sum(drawdowns)/len(drawdowns):.2%}")

    # 显示最佳 10 只
    top10 = sorted(results, key=lambda x: x['total_return'], reverse=True)[:10]
    print(f"\n收益最高 10 只:")
    for r in top10:
        print(f"  {r['code']}: {r['total_return']:.2%} (夏普 {r['sharpe_ratio']:.2f})")

    # 显示最差 5 只
    bottom5 = sorted(results, key=lambda x: x['total_return'])[:5]
    print(f"\n收益最低 5 只:")
    for r in bottom5:
        print(f"  {r['code']}: {r['total_return']:.2%} (夏普 {r['sharpe_ratio']:.2f})")


def main():
    parser = argparse.ArgumentParser(description='批量回测')
    parser.add_argument('--all', action='store_true', help='回测全部股票')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--start', type=str, default='2024-07-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2026-07-01', help='结束日期')

    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.codes:
        codes = args.codes
    elif args.all:
        codes = pipeline.get_stock_pool()
    else:
        codes = ['sh.600519', 'sh.600036', 'sz.000858']

    print(f"\n{'='*60}")
    print("批量回测")
    print(f"{'='*60}")
    print(f"股票数量: {len(codes)}")
    print(f"回测区间: {args.start} ~ {args.end}")

    results = batch_backtest(codes, args.start, args.end)

    show_summary(results)

    # 保存结果
    if results:
        results_file = os.path.join(RESULTS_DIR, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存: {results_file}")


if __name__ == '__main__':
    main()
