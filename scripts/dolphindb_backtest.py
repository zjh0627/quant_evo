#!/usr/bin/env python3
"""
DolphinDB 数据驱动回测
======================

使用DolphinDB数据进行策略回测

使用方法:
    python scripts/dolphindb_backtest.py --start 2026-01-01 --end 2026-07-01
    python scripts/dolphindb_backtest.py --all-stocks --start 2025-01-01
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dolphindb_reader import DolphinDBReader
from src.backtest.light_backtest import LightBacktest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_market_data(reader: DolphinDBReader, codes: list, start_date: str, end_date: str) -> dict:
    """从DolphinDB加载市场数据"""
    stock_data = {}

    for code in codes:
        try:
            sd = start_date.replace('-', '.')
            ed = end_date.replace('-', '.')

            query = f'''
                select * from loadTable("dfs://quant_evo_market", "daily_k")
                where security_id = "{code}" and trade_date between {sd} and {ed}
                order by trade_date
            '''

            df = reader.execute(query)

            if df is not None and len(df) > 0:
                df = df.rename(columns={'trade_date': 'date', 'security_id': 'code'})
                stock_data[code] = df

        except Exception as e:
            print(f"  加载 {code} 失败: {e}")

    return stock_data


def load_factor_signals(reader: DolphinDBReader, codes: list, start_date: str, end_date: str) -> dict:
    """从DolphinDB加载因子信号"""
    signals = {}

    for code in codes:
        try:
            sd = start_date.replace('-', '.')
            ed = end_date.replace('-', '.')

            query = f'''
                select * from loadTable("dfs://quant_evo_factors", "factor_daily")
                where security_id = "{code}" and trade_date between {sd} and {ed}
                order by trade_date
            '''

            df = reader.execute(query)

            if df is not None and len(df) > 0:
                df = df.rename(columns={'trade_date': 'date', 'security_id': 'code'})
                signals[code] = df[['date', 'signal', 'composite_score']]

        except Exception as e:
            pass

    return signals


def generate_signals_from_scores(stock_data: dict, scores_dict: dict) -> dict:
    """基于综合评分生成交易信号"""
    all_signals = {}

    for code, df in stock_data.items():
        if code not in scores_dict:
            continue

        scores_df = scores_dict[code]
        if scores_df is None or len(scores_df) == 0:
            continue

        # 合并数据
        df = df.copy()
        df = df.merge(scores_df[['date', 'composite_score']], on='date', how='left')

        # 生成信号
        df['score'] = df['composite_score'].fillna(50)
        df['signal'] = 'HOLD'
        df.loc[df['score'] >= 70, 'signal'] = 'BUY'
        df.loc[df['score'] <= 40, 'signal'] = 'SELL'

        all_signals[code] = df[['date', 'signal', 'score']]

    return all_signals


def run_backtest(stock_data: dict, signals: dict, start_date: str, end_date: str) -> dict:
    """运行回测"""
    if not stock_data:
        return None

    bt = LightBacktest(initial_capital=1000000)
    result = bt.run(stock_data, signals, start_date, end_date)

    return result


def backtest_portfolio(reader: DolphinDBReader, codes: list, start_date: str, end_date: str,
                      min_score: int = 60, max_positions: int = 10) -> dict:
    """组合回测"""
    print(f"\n加载市场数据 ({len(codes)} 只股票)...")

    stock_data = load_market_data(reader, codes, start_date, end_date)
    print(f"加载了 {len(stock_data)} 只股票的市场数据")

    print("加载信号数据...")
    scores_dict = load_factor_signals(reader, codes, start_date, end_date)
    print(f"加载了 {len(scores_dict)} 只股票的信号数据")

    if not stock_data:
        print("没有市场数据")
        return None

    # 生成信号
    print("生成交易信号...")
    signals = generate_signals_from_scores(stock_data, scores_dict)

    # 运行回测
    print("运行回测...")
    result = run_backtest(stock_data, signals, start_date, end_date)

    return result


def get_all_stock_codes(reader: DolphinDBReader) -> list:
    """获取所有股票代码"""
    try:
        df = reader.execute('select distinct security_id from loadTable("dfs://quant_evo_market", "daily_k")')
        return df['security_id'].tolist()
    except:
        return []


def main():
    parser = argparse.ArgumentParser(description='DolphinDB数据驱动回测')
    parser.add_argument('--start', type=str, default='2026-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2026-07-01', help='结束日期')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--all-stocks', action='store_true', help='使用全部股票')
    parser.add_argument('--limit', type=int, default=50, help='限制股票数量')

    args = parser.parse_args()

    reader = DolphinDBReader()

    # 获取股票列表
    if args.codes:
        codes = args.codes
    elif args.all_stocks:
        codes = get_all_stock_codes(reader)
    else:
        codes = get_all_stock_codes(reader)[:args.limit]

    print(f"\n{'='*60}")
    print(f"DolphinDB 数据驱动回测")
    print(f"{'='*60}")
    print(f"回测区间: {args.start} ~ {args.end}")
    print(f"股票数量: {len(codes)}")

    result = backtest_portfolio(reader, codes, args.start, args.end)

    if result:
        print(f"\n{'='*60}")
        print("回测结果")
        print(f"{'='*60}")
        print(f"总收益率: {result.total_return:.2%}")
        print(f"年化收益率: {result.annual_return:.2%}")
        print(f"夏普比率: {result.sharpe_ratio:.3f}")
        print(f"最大回撤: {result.max_drawdown:.2%}")
        print(f"胜率: {result.win_rate:.1%}")
        print(f"交易次数: {result.total_trades}")
    else:
        print("\n回测失败")

    reader.disconnect()


if __name__ == '__main__':
    main()