#!/usr/bin/env python3
"""
批量计算每日因子并写入DolphinDB
================================

为每只股票计算每个交易日的因子数据

使用方法:
    python scripts/batch_calc_all_factors.py --codes sh.600519
    python scripts/batch_calc_all_factors.py --all --start 2025-01-01 --end 2026-07-01
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dolphindb_reader import DolphinDBReader
from src.factors.technical import FactorCalculator
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calculate_daily_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日因子并返回完整DataFrame

    Args:
        df: 日线数据

    Returns:
        DataFrame 包含所有日期的因子
    """
    calc = FactorCalculator()

    # 计算所有因子
    factors_df = calc.calculate_all_factors(df)

    if factors_df is None or len(factors_df) == 0:
        return pd.DataFrame()

    # 计算综合评分和信号
    scores = []
    signals = []
    for i in range(len(factors_df)):
        row_df = factors_df.iloc[:i+1]
        score = calc.get_composite_score(row_df)
        signal = calc.generate_signal(row_df)
        scores.append(score)
        signals.append(signal)

    factors_df['composite_score'] = scores
    factors_df['signal'] = signals

    return factors_df


def get_dolphindb_connection():
    """获取DolphinDB连接"""
    return DolphinDBReader()


def transform_for_dolphindb(df: pd.DataFrame) -> pd.DataFrame:
    """转换DataFrame格式以适应DolphinDB"""
    df = df.copy()

    # 确保有必要的列
    if 'trade_date' not in df.columns and 'date' in df.columns:
        df['trade_date'] = df['date']

    # 只选择需要的列
    needed_cols = ['trade_date', 'security_id', 'composite_score', 'signal']
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols].copy()

    # 转换日期格式
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y.%m.%d')

    return df


def process_and_write_stock(reader: DolphinDBReader, code: str, start_date: str, end_date: str) -> int:
    """
    处理单只股票的所有日期因子并写入DolphinDB

    Returns:
        写入的记录数
    """
    try:
        # 从DolphinDB加载数据
        sd = start_date.replace('-', '.')
        ed = end_date.replace('-', '.')
        query = f'''
            select * from loadTable("dfs://quant_evo_market", "daily_k")
            where security_id = "{code}" and trade_date between {sd} and {ed}
            order by trade_date
        '''

        df = reader.execute(query)

        if df is None or len(df) == 0:
            return 0

        # 重命名列
        df = df.rename(columns={'trade_date': 'date', 'security_id': 'code'})

        # 计算每日因子
        factors_df = calculate_daily_factors(df)

        if len(factors_df) == 0:
            return 0

        # 准备输出
        factors_df['security_id'] = code
        factors_df['trade_date'] = factors_df['date']

        # 转换并写入
        output_df = transform_for_dolphindb(factors_df)

        if len(output_df) > 0:
            reader.writer.session.upload({'df': output_df})
            reader.writer.session.run(
                f'append!(loadTable("dfs://quant_evo_factors", "factor_daily"), df)'
            )
            return len(output_df)

    except Exception as e:
        print(f"  错误: {e}")

    return 0


def process_stock_codes(codes: list, reader: DolphinDBReader, start_date: str, end_date: str) -> dict:
    """批量处理多只股票"""
    total = 0
    success = 0
    failed = []

    for i, code in enumerate(codes):
        print(f"[{i+1}/{len(codes)}] {code}...", end=' ')

        count = process_and_write_stock(reader, code, start_date, end_date)
        if count > 0:
            print(f"✓ {count} 条")
            success += 1
            total += count
        else:
            print(f"✗")
            failed.append(code)

    return {
        'total': total,
        'success': success,
        'failed': failed
    }


def get_stock_codes(reader: DolphinDBReader, limit: int = None) -> list:
    """获取DolphinDB中的股票列表"""
    stocks = reader.execute('select distinct security_id from loadTable("dfs://quant_evo_market", "daily_k")')
    codes = stocks['security_id'].tolist()
    if limit:
        codes = codes[:limit]
    return codes


def verify_dolphindb_data(reader: DolphinDBReader):
    """验证DolphinDB因子数据"""
    print("\n=== DolphinDB 因子数据验证 ===\n")

    try:
        cnt = reader.execute('select count(*) as cnt from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"factor_daily 总记录数: {cnt.iloc[0][0]:,}")

        min_date = reader.execute('select min(trade_date) as d from loadTable("dfs://quant_evo_factors", "factor_daily")')
        max_date = reader.execute('select max(trade_date) as d from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"日期范围: {min_date.iloc[0][0]} ~ {max_date.iloc[0][0]}")

        stocks = reader.execute('select count(distinct security_id) as cnt from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"覆盖股票数: {stocks.iloc[0][0]}")

        signals = reader.execute('''
            select signal, count(*) as cnt
            from loadTable("dfs://quant_evo_factors", "factor_daily")
            group by signal
        ''')
        if signals is not None and len(signals) > 0:
            print(f"\n信号分布:")
            for _, row in signals.iterrows():
                print(f"  {row['signal']}: {row['cnt']}")

        scores = reader.execute('''
            select avg(composite_score) as avg_score
            from loadTable("dfs://quant_evo_factors", "factor_daily")
        ''')
        if scores is not None and len(scores) > 0:
            print(f"\n平均评分: {scores.iloc[0]['avg_score']:.1f}")

    except Exception as e:
        print(f"验证失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='批量计算每日因子并写入DolphinDB')
    parser.add_argument('--all', action='store_true', help='处理全部股票')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--limit', type=int, default=10, help='限制股票数量')
    parser.add_argument('--start', type=str, default='2025-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2026-07-01', help='结束日期')

    args = parser.parse_args()

    reader = get_dolphindb_connection()

    if args.codes:
        codes = args.codes
    elif args.all:
        codes = get_stock_codes(reader)
    else:
        codes = get_stock_codes(reader, limit=args.limit)

    print(f"\n{'='*60}")
    print("批量计算每日因子 -> DolphinDB")
    print(f"{'='*60}")
    print(f"股票数量: {len(codes)}")
    print(f"日期范围: {args.start} ~ {args.end}")

    result = process_stock_codes(codes, reader, args.start, args.end)

    print(f"\n{'='*60}")
    print("完成汇总")
    print(f"{'='*60}")
    print(f"成功: {result['success']} 只股票")
    print(f"总记录: {result['total']:,} 条")
    if result['failed']:
        print(f"失败: {len(result['failed'])} 只")
        print(f"失败列表: {result['failed'][:10]}")

    verify_dolphindb_data(reader)


if __name__ == '__main__':
    main()