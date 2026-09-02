#!/usr/bin/env python3
"""
批量计算因子并写入DolphinDB
============================

计算股票池所有股票的因子，并写入DolphinDB factor_daily表

使用方法:
    python scripts/batch_calc_factors_to_dolphindb.py --all
    python scripts/batch_calc_factors_to_dolphindb.py --codes sh.600519 sz.000858
    python scripts/batch_calc_factors_to_dolphindb.py --verify  # 验证DolphinDB数据
"""

import argparse
import sys
import os
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dolphindb_reader import DolphinDBReader
from src.factors.technical import FactorCalculator
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dolphindb_connection():
    """获取DolphinDB连接"""
    return DolphinDBReader()


def calculate_and_format_factors(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    计算单只股票的因子并格式化为DolphinDB格式

    Args:
        code: 股票代码
        df: 日线数据

    Returns:
        DataFrame 符合 factor_daily 表结构 (4列版本)
    """
    calc = FactorCalculator()

    # 计算所有因子（返回DataFrame，包含原始列+因子列）
    factors_df = calc.calculate_all_factors(df)

    if factors_df is None or len(factors_df) == 0:
        return pd.DataFrame()

    # 获取最新一天的因子
    latest_row = factors_df.iloc[-1]
    trade_date = latest_row.get('date', latest_row.get('trade_date'))

    if trade_date is None:
        return pd.DataFrame()

    # 计算综合评分和信号
    score = calc.get_composite_score(factors_df)
    signal = calc.generate_signal(factors_df)

    # 构建结果（只包含4列：trade_date, security_id, composite_score, signal）
    result = pd.DataFrame([{
        'trade_date': trade_date,
        'security_id': code,
        'composite_score': score,
        'signal': signal
    }])

    return result


def transform_for_dolphindb(df: pd.DataFrame) -> pd.DataFrame:
    """转换DataFrame格式以适应DolphinDB"""
    df = df.copy()

    # 转换日期格式
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y.%m.%d')

    # 确保数值列是数值类型
    numeric_cols = [c for c in df.columns if c not in ['trade_date', 'security_id', 'signal']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def write_to_dolphindb(reader: DolphinDBReader, df: pd.DataFrame, table_name='factor_daily', db_path='dfs://quant_evo_factors'):
    """写入DolphinDB"""
    if df is None or len(df) == 0:
        return 0

    try:
        df = transform_for_dolphindb(df)
        reader.writer.session.upload({'df': df})
        reader.writer.session.run(f'append!(loadTable("{db_path}", "{table_name}"), df)')
        return len(df)
    except Exception as e:
        print(f"  写入失败: {e}")
        return 0


def batch_calculate_and_write(codes: list, reader: DolphinDBReader, start_date: str = None, end_date: str = None) -> dict:
    """批量计算因子并写入DolphinDB"""
    total = 0
    success = 0
    failed = []

    for i, code in enumerate(codes):
        print(f"[{i+1}/{len(codes)}] {code}...", end=' ')

        try:
            # 从DolphinDB加载数据
            if start_date and end_date:
                sd = start_date.replace('-', '.')
                ed = end_date.replace('-', '.')
                query = f'''
                    select * from loadTable("dfs://quant_evo_market", "daily_k")
                    where security_id = "{code}" and trade_date between {sd} and {ed}
                    order by trade_date
                '''
            else:
                query = f'''
                    select * from loadTable("dfs://quant_evo_market", "daily_k")
                    where security_id = "{code}"
                    order by trade_date
                '''

            df = reader.execute(query)

            if df is None or len(df) == 0 or len(df) < 30:
                print(f"数据不足")
                failed.append(code)
                continue

            # 重命名列
            df = df.rename(columns={'trade_date': 'date', 'security_id': 'code'})

            # 计算因子
            factor_df = calculate_and_format_factors(code, df)

            if len(factor_df) > 0:
                # 写入DolphinDB
                count = write_to_dolphindb(reader, factor_df)
                if count > 0:
                    score = factor_df['composite_score'].iloc[0]
                    signal = factor_df['signal'].iloc[0]
                    print(f"✓ signal={signal}, score={score:.1f}")
                    success += 1
                    total += count
                else:
                    print(f"写入失败")
                    failed.append(code)
            else:
                print(f"因子计算失败")
                failed.append(code)

        except Exception as e:
            print(f"错误: {e}")
            failed.append(code)

    return {
        'total': total,
        'success': success,
        'failed': failed
    }


def verify_dolphindb_data(reader: DolphinDBReader):
    """验证DolphinDB因子数据"""
    print("\n=== DolphinDB 因子数据验证 ===\n")

    try:
        # 统计
        cnt = reader.execute('select count(*) as cnt from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"factor_daily 总记录数: {cnt.iloc[0][0]:,}")

        # 时间范围
        min_date = reader.execute('select min(trade_date) as d from loadTable("dfs://quant_evo_factors", "factor_daily")')
        max_date = reader.execute('select max(trade_date) as d from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"日期范围: {min_date.iloc[0][0]} ~ {max_date.iloc[0][0]}")

        # 股票数量
        stocks = reader.execute('select distinct security_id from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"覆盖股票数: {len(stocks)}")

        # 信号分布
        signals = reader.execute('''
            select signal, count(*) as cnt
            from loadTable("dfs://quant_evo_factors", "factor_daily")
            group by signal
        ''')
        if signals is not None and len(signals) > 0:
            print(f"\n信号分布:")
            for _, row in signals.iterrows():
                print(f"  {row['signal']}: {row['cnt']}")

        # 评分统计
        scores = reader.execute('''
            select avg(composite_score) as avg_score,
                   min(composite_score) as min_score,
                   max(composite_score) as max_score
            from loadTable("dfs://quant_evo_factors", "factor_daily")
        ''')
        if scores is not None and len(scores) > 0:
            print(f"\n评分统计:")
            print(f"  平均: {scores.iloc[0]['avg_score']:.1f}")
            print(f"  最低: {scores.iloc[0]['min_score']:.1f}")
            print(f"  最高: {scores.iloc[0]['max_score']:.1f}")

        # 最新数据预览
        print(f"\n最新数据:")
        latest = reader.execute('select top 5 * from loadTable("dfs://quant_evo_factors", "factor_daily") order by trade_date desc')
        if latest is not None and len(latest) > 0:
            print(latest[['trade_date', 'security_id', 'composite_score', 'signal']])

    except Exception as e:
        print(f"验证失败: {e}")


def get_stock_codes(reader: DolphinDBReader, limit: int = None) -> list:
    """获取DolphinDB中的股票列表"""
    stocks = reader.execute('select distinct security_id from loadTable("dfs://quant_evo_market", "daily_k")')
    codes = stocks['security_id'].tolist()
    if limit:
        codes = codes[:limit]
    return codes


def main():
    parser = argparse.ArgumentParser(description='批量计算因子并写入DolphinDB')
    parser.add_argument('--all', action='store_true', help='计算全部股票')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--limit', type=int, default=None, help='限制股票数量')
    parser.add_argument('--verify', action='store_true', help='仅验证数据')
    parser.add_argument('--start', type=str, default='2024-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2026-07-01', help='结束日期')

    args = parser.parse_args()

    reader = get_dolphindb_connection()

    if args.verify:
        verify_dolphindb_data(reader)
        return

    # 获取股票列表
    if args.codes:
        codes = args.codes
    elif args.all:
        codes = get_stock_codes(reader, limit=args.limit)
    else:
        codes = get_stock_codes(reader, limit=10)

    print(f"\n{'='*60}")
    print("批量因子计算 -> DolphinDB")
    print(f"{'='*60}")
    print(f"股票数量: {len(codes)}")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"目标表: dfs://quant_evo_factors/factor_daily")

    result = batch_calculate_and_write(codes, reader, args.start, args.end)

    print(f"\n{'='*60}")
    print("完成汇总")
    print(f"{'='*60}")
    print(f"成功: {result['success']} 只")
    print(f"失败: {len(result['failed'])} 只")
    if result['failed']:
        print(f"失败列表: {result['failed'][:10]}")

    # 验证
    print()
    verify_dolphindb_data(reader)


if __name__ == '__main__':
    main()