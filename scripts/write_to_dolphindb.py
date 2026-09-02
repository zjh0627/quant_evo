#!/usr/bin/env python3
"""
DolphinDB 批量数据写入脚本
===========================

将 Parquet 数据批量写入 DolphinDB

使用方法:
    python scripts/write_to_dolphindb.py --write --table daily_k
    python scripts/write_to_dolphindb.py --write-all
    python scripts/write_to_dolphindb.py --verify
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphindb import session
import pandas as pd

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')


def get_connection():
    """获取 DolphinDB 连接"""
    s = session()
    s.connect('localhost', 8848)
    s.login('admin', '123456')
    return s


def transform_df(df):
    """转换 DataFrame 格式以适应 DolphinDB 表"""
    df = df.copy()

    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y.%m.%d')

    # 股票代码
    df['security_id'] = df['code']

    # 重命名列
    if 'pctChg' in df.columns:
        df['pct_chg'] = df['pctChg']
        df = df.drop('pctChg', axis=1)

    # 补全必填列
    df['is_suspended'] = df['is_suspended'] if 'is_suspended' in df.columns else False
    df['is_limit_up'] = df['is_limit_up'] if 'is_limit_up' in df.columns else False
    df['is_limit_down'] = df['is_limit_down'] if 'is_limit_down' in df.columns else False

    # pre_close: 前一日收盘价
    df['pre_close'] = df['close'].shift(1).fillna(df['close'])

    # adj_factor: 复权因子，默认为 1.0
    df['adj_factor'] = df['adj_factor'] if 'adj_factor' in df.columns else 1.0

    # 选择并排序列
    columns = ['trade_date', 'security_id', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pct_chg', 'is_suspended',
               'is_limit_up', 'is_limit_down', 'pre_close', 'adj_factor']

    # 只保留存在的列
    cols = [c for c in columns if c in df.columns]
    return df[cols]


def write_file(s, file_path, table_name='daily_k', db_path='dfs://quant_evo_market'):
    """写入单个文件"""
    try:
        df = pd.read_parquet(file_path)
        if df is None or len(df) == 0:
            return 0

        df = transform_df(df)
        s.upload({'df': df})
        s.run(f'append!(loadTable("{db_path}", "{table_name}"), df)')
        return len(df)
    except Exception as e:
        print(f"  写入失败 {os.path.basename(file_path)}: {e}")
        return 0


def write_batch(s, files, table_name, db_path):
    """批量写入文件"""
    total = 0
    for f in files:
        count = write_file(s, f, table_name, db_path)
        total += count
    return total


def verify_tables(s):
    """验证所有表"""
    tables = [
        ('dfs://quant_evo_market', 'daily_k'),
        ('dfs://quant_evo_min1k', 'min_1k'),
        ('dfs://quant_evo_min5k', 'min_5k'),
        ('dfs://quant_evo_min15k', 'min_15k'),
        ('dfs://quant_evo_metadata', 'fundamental'),
        ('dfs://quant_evo_mktflag', 'market_flag'),
        ('dfs://quant_evo_factors', 'factor_daily'),
        ('dfs://quant_evo_factor_min', 'factor_min'),
    ]

    print("\n验证 DolphinDB 表:")
    total = 0
    for db, table in tables:
        try:
            result = s.run(f"select count(*) as cnt from loadTable('{db}', '{table}')")
            count = result['cnt'].iloc[0] if isinstance(result, pd.DataFrame) else 0
            print(f"  {db}/{table}: {count:,} 条")
            total += count
        except Exception as e:
            print(f"  {db}/{table}: 错误 - {str(e)[:50]}")
    print(f"\n总记录数: {total:,}")
    return total


def main():
    parser = argparse.ArgumentParser(description='DolphinDB 批量数据写入')
    parser.add_argument('--write', action='store_true', help='写入数据')
    parser.add_argument('--write-all', action='store_true', help='写入所有数据')
    parser.add_argument('--table', type=str, default='daily_k', help='表名')
    parser.add_argument('--db', type=str, default='dfs://quant_evo_market', help='数据库路径')
    parser.add_argument('--verify', action='store_true', help='验证表')
    parser.add_argument('--batch', type=int, default=50, help='每批文件数')

    args = parser.parse_args()

    s = get_connection()

    if args.verify:
        verify_tables(s)
        s.close()
        return

    if args.write or args.write_all:
        print(f"\n{'='*60}")
        print(f"DolphinDB 数据写入")
        print(f"{'='*60}")
        print(f"表: {args.table}")
        print(f"数据库: {args.db}")

        # 获取所有 Parquet 文件
        files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.parquet')]
        print(f"Parquet 文件数: {len(files)}")

        total = 0
        for i in range(0, len(files), args.batch):
            batch = files[i:i+args.batch]
            batch_num = i // args.batch + 1
            total_batches = (len(files) + args.batch - 1) // args.batch

            print(f"\n批次 {batch_num}/{total_batches} ({len(batch)} 文件)...")

            try:
                count = write_batch(s, batch, args.table, args.db)
                total += count
                print(f"  本批写入: {count:,} 条")
            except Exception as e:
                print(f"  批次失败: {e}")

        print(f"\n写入完成: {total:,} 条")

        # 验证
        print("\n验证:")
        verify_tables(s)

    s.close()


if __name__ == '__main__':
    main()
