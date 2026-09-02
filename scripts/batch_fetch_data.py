#!/usr/bin/env python3
"""
批量获取股票数据
================

获取股票池所有股票的历史数据

使用方法:
    python scripts/batch_fetch_data.py --all
    python scripts/batch_fetch_data.py --batch 20
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import DataPipeline


def batch_fetch(codes, batch_size=20):
    """分批获取数据"""
    pipeline = DataPipeline()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    total = len(codes)
    success = 0
    failed = []

    for i in range(0, total, batch_size):
        batch = codes[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n批次 {batch_num}/{total_batches}: 获取 {len(batch)} 只...")
        print(f"时间范围: {start_date} ~ {end_date}")

        try:
            data = pipeline.fetch_and_save(batch, start_date, end_date, use_cache=True)
            success += len(data)
            print(f"  成功: {len(data)} 只")
        except Exception as e:
            print(f"  批次失败: {e}")
            failed.extend(batch)

    print(f"\n{'='*60}")
    print(f"数据获取完成")
    print(f"成功: {success}/{total} 只")
    if failed:
        print(f"失败: {len(failed)} 只")
        print(f"失败列表: {failed[:10]}...")

    return success, failed


def main():
    parser = argparse.ArgumentParser(description='批量获取股票数据')
    parser.add_argument('--all', action='store_true', help='获取全部股票')
    parser.add_argument('--batch', type=int, default=20, help='每批数量')
    parser.add_argument('--test', type=int, default=0, help='测试数量')

    args = parser.parse_args()

    pipeline = DataPipeline()
    codes = pipeline.get_stock_pool()
    print(f"股票池: {len(codes)} 只")

    if args.all:
        print("全量模式: 获取全部股票")
        success, failed = batch_fetch(codes, args.batch)
    elif args.test > 0:
        test_codes = codes[:args.test]
        print(f"测试模式: 获取前 {args.test} 只")
        success, failed = batch_fetch(test_codes, args.batch)
    else:
        print("使用 --test N 或 --all")
        return

    # 统计
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    parquet_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
    print(f"\nParquet 文件数: {len(parquet_files)}")


if __name__ == '__main__':
    main()
