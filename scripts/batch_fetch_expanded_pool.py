#!/usr/bin/env python3
"""
批量获取扩展股票池数据
======================

从 baostock 批量获取扩展股票池的日线数据

使用方法:
    python scripts/batch_fetch_expanded_pool.py
    python scripts/batch_fetch_expanded_pool.py --codes sh.600519 sz.000858
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')


def get_expanded_stock_pool():
    """获取扩展股票池"""
    # Look in parent directory
    pool_path = os.path.join(PROJECT_ROOT, 'data', 'expanded_stock_pool.json')
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            data = json.load(f)
            return data.get('stocks', [])
    return []


def fetch_stock_kline(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取单只股票的K线数据

    Args:
        code: 股票代码 (sh.600519 或 sz.000858)
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)

    Returns:
        DataFrame 包含日线数据
    """
    # 查询数据 - baostock expects YYYY-MM-DD format
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        return pd.DataFrame()

    df = pd.DataFrame(data_list, columns=rs.fields)

    # 转换数据类型
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 添加原始代码列
    df['code_raw'] = code

    return df


def batch_fetch(codes: list, start_date: str, end_date: str, force_update: bool = False) -> dict:
    """
    批量获取多只股票数据

    Args:
        codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        force_update: 是否强制更新已存在的文件

    Returns:
        统计信息 dict
    """
    total = len(codes)
    success = 0
    failed = []
    skipped = 0

    for i, code in enumerate(codes):
        # 检查是否已存在
        filename = f"kline_{code.replace('.', '_')}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath) and not force_update:
            # 检查文件更新时间
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if datetime.now() - mtime < timedelta(days=1):
                skipped += 1
                print(f"[{i+1}/{total}] {code}: 已存在，跳过")
                continue

        print(f"[{i+1}/{total}] {code}...", end=' ', flush=True)

        try:
            df = fetch_stock_kline(code, start_date, end_date)

            if df is not None and len(df) > 0:
                # 保存到CSV
                df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"✓ {len(df)} 条")
                success += 1
            else:
                print(f"✗ 无数据")
                failed.append(code)

        except Exception as e:
            print(f"✗ 错误: {e}")
            failed.append(code)

    return {
        'total': total,
        'success': success,
        'failed': failed,
        'skipped': skipped
    }


def verify_data(codes: list) -> dict:
    """验证已下载的数据"""
    stats = {}
    for code in codes:
        filename = f"kline_{code.replace('.', '_')}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, nrows=5)
            stats[code] = {
                'exists': True,
                'rows': len(pd.read_csv(filepath)),
                'first_date': df['date'].iloc[0] if len(df) > 0 else None,
                'last_date': None  # 需要读取完整文件
            }
        else:
            stats[code] = {'exists': False}

    return stats


def main():
    parser = argparse.ArgumentParser(description='批量获取扩展股票池数据')
    parser.add_argument('--start', type=str, default='2024-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2026-07-29', help='结束日期')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--force', action='store_true', help='强制更新已存在的文件')
    parser.add_argument('--verify', action='store_true', help='仅验证数据')

    args = parser.parse_args()

    # 登录 baostock
    print("登录 baostock...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    print(f"登录成功: {lg.error_msg}")

    # 获取股票列表
    if args.codes:
        codes = args.codes
    else:
        codes = get_expanded_stock_pool()

    print(f"\n{'='*60}")
    print(f"批量获取股票数据")
    print(f"{'='*60}")
    print(f"股票数量: {len(codes)}")
    print(f"日期范围: {args.start} ~ {args.end}")

    if args.verify:
        print("\n验证数据...")
        stats = verify_data(codes)
        exists_count = sum(1 for s in stats.values() if s.get('exists'))
        print(f"已下载: {exists_count}/{len(codes)}")
        for code, stat in stats.items():
            if not stat.get('exists'):
                print(f"  ✗ {code}: 未下载")
    else:
        result = batch_fetch(codes, args.start, args.end, args.force)

        print(f"\n{'='*60}")
        print("完成汇总")
        print(f"{'='*60}")
        print(f"成功: {result['success']} 只")
        print(f"跳过: {result['skipped']} 只")
        print(f"失败: {len(result['failed'])} 只")
        if result['failed']:
            print(f"失败列表: {result['failed'][:10]}")

    bs.logout()


if __name__ == '__main__':
    main()