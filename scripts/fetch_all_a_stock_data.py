#!/usr/bin/env python3
"""
批量获取A股全部历史数据
===================
获取所有A股的历史K线数据

使用方法:
    python scripts/fetch_all_a_stock_data.py --limit 500
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import json
import time

sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import akshare as ak
import baostock as bs
import pandas as pd
import numpy as np

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
MODEL_DIR = f'{DATA_DIR}/models'

os.makedirs(CACHE_DIR, exist_ok=True)

# 目标日期范围
START_DATE = '2024-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')


def get_all_a_stocks():
    """获取全部A股列表"""
    print("获取A股列表...")
    df = ak.stock_info_a_code_name()

    # 过滤ST和退市
    df = df[~df['name'].str.contains('ST|退市|暂停上市', na=False)]

    # 添加前缀
    df['code'] = df['code'].apply(
        lambda x: f"sh.{x}" if x.startswith('6') else f"sz.{x}"
    )

    print(f"共 {len(df)} 只A股")
    return df


def fetch_kline_baostock(code: str) -> pd.DataFrame:
    """用baostock获取K线数据"""
    bs_code = code.replace('sh.', '').replace('sz.', '')
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,amount",
        start_date=START_DATE.replace('-', ''),
        end_date=END_DATE.replace('-', ''),
        frequency="d",
        adjustflag="2"  # 前复权
    )

    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())

    if not data:
        return None

    df = pd.DataFrame(data, columns=rs.fields)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # 过滤无效数据
    df = df.dropna(subset=['close', 'volume'])
    df = df[df['volume'] > 0]

    if len(df) < 60:
        return None

    return df


def save_kline(code: str, df: pd.DataFrame):
    """保存K线数据到parquet"""
    code_fmt = code.replace('.', '_')
    filepath = f'{CACHE_DIR}/kline_{code_fmt}.parquet'
    df.to_parquet(filepath, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='限制数量，0表示全部')
    parser.add_argument('--test', action='store_true', help='测试模式，只获取10只')
    args = parser.parse_args()

    # 登录baostock
    lg = bs.login()
    print(f"Baostock登录: {lg.error_msg}")

    # 获取股票列表
    stocks_df = get_all_a_stocks()

    if args.test:
        stocks_df = stocks_df.head(10)
        print(f"测试模式: 只获取 {len(stocks_df)} 只")
    elif args.limit > 0:
        stocks_df = stocks_df.head(args.limit)
        print(f"限制: {len(stocks_df)} 只")

    # 统计
    sh_count = len(stocks_df[stocks_df['code'].str.startswith('sh.')])
    sz_count = len(stocks_df[stocks_df['code'].str.startswith('sz.')])
    print(f"\n上交所: {sh_count}, 深交所: {sz_count}")

    # 获取数据
    success = 0
    fail = 0
    skip = 0

    for idx, row in stocks_df.iterrows():
        code = row['code']
        name = row['name']
        code_fmt = code.replace('.', '_')
        filepath = f'{CACHE_DIR}/kline_{code_fmt}.parquet'

        # 检查是否已有数据
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            if len(existing) >= 200:
                skip += 1
                continue

        # 获取数据
        try:
            df = fetch_kline_baostock(code)
            if df is not None and len(df) >= 60:
                save_kline(code, df)
                success += 1
                if success % 50 == 0:
                    print(f"进度: {success} 只成功, {fail} 失败, {skip} 跳过")
            else:
                fail += 1
        except Exception as e:
            fail += 1

        # 每100只休息一下，避免请求过快
        if (success + fail) % 100 == 0:
            time.sleep(0.5)

    bs.logout()

    print(f"\n完成!")
    print(f"成功: {success}, 失败: {fail}, 跳过: {skip}")


if __name__ == '__main__':
    main()
