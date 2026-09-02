#!/usr/bin/env python3
"""
股票池扩展脚本
==============

将股票池从 18 只扩展到 200 只，覆盖主要行业

使用方法:
    python scripts/expand_stock_pool.py --expand --target 200
"""

import argparse
import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_POOL_FILE = os.path.join(PROJECT_ROOT, 'data', 'stock_pool.json')


def get_stock_list():
    """获取 A 股全市场列表"""
    import akshare as ak
    print("获取 A 股全市场列表...")
    df = ak.stock_info_a_code_name()
    return df


def select_stock_pool(df, target_size=200):
    """选择股票池 - 按代码规律分散"""
    print(f"\n选择 {target_size} 只股票...")

    # 过滤 ST 和退市
    df = df[~df['name'].str.contains('ST|退市|^\\*', regex=True, na=False)]

    # 添加市场前缀
    df['code'] = df['code'].apply(lambda x: f"sh.{x}" if x.startswith('6') else f"sz.{x}")

    # 按代码排序，均匀选取
    # 上交所(6开头)和深交所(0,3开头)交替
    sh_stocks = df[df['code'].str.startswith('sh.')].copy()
    sz_stocks = df[df['code'].str.startswith('sz.')].copy()

    sh_stocks = sh_stocks.sort_values('code')
    sz_stocks = sz_stocks.sort_values('code')

    # 计算选取间隔
    sh_interval = max(1, int(len(sh_stocks) // (target_size * 0.6)))
    sz_interval = max(1, int(len(sz_stocks) // (target_size * 0.4)))

    selected = []
    # 从上交所选取 60%
    for i in range(0, len(sh_stocks), sh_interval):
        if len(selected) >= int(target_size * 0.6):
            break
        selected.append(sh_stocks.iloc[i]['code'])

    # 从深交所选取 40%
    for i in range(0, len(sz_stocks), sz_interval):
        if len(selected) >= target_size:
            break
        selected.append(sz_stocks.iloc[i]['code'])

    print(f"选取完成: {len(selected)} 只")
    return selected


def save_stock_pool(stock_codes, filename=STOCK_POOL_FILE):
    """保存股票池"""
    data = {
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': len(stock_codes),
        'stocks': stock_codes
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"股票池已保存: {filename}")
    return data


def load_stock_pool(filename=STOCK_POOL_FILE):
    """加载股票池"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def show_stock_pool():
    """显示股票池"""
    pool = load_stock_pool()
    if pool:
        print(f"股票池: {pool['count']} 只")
        print(f"更新时间: {pool['last_update']}")
        print("\n前 20 只:")
        for i, code in enumerate(pool['stocks'][:20]):
            print(f"  {i+1}. {code}")
        print(f"\n后 20 只:")
        for i, code in enumerate(pool['stocks'][-20:], len(pool['stocks'])-19):
            print(f"  {i}. {code}")
    else:
        print("股票池文件不存在")


def main():
    parser = argparse.ArgumentParser(description='股票池扩展')
    parser.add_argument('--expand', action='store_true', help='扩展股票池')
    parser.add_argument('--target', type=int, default=200, help='目标数量')
    parser.add_argument('--show', action='store_true', help='显示股票池')

    args = parser.parse_args()

    if args.show:
        show_stock_pool()
        return

    if args.expand:
        print(f"\n{'='*60}")
        print("股票池扩展")
        print(f"{'='*60}")

        df = get_stock_list()
        print(f"全市场: {len(df)} 只")

        stock_codes = select_stock_pool(df, args.target)
        pool = save_stock_pool(stock_codes)

        print(f"\n扩展完成: {pool['count']} 只股票")
    else:
        show_stock_pool()


if __name__ == '__main__':
    main()
