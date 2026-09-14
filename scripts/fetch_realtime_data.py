#!/usr/bin/env python3
"""
带重试机制的数据获取脚本
"""
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import akshare as ak
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List

MAX_RETRIES = 5
INITIAL_DELAY = 2
BACKOFF_FACTOR = 2

def fetch_with_retry(func, *args, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY, **kwargs):
    """带指数退避重试的获取函数"""
    last_exception = None

    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                print(f"  重试成功 (尝试 {attempt + 1})")
            return result
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (BACKOFF_FACTOR ** attempt)
                print(f"  获取失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:50]}, {delay:.1f}秒后重试...")
                time.sleep(delay)
            else:
                print(f"  最终失败: {str(e)[:80]}")

    raise last_exception

def fetch_stock_hist(symbol: str, days: int = 300) -> Optional[pd.DataFrame]:
    """获取单只股票历史数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    df = fetch_with_retry(
        ak.stock_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date
    )
    return df

def fetch_margin_data() -> Optional[pd.DataFrame]:
    """获取融资融券数据"""
    print("\n获取融资融券数据...")
    df = fetch_with_retry(ak.macro_china_market_margin_sh)
    return df

def fetch_cffex_data(date_fmt: str, product: str) -> Optional[dict]:
    """获取CFFEX持仓数据"""
    import xml.etree.ElementTree as ET

    url = f"http://www.cffex.com.cn/sj/ccpm/{date_fmt}/{product}.xml"

    import requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Referer': 'http://www.cffex.com.cn/ccpm/'
    }

    response = fetch_with_retry(requests.get, url, headers=headers, timeout=30)

    if 'instrumentid' not in response.text:
        return None

    root = ET.fromstring(response.text)
    total_long, total_short = 0, 0

    for data in root.findall('.//data'):
        shortname = data.find('shortname').text.strip() if data.find('shortname') is not None else ''
        datatypeid = data.find('datatypeid').text.strip() if data.find('datatypeid') is not None else ''
        volume = int(data.find('volume').text.strip() if data.find('volume') is not None else '0')

        if '中信期货' in shortname and datatypeid in ['1', '2']:
            if datatypeid == '1':
                total_long += volume
            else:
                total_short += volume

    return {'多单': total_long, '空单': total_short, 'net': total_short - total_long}

def update_all_data():
    """更新所有数据"""
    print("="*60)
    print(f"数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Load stock pool
    pool_path = 'data/expanded_stock_pool.json'
    with open(pool_path, 'r') as f:
        pool = json.load(f)
    codes = pool.get('stocks', [])[:50]  # Top 50 for efficiency

    success_count = 0
    fail_count = 0

    print(f"\n更新股票数据 ({len(codes)} 只)...")
    for code in codes:
        try:
            # Convert code format
            if code.startswith('sh.'):
                symbol = code.replace('sh.', '')
            else:
                symbol = code.replace('sz.', '')

            # Fetch data (300 days for enough lookback)
            df = fetch_stock_hist(symbol, days=300)

            if df is not None and len(df) > 0:
                # Save to parquet
                cache_file = f"data/cache/kline_{code.replace('.', '_')}.parquet"
                df['code'] = code
                df.to_parquet(cache_file)
                success_count += 1
                if success_count % 10 == 0:
                    print(f"  进度: {success_count}/{len(codes)}")
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(f"  ✗ {code}: {str(e)[:40]}")

    print(f"\n股票数据更新完成: 成功 {success_count}, 失败 {fail_count}")

    # Update margin data
    try:
        margin_df = fetch_margin_data()
        if margin_df is not None:
            margin_df.to_json('data/cache/margin_balance_history.json', orient='records')
            print(f"融资融券数据已更新: {len(margin_df)} 条")
    except Exception as e:
        print(f"融资融券数据更新失败: {e}")

    # Update CFFEX data for recent dates
    print("\n更新CFFEX持仓数据...")
    today = datetime.now()
    for i in range(10):  # Last 10 trading days
        date = today - timedelta(days=i)
        if date.weekday() >= 5:  # Skip weekend
            continue

        date_fmt = date.strftime('%Y%m/') + date.strftime('%d')

        for product in ['IF', 'IH', 'IC', 'IM']:
            try:
                result = fetch_cffex_data(date_fmt, product)
                if result:
                    # Save to cache
                    cache_file = f"data/cache/cffex_{date.strftime('%Y%m%d')}_{product}.json"
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'date': date.strftime('%Y-%m-%d'),
                            'product': product,
                            **result
                        }, f)
            except Exception as e:
                pass  # Silently continue

    print("\n" + "="*60)
    print("数据更新完成!")
    print("="*60)

if __name__ == '__main__':
    update_all_data()
