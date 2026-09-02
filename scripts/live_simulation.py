#!/usr/bin/env python3
"""
实时模拟操作 v2
===============
使用本地CSV历史数据 + 新浪财经实时价格
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import requests

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
LIVE_SIGNALS_FILE = f'{CACHE_DIR}/live_signals.json'

def load_stock_names():
    """加载股票名称"""
    pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('names', {})
    return {}

def get_sina_realtime_prices(codes: list) -> dict:
    """从新浪财经获取实时价格"""
    prices = {}
    try:
        code_str = ','.join([c.replace('.', '') for c in codes])
        url = f'http://hq.sinajs.cn/list={code_str}'
        headers = {'Referer': 'http://finance.sina.com.cn'}
        r = requests.get(url, timeout=5, headers=headers)
        r.encoding = 'gbk'
        text = r.text

        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            if '="' in line and i < len(codes):
                code = codes[i]
                data = line.split('="')[1].strip('";').split(',')
                if len(data) > 3:
                    try:
                        prev_close = float(data[2])
                        current = float(data[3])
                        prices[code] = {
                            'close': current,
                            'prev_close': prev_close,
                            'change': (current - prev_close) / prev_close * 100 if prev_close > 0 else 0
                        }
                    except:
                        pass
    except Exception as e:
        print(f"获取实时价格失败: {e}")
    return prices

def load_local_data(code: str) -> pd.DataFrame:
    """从本地CSV加载历史数据"""
    code_fmt = code.replace('.', '_')
    filepath = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df[df['volume'] > 0]
        return df
    return None

def calculate_signals(df: pd.DataFrame, current_price: float = None) -> dict:
    """计算单只股票信号"""
    if df is None or len(df) < 20:
        return None

    close = df['close'].values

    # 如果有实时价格，用实时价格替换最后收盘价
    if current_price and current_price > 0:
        close = close.copy()
        close[-1] = current_price

    volume = df['volume'].values

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss != 0 else 50
    rsi = 100 - (100 / (1 + rs))
    if pd.isna(rsi): rsi = 50

    # 动量
    mom_5d = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
    mom_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

    # 均线
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    ma60 = pd.Series(close).rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
    uptrend = ma20 > ma60

    # 综合评分
    score = 50
    if rsi < 30:
        score += 25
    elif rsi > 70:
        score -= 25
    elif rsi < 40:
        score += 10

    if uptrend:
        score += 15

    if mom_20d > 0.1:
        score += 10
    elif mom_20d < -0.1:
        score -= 10

    # 信号 - 调整阈值产生更多交易
    if score > 60 and rsi < 55:
        signal = 'BUY'
    elif score < 45 or rsi > 68:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'rsi': float(rsi),
        'momentum_5d': float(mom_5d * 100),
        'momentum_20d': float(mom_20d * 100),
        'uptrend': bool(uptrend),
        'score': float(score),
        'signal': signal,
        'close': float(close[-1])
    }

def scan_live_signals(codes: list):
    """扫描实时信号"""
    names = load_stock_names()

    print(f"扫描 {len(codes)} 只股票的实时信号...")

    # 获取实时价格
    prices = get_sina_realtime_prices(codes[:50])
    print(f"获取实时价格: {len(prices)} 只")

    signals = []

    for code in codes:
        df = load_local_data(code)
        if df is None or len(df) < 60:
            continue

        current_price = prices.get(code, {}).get('close')

        result = calculate_signals(df, current_price)
        if result is None:
            continue

        result['code'] = code
        result['name'] = names.get(code, code)
        result['date'] = df['date'].iloc[-1]
        result['prev_close'] = prices.get(code, {}).get('prev_close', result['close'])

        signals.append(result)

        if result['signal'] != 'HOLD':
            print(f"  {result['name']}: {result['signal']} (评分:{result['score']:.0f}, RSI:{result['rsi']:.1f})")

    # 排序
    signals.sort(key=lambda x: (0 if x['signal'] == 'BUY' else 1 if x['signal'] == 'SELL' else 2, -x['score']))

    return signals

def main():
    print("=" * 60)
    print("实时信号扫描 v2 (新浪财经)")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载股票池
    pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:50]
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

    # 扫描信号
    signals = scan_live_signals(codes)

    # 保存
    with open(LIVE_SIGNALS_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'signals': signals
        }, f, ensure_ascii=False, indent=2)

    # 统计
    buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals if s['signal'] == 'SELL')

    print(f"\n信号统计: 买入={buy_count}, 卖出={sell_count}, 持有={len(signals)-buy_count-sell_count}")
    print(f"结果已保存到 {LIVE_SIGNALS_FILE}")

if __name__ == '__main__':
    main()
