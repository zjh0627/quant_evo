#!/usr/bin/env python3
"""
情绪因子信号模块
================
基于融资融券余额和市场宽度生成交易信号

融资融券余额是市场情绪的重要反向指标:
- 融资余额大幅增加 → 市场过热，可能回调
- 融资余额大幅减少 → 市场恐慌，可能反弹
- 融资买入额占成交额比例 → 市场参与度热情
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

CACHE_DIR = '/Users/keira/project/claude/quant_evo/data/cache'
MARGIN_CACHE_FILE = f'{CACHE_DIR}/margin_balance_history.json'

def get_margin_data(days: int = 20) -> pd.DataFrame:
    """
    获取最近N天的融资融券余额数据

    Args:
        days: 获取天数

    Returns:
        DataFrame with columns: date, margin_balance, margin_buy_amount, short_balance, total_balance
    """
    cache_file = MARGIN_CACHE_FILE

    # 尝试从缓存加载
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    else:
        # 从AKShare获取
        df = fetch_margin_data()

    # 按日期排序
    df = df.sort_values('date').tail(days).reset_index(drop=True)
    return df

def fetch_margin_data() -> pd.DataFrame:
    """从AKShare获取融资融券数据并缓存"""
    try:
        import akshare as ak

        # 获取沪市和深市融资融券余额
        margin_sh = ak.macro_china_market_margin_sh()
        margin_sz = ak.macro_china_market_margin_sz()

        # 合并
        margin_sh = margin_sh[['日期', '融资余额', '融资买入额', '融券余额', '融资融券余额']]
        margin_sh.columns = ['date', 'margin_balance_sh', 'margin_buy_sh', 'short_balance_sh', 'total_balance_sh']

        margin_sz = margin_sz[['日期', '融资余额', '融资买入额', '融券余额', '融资融券余额']]
        margin_sz.columns = ['date', 'margin_balance_sz', 'margin_buy_sz', 'short_balance_sz', 'total_balance_sz']

        df = pd.merge(margin_sh, margin_sz, on='date', how='outer')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 计算总额
        df['margin_balance'] = df['margin_balance_sh'].fillna(0) + df['margin_balance_sz'].fillna(0)
        df['short_balance'] = df['short_balance_sh'].fillna(0) + df['short_balance_sz'].fillna(0)
        df['total_balance'] = df['total_balance_sh'].fillna(0) + df['total_balance_sz'].fillna(0)
        df['margin_buy'] = df['margin_buy_sh'].fillna(0) + df['margin_buy_sz'].fillna(0)

        # 只保留需要的列
        df = df[['date', 'margin_balance', 'short_balance', 'total_balance', 'margin_buy']]

        # 缓存
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(MARGIN_CACHE_FILE, 'w') as f:
            json.dump(df.to_dict('records'), f, ensure_ascii=False)

        print(f"[Margin] 融资融券数据已更新: {len(df)} 条记录")
        return df

    except Exception as e:
        print(f"[Margin] 获取数据失败: {e}")
        return pd.DataFrame()

def calculate_margin_change(df: pd.DataFrame) -> pd.DataFrame:
    """计算融资融券余额变化"""
    df = df.copy()

    # 日变化
    df['margin_change'] = df['margin_balance'].diff()
    df['margin_change_pct'] = df['margin_balance'].pct_change()

    # 5日变化
    df['margin_change_5d'] = df['margin_balance'].diff(5)
    df['margin_change_5d_pct'] = df['margin_balance'].pct_change(5)

    # 20日变化
    df['margin_change_20d'] = df['margin_balance'].diff(20)
    df['margin_change_20d_pct'] = df['margin_balance'].pct_change(20)

    # 融资余额历史分位数
    df['margin_percentile'] = df['margin_balance'].rank(pct=True)

    # 计算融资买入额占余额比例
    df['margin_buy_ratio'] = df['margin_buy'] / (df['margin_balance'] + 1)

    return df

def generate_margin_signal() -> Dict:
    """
    生成融资融券情绪信号

    Returns:
        {
            'signal': 'BULLISH'/'BEARISH'/'NEUTRAL',
            'confidence': 0-100,
            'description': str,
            'margin_balance': float,
            'margin_change_5d_pct': float,
            'margin_percentile': float
        }
    """
    df = get_margin_data(30)

    if len(df) < 5:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'description': '数据不足',
            'margin_balance': 0,
            'margin_change_5d_pct': 0,
            'margin_percentile': 0
        }

    df = calculate_margin_change(df)
    latest = df.iloc[-1]

    signal = 'NEUTRAL'
    confidence = 50
    description = ''

    margin_balance = latest['margin_balance'] / 1e12  # 转为万亿
    change_5d_pct = latest['margin_change_5d_pct'] * 100 if pd.notna(latest['margin_change_5d_pct']) else 0
    change_20d_pct = latest['margin_change_20d_pct'] * 100 if pd.notna(latest['margin_change_20d_pct']) else 0
    percentile = latest['margin_percentile'] if pd.notna(latest['margin_percentile']) else 0.5

    # 信号逻辑
    if change_5d_pct > 3:  # 5日内融资余额增加超过3%
        signal = 'BEARISH'
        confidence = 70
        description = f'融资余额5日增{change_5d_pct:.1f}%，市场过热警告'
    elif change_5d_pct < -3:  # 5日内融资余额减少超过3%
        signal = 'BULLISH'
        confidence = 70
        description = f'融资余额5日降{abs(change_5d_pct):.1f}%，恐慌抄底信号'
    elif change_20d_pct > 8:  # 20日融资余额增加超过8%
        signal = 'BEARISH'
        confidence = 65
        description = f'融资余额20日增{change_20d_pct:.1f}%，杠杆资金堆积'
    elif change_20d_pct < -8:  # 20日融资余额减少超过8%
        signal = 'BULLISH'
        confidence = 65
        description = f'融资余额20日降{abs(change_20d_pct):.1f}%，去杠杆尾声'
    elif percentile > 0.9:  # 融资余额处于历史高位
        signal = 'BEARISH'
        confidence = 55
        description = f'融资余额处历史高位({percentile*100:.0f}%)'
    elif percentile < 0.1:  # 融资余额处于历史低位
        signal = 'BULLISH'
        confidence = 55
        description = f'融资余额处历史低位({percentile*100:.0f}%)'
    else:
        description = f'融资余额稳定({margin_balance:.2f}万亿)'

    return {
        'signal': signal,
        'confidence': confidence,
        'description': description,
        'margin_balance': margin_balance,
        'margin_change_5d_pct': change_5d_pct,
        'margin_change_20d_pct': change_20d_pct,
        'margin_percentile': percentile,
        'dates': df['date'].tolist()[-10:],
        'balances': df['margin_balance'].tolist()[-10:]
    }

def print_margin_status():
    """打印融资融券状态"""
    signal_data = generate_margin_signal()

    print("\n" + "=" * 60)
    print("融资融券情绪信号")
    print("=" * 60)

    if signal_data['dates']:
        print(f"\n最近10个交易日融资余额:")
        for i, date in enumerate(signal_data['dates'][-10:]):
            balance = signal_data['balances'][i] / 1e12
            print(f"  {date}: {balance:.3f}万亿")

    print(f"\n信号: {signal_data['signal']}")
    print(f"置信度: {signal_data['confidence']}%")
    print(f"描述: {signal_data['description']}")
    print(f"融资余额: {signal_data['margin_balance']:.3f}万亿")
    print(f"5日变化: {signal_data['margin_change_5d_pct']:+.2f}%")
    print(f"20日变化: {signal_data['margin_change_20d_pct']:+.2f}%")
    print(f"历史分位: {signal_data['margin_percentile']*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    print_margin_status()
