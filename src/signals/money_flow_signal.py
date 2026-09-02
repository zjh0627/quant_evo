#!/usr/bin/env python3
"""
资金流因子信号模块
================
基于东方财富资金流数据生成交易信号

资金流是重要的短期 alpha 来源:
- 主力净流入 = 股价上涨的驱动因素
- 连续资金流入 = 机构建仓信号
- 资金流背离 = 反转信号
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List

CACHE_DIR = '/Users/keira/project/claude/quant_evo/data/cache'
MONEY_FLOW_RANK_FILE = f'{CACHE_DIR}/money_flow_rank.json'

def get_money_flow_rank(indicator: str = "5日") -> pd.DataFrame:
    """
    获取资金流排名数据

    Args:
        indicator: "今日", "3日", "5日", "10日"

    Returns:
        DataFrame with stock fund flow data
    """
    try:
        import akshare as ak

        df = ak.stock_individual_fund_flow_rank(indicator=indicator)
        return df

    except Exception as e:
        print(f"[MoneyFlow] 获取数据失败: {e}")
        return pd.DataFrame()

def calculate_money_flow_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算整体市场资金流评分

    Args:
        df: 资金流排名数据

    Returns:
        {
            'score': 0-100,
            'main_inflow': bool,  # 主力是否净流入
            'inflow_amount': float,
            'top_inflow_stocks': List[str],
            'top_outflow_stocks': List[str]
        }
    """
    if df.empty:
        return {
            'score': 50,
            'main_inflow': False,
            'inflow_amount': 0,
            'top_inflow_stocks': [],
            'top_outflow_stocks': []
        }

    # 东方财富资金流列名
    # 主力净流入 = 超大单净流入 + 大单净流入
    possible_cols = [
        '主力净流入-净额',
        '主力净流入',
        '超大单净流入',
        '大单净流入',
        '净流入',
        '主力净流入净额'
    ]

    inflow_col = None
    for col in possible_cols:
        if col in df.columns:
            inflow_col = col
            break

    if inflow_col is None:
        print(f"[MoneyFlow] 未知列名: {df.columns.tolist()}")
        # 使用第一列数值列
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                inflow_col = col
                break

    if inflow_col is None:
        return {
            'score': 50,
            'main_inflow': False,
            'inflow_amount': 0,
            'top_inflow_stocks': [],
            'top_outflow_stocks': []
        }

    # 计算整体资金流
    total_inflow = df[inflow_col].sum()
    inflow_count = (df[inflow_col] > 0).sum()
    outflow_count = (df[inflow_col] < 0).sum()

    # 计算得分
    # 资金净流入且流入股票多于流出股票 = 看多
    if total_inflow > 0 and inflow_count > outflow_count:
        score = 50 + min(40, (inflow_count / len(df)) * 50)
        main_inflow = True
    elif total_inflow < 0 and outflow_count > inflow_count:
        score = 50 - min(40, (outflow_count / len(df)) * 50)
        main_inflow = False
    else:
        score = 50
        main_inflow = total_inflow > 0

    # 获取流入/流出最多的股票
    df_sorted = df.sort_values(inflow_col, ascending=False)
    top_inflow = df_sorted.head(5)['股票代码'].tolist() if '股票代码' in df.columns else []
    top_outflow = df_sorted.tail(5)['股票代码'].tolist() if '股票代码' in df.columns else []

    return {
        'score': min(100, max(0, score)),
        'main_inflow': main_inflow,
        'inflow_amount': float(total_inflow),
        'inflow_count': int(inflow_count),
        'outflow_count': int(outflow_count),
        'top_inflow_stocks': top_inflow,
        'top_outflow_stocks': top_outflow,
        'indicator': indicator
    }

def generate_money_flow_signal() -> Dict:
    """
    生成资金流信号

    Returns:
        {
            'signal': 'BULLISH'/'BEARISH'/'NEUTRAL',
            'confidence': 0-100,
            'description': str,
            'score': float,
            'inflow_amount': float,
            'inflow_count': int,
            'outflow_count': int
        }
    """
    # 获取5日资金流排名
    df = get_money_flow_rank(indicator="5日")

    if df.empty:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'description': '数据获取失败',
            'score': 50,
            'inflow_amount': 0,
            'inflow_count': 0,
            'outflow_count': 0
        }

    result = calculate_money_flow_score(df)

    # 信号逻辑
    signal = 'NEUTRAL'
    confidence = 50
    description = ''

    inflow_pct = result['inflow_count'] / (result['inflow_count'] + result['outflow_count'] + 1)
    inflow_amount_b = result['inflow_amount'] / 1e9  # 转为亿元
    inflow_cnt = result['inflow_count']
    outflow_cnt = result['outflow_count']

    if result['main_inflow'] and inflow_pct > 0.6:
        signal = 'BULLISH'
        confidence = 65
        description = f'主力净流入{inflow_amount_b:.1f}亿, {inflow_cnt}只股票资金流入'
    elif not result['main_inflow'] and inflow_pct < 0.4:
        signal = 'BEARISH'
        confidence = 65
        description = f'主力净流出{abs(inflow_amount_b):.1f}亿, {outflow_cnt}只股票资金流出'
    elif result['main_inflow']:
        signal = 'BULLISH'
        confidence = 55
        description = f'主力小幅净流入{inflow_amount_b:.1f}亿'
    else:
        signal = 'BEARISH'
        confidence = 55
        description = f'主力小幅净流出{abs(inflow_amount_b):.1f}亿'

    return {
        'signal': signal,
        'confidence': confidence,
        'description': description,
        'score': result['score'],
        'inflow_amount': result['inflow_amount'],
        'inflow_count': result['inflow_count'],
        'outflow_count': result['outflow_count'],
        'top_inflow_stocks': result.get('top_inflow_stocks', []),
        'top_outflow_stocks': result.get('top_outflow_stocks', [])
    }

def print_money_flow_status():
    """打印资金流状态"""
    signal_data = generate_money_flow_signal()

    print("\n" + "=" * 60)
    print("主力资金流信号 (东方财富)")
    print("=" * 60)

    print(f"\n信号: {signal_data['signal']}")
    print(f"置信度: {signal_data['confidence']}%")
    print(f"描述: {signal_data['description']}")
    print(f"得分: {signal_data['score']:.1f}")
    print(f"净流入额: {signal_data['inflow_amount']/1e9:+.2f}亿元")
    print(f"流入股票: {signal_data['inflow_count']}只")
    print(f"流出股票: {signal_data['outflow_count']}只")

    if signal_data['top_inflow_stocks']:
        print(f"\n资金流入前五: {signal_data['top_inflow_stocks']}")
    if signal_data['top_outflow_stocks']:
        print(f"资金流出前五: {signal_data['top_outflow_stocks']}")

    print("=" * 60)


if __name__ == '__main__':
    print_money_flow_status()
