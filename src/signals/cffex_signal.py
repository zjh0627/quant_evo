#!/usr/bin/env python3
"""
CFFEX机构持仓信号 (简化版 - 使用缓存数据)
"""
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import os
from typing import Dict, Optional

CACHE_FILE = '/Users/keira/project/claude/quant_evo/data/cache/citic_positions_history.json'

def load_cached_data() -> Dict:
    """加载缓存的中信持仓数据"""
    if not os.path.exists(CACHE_FILE):
        return {}

    with open(CACHE_FILE, 'r') as f:
        data = json.load(f)

    # 按日期汇总所有品种
    daily = {}
    for key, item in data.items():
        date = item['date']
        if date not in daily:
            daily[date] = 0
        daily[date] += item['net']  # 净空 = 空单 - 多单

    return daily

def get_recent_data(days: int = 5) -> list:
    """获取最近N天的数据"""
    daily = load_cached_data()
    dates = sorted(daily.keys())[-days:]

    result = []
    for d in dates:
        result.append({'date': d, 'net': daily[d]})

    # 计算日变化
    for i in range(len(result)):
        if i == 0:
            result[i]['change'] = 0
        else:
            result[i]['change'] = result[i-1]['net'] - result[i]['net']

    return result

def generate_signal() -> Dict:
    """生成交易信号"""
    data = get_recent_data(5)

    if len(data) < 2:
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'description': '数据不足',
            'net_short': 0,
            'change': 0
        }

    current = data[-1]
    changes = [d['change'] for d in data[1:]]

    # 信号逻辑
    signal = 'NEUTRAL'
    confidence = 50
    description = ''

    current_net = current['net']
    current_change = current['change']

    # 1. 净空单创近期新高
    max_net = max(d['net'] for d in data)
    if current_net >= max_net and current_change > 0:
        signal = 'BULLISH'
        confidence = 70
        description = f'净空单创阶段新高({current_net/10000:.1f}万手)，机构大规模建空'

    # 2. 净空单大幅增加(>3000手)
    elif current_change > 3000:
        signal = 'BULLISH'
        confidence = 65
        description = f'净空单大增{current_change/10000:.1f}万手，短期底部信号'

    # 3. 连续3天净空增加
    elif len([c for c in changes[-3:] if c > 0]) >= 3:
        signal = 'BEARISH'
        confidence = 60
        description = '净空单连续增加，机构看空情绪累积'

    # 4. 净空单大幅下降
    elif current_change < -3000:
        signal = 'BULLISH'
        confidence = 55
        description = f'净空单减少{abs(current_change)/10000:.1f}万手，机构减持空头'

    # 5. 连续净空下降
    elif len([c for c in changes[-3:] if c < 0]) >= 3:
        signal = 'BEARISH'
        confidence = 55
        description = '净空单持续下降，机构看多'

    else:
        description = f'净空单稳定({current_net/10000:.1f}万手)'

    return {
        'signal': signal,
        'confidence': confidence,
        'description': description,
        'net_short': current_net,
        'change': current_change,
        'dates': [d['date'] for d in data],
        'values': [d['net'] for d in data],
        'changes': [d['change'] for d in data]
    }

def print_signal():
    """打印信号状态"""
    signal_data = generate_signal()

    print("\n" + "=" * 60)
    print("机构持仓信号 (CFFEX中信期货)")
    print("=" * 60)

    if signal_data['dates']:
        print(f"\n最近{len(signal_data['dates'])}个交易日净空单:")
        for i, date in enumerate(signal_data['dates']):
            val = signal_data['values'][i]
            chg = signal_data['changes'][i]
            chg_str = f"{chg:+,.0f}" if chg != 0 else "-"
            print(f"  {date}: {val:>10,} ({chg_str})")

    print(f"\n信号: {signal_data['signal']}")
    print(f"置信度: {signal_data['confidence']}%")
    print(f"描述: {signal_data['description']}")
    print("=" * 60)

if __name__ == '__main__':
    print_signal()