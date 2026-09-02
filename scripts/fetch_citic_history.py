#!/usr/bin/env python3
"""
获取中信期货历史持仓数据并与市场走势对比
"""
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import pandas as pd

CFFEX_URL = "http://www.cffex.com.cn/sj/ccpm/{}/{}.xml"

def get_citic_positions(date_fmt, product):
    """获取指定日期和品种的中信持仓汇总"""
    url = CFFEX_URL.format(date_fmt, product)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Referer': 'http://www.cffex.com.cn/ccpm/'
    }

    try:
        resp = requests.get(url, timeout=20)
        if 'instrumentid' not in resp.text:
            return None

        root = ET.fromstring(resp.text)
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
    except:
        return None

def get_all_mondays(start_date, end_date):
    """获取所有周一日期"""
    mondays = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # 找到第一个周一
    while current.weekday() != 0:
        current += timedelta(days=1)

    while current <= end:
        mondays.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=7)

    return mondays

if __name__ == '__main__':
    # 获取过去2个月的周一数据
    end_date = '2026-08-31'
    start_date = '2026-07-01'

    mondays = get_all_mondays(start_date, end_date)
    print(f"周一列表 ({len(mondays)} 个): {mondays}")

    products = ['IF', 'IH', 'IC', 'IM']
    product_names = {'IF': '沪深300', 'IH': '上证50', 'IC': '中证500', 'IM': '中证1000'}

    # 获取持仓数据
    holdings_data = []

    for date in mondays:
        date_fmt = date.replace('-', '')[:6] + '/' + date.split('-')[2]
        print(f"\n获取 {date} 数据...")

        for product in products:
            result = get_citic_positions(date_fmt, product)
            if result:
                holdings_data.append({
                    'date': date,
                    'product': product,
                    'product_name': product_names[product],
                    '多单': result['多单'],
                    '空单': result['空单'],
                    'net': result['net']
                })
                print(f"  {product}: 多={result['多单']:,} 空={result['空单']:,} 净={result['net']:,}")

        # 避免请求过快
        import time
        time.sleep(0.5)

    # 保存持仓数据
    holdings_file = '/Users/keira/project/claude/quant_evo/data/cache/citic_holdings_weekly.json'
    with open(holdings_file, 'w') as f:
        json.dump(holdings_data, f, ensure_ascii=False, indent=2)
    print(f"\n持仓数据已保存: {holdings_file}")

    # 获取市场指数数据
    print("\n获取市场数据...")

    # 从现有数据获取指数数据
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')
    from src.simulation.evolving_portfolio import load_stock_data

    # 加载沪深300指数数据（用IF期货代替）
    # 这里简化处理，使用IF期货的收盘价作为市场代理
    # 实际应该用沪深300指数

    # 生成市场数据（简化版 - 使用持仓数据对应日期的市场表现）
    # 实际上我们需要获取实际的指数涨跌

    print("\n" + "="*70)
    print("中信期货净持仓与市场走势对比分析")
    print("="*70)

    # 创建DataFrame
    df = pd.DataFrame(holdings_data)

    # 按产品和日期排序
    df = df.sort_values(['product', 'date'])

    # 计算净持仓变化
    for product in products:
        product_df = df[df['product'] == product].copy()
        if len(product_df) >= 2:
            product_df['net_change'] = product_df['net'].diff()

            print(f"\n{product_names[product]}:")
            print("-" * 60)
            for _, row in product_df.iterrows():
                date = row['date']
                net = row['net']
                net_change = row.get('net_change', 0)
                direction = "净空" if net > 0 else "净多"
                change_str = f"{net_change:+,.0f}" if pd.notna(net_change) else "N/A"
                print(f"{date}: 净持仓 {abs(net):>8,} 手 {direction} (日变化 {change_str})")
