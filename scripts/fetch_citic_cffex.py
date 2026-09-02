#!/usr/bin/env python3
"""
CFFEX会员持仓数据获取工具
========================
从中国金融期货交易所官网获取会员持仓排名数据

Usage:
    python fetch_citic_cffex.py 2026-08-31
    python fetch_citic_cffex.py 2026-08-29 IF  # 只查沪深300
"""
import sys
import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

CFFEX_URL = "http://www.cffex.com.cn/sj/ccpm/{}/{}.xml"
PRODUCT_NAMES = {
    'IF': '沪深300',
    'IH': '上证50',
    'IC': '中证500',
    'IM': '中证1000',
    'TS': '2年期国债',
    'TF': '5年期国债',
    'T': '10年期国债',
    'TL': '30年期国债'
}

def get_positions(date_str, products=None):
    """
    获取指定日期的持仓数据

    Args:
        date_str: 形如 '202608/31' 或 '2026-08-31'
        products: 产品代码列表，如 ['IF', 'IH']，None表示全部

    Returns:
        dict: {product: {contract: {多单, 空单, var多单, var空单}}}
    """
    # 转换日期格式
    if '-' in date_str:
        parts = date_str.split('-')
        date_fmt = parts[0] + parts[1] + '/' + parts[2]
    else:
        date_fmt = date_str

    if products is None:
        products = ['IF', 'IH', 'IC', 'IM']

    results = {}

    for product in products:
        url = CFFEX_URL.format(date_fmt, product)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'http://www.cffex.com.cn/ccpm/'
        }

        try:
            resp = requests.get(url, timeout=15)
            if 'instrumentid' not in resp.text:
                results[product] = None
                continue

            root = ET.fromstring(resp.text)
            contracts = defaultdict(lambda: {'多单': 0, '空单': 0, 'var多单': 0, 'var空单': 0})

            for data in root.findall('.//data'):
                instrumentid = data.find('instrumentid').text.strip() if data.find('instrumentid') is not None else ''
                shortname = data.find('shortname').text.strip() if data.find('shortname') is not None else ''
                datatypeid = data.find('datatypeid').text.strip() if data.find('datatypeid') is not None else ''
                volume = int(data.find('volume').text.strip() if data.find('volume') is not None else '0')
                varvolume = int(data.find('varvolume').text.strip() if data.find('varvolume') is not None else '0')

                if '中信期货' in shortname and datatypeid in ['1', '2']:
                    dtype = {1: '多单', 2: '空单'}.get(int(datatypeid), None)
                    if dtype:
                        contracts[instrumentid][dtype] += volume
                        contracts[instrumentid]['var' + dtype] += varvolume

            results[product] = dict(contracts)

        except Exception as e:
            results[product] = {'error': str(e)}

    return results

def get_citic_net_positions(date_str, products=None):
    """获取中信期货净持仓汇总"""
    positions = get_positions(date_str, products)

    summary = {}
    for product, contracts in positions.items():
        if contracts is None:
            summary[product] = {'多单': 0, '空单': 0, 'var多单': 0, 'var空单': 0, 'error': '无数据'}
            continue
        if 'error' in contracts:
            summary[product] = {'多单': 0, '空单': 0, 'var多单': 0, 'var空单': 0, 'error': contracts.get('error')}
            continue

        total_long, total_short = 0, 0
        total_var_long, total_var_short = 0, 0
        for contract, data in contracts.items():
            total_long += data['多单']
            total_short += data['空单']
            total_var_long += data['var多单']
            total_var_short += data['var空单']

        summary[product] = {
            '多单': total_long,
            '空单': total_short,
            'var多单': total_var_long,
            'var空单': total_var_short,
            'net': total_short - total_long
        }

    return summary

def print_report(date_input, products=None):
    """打印持仓报告"""
    # 转换日期格式
    if '-' in date_input:
        parts = date_input.split('-')
        date_fmt = parts[0] + parts[1] + '/' + parts[2]
    else:
        date_fmt = date_input
        date_input = date_fmt[:4] + '-' + date_fmt[4:6] + '-' + date_fmt[7:]

    summary = get_citic_net_positions(date_fmt, products)

    print("=" * 75)
    print(f"中信期货净持仓 {date_input}")
    print("=" * 75)

    total_net = 0
    for product in ['IF', 'IH', 'IC', 'IM']:
        if products and product not in products:
            continue
        data = summary.get(product, {})
        name = PRODUCT_NAMES.get(product, product)

        if 'error' in data:
            print(f"\n{product} {name}: {data['error']}")
            continue

        net = data.get('net', 0)
        direction = "净空" if net > 0 else "净多"
        print(f"\n{product} {name}:")
        print(f"  多单: {data['多单']:,} 手 (较前日 {data['var多单']:+,} 手)")
        print(f"  空单: {data['空单']:,} 手 (较前日 {data['var空单']:+,} 手)")
        print(f"  净持仓: {abs(net):,} 手 {direction}")
        total_net += net

    print("\n" + "-" * 75)
    print(f"合计净持仓: {abs(total_net):,} 手 ({'净空' if total_net > 0 else '净多'})")
    print("=" * 75)

if __name__ == '__main__':
    # 解析命令行参数
    if len(sys.argv) < 2:
        # 默认获取最近交易日
        today = datetime.now()
        date_input = today.strftime('%Y-%m-%d')
    else:
        date_input = sys.argv[1]

    products = None
    if len(sys.argv) >= 3:
        products = [sys.argv[2]]

    print_report(date_input, products)