#!/usr/bin/env python3
"""
机构持仓信号模块
================
基于中国金融期货交易所(CFFEX)会员持仓数据生成交易信号

中信期货净空单作为市场反向指标：
- 净空单大幅增加 → 次日市场可能反弹
- 净空单创阶段性新高 → 市场可能接近底部
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

CFFEX_URL = "http://www.cffex.com.cn/sj/ccpm/{}/{}.xml"
CACHE_DIR = '/Users/keira/project/claude/quant_evo/data/cache'

class CFFEXSignal:
    """CFFEX机构持仓信号生成器"""

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or f'{CACHE_DIR}/cffex_daily_cache.json'
        self.data = {}
        self.load_cache()

    def load_cache(self):
        """加载缓存数据"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.data = json.load(f)
            except:
                self.data = {}

    def save_cache(self):
        """保存缓存数据"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False)

    def get_citic_positions(self, date_fmt: str, product: str) -> Optional[Dict]:
        """
        获取指定日期和品种的中信期货净持仓

        Args:
            date_fmt: 日期格式 YYYYMMDD/YY 如 '202608/31'
            product: 品种代码如 'IF', 'IH', 'IC', 'IM'

        Returns:
            {'多单': int, '空单': int, 'net': int} 或 None
        """
        url = CFFEX_URL.format(date_fmt, product)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'http://www.cffex.com.cn/ccpm/'
        }

        try:
            resp = requests.get(url, timeout=30)
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
        except Exception as e:
            print(f"[CFFEX] 获取{product}数据失败: {e}")
            return None

    def get_total_net_short(self, date_fmt: str) -> Optional[int]:
        """
        获取某日所有品种的中信期货净空单总量

        Args:
            date_fmt: 日期格式

        Returns:
            净空单总量(int)或None
        """
        products = ['IF', 'IH', 'IC', 'IM']
        total_net = 0

        for product in products:
            result = self.get_citic_positions(date_fmt, product)
            if result:
                total_net += result['net']

        result_total = total_net if total_net > 0 else None

        # 缓存数据
        if result_total is not None:
            self.data[date_fmt] = {'net': result_total}
            self.save_cache()

        return result_total

    def get_recent_net_short(self, days: int = 5) -> Dict:
        """
        获取最近N天的净空单数据（优先使用缓存）

        Args:
            days: 天数

        Returns:
            {'dates': [...], 'values': [...], 'changes': [...]}
        """
        # 先尝试从缓存读取
        today = datetime.now()
        results = {'dates': [], 'values': [], 'changes': []}

        for i in range(days * 2):  # 扩大搜索范围以确保找到足够的工作日
            date = today - timedelta(days=i)
            if date.weekday() >= 5:  # 跳过周末
                continue

            date_display = date.strftime('%Y-%m-%d')
            date_key = date.strftime('%Y-%m-%d')

            # 尝试从缓存获取
            net = self._get_cached_net_short(date_key)
            if net is not None:
                results['dates'].insert(0, date_display)
                results['values'].insert(0, net)

            if len(results['dates']) >= days:
                break

        # 如果缓存数据不足，尝试获取今天的数据
        if len(results['dates']) < days:
            try:
                today_fmt = today.strftime('%Y%m/') + today.strftime('%d')
                net = self.get_total_net_short(today_fmt)
                if net is not None:
                    today_str = today.strftime('%Y-%m-%d')
                    if today_str not in results['dates']:
                        results['dates'].append(today_str)
                        results['values'].append(net)
            except:
                pass

        # 计算变化
        for i in range(1, len(results['values'])):
            results['changes'].insert(0, results['values'][i-1] - results['values'][i])
        results['changes'].insert(0, 0)  # 第一个无变化

        return results

    def _get_cached_net_short(self, date: str) -> Optional[int]:
        """从缓存获取某日净空单"""
        if date in self.data:
            return self.data[date].get('net')
        return None

    def generate_signal(self) -> Tuple[str, float, str]:
        """
        生成机构持仓信号

        Returns:
            (signal, confidence, description)
            signal: 'BULLISH', 'BEARISH', 'NEUTRAL'
            confidence: 0-100
            description: 信号描述
        """
        recent = self.get_recent_net_short(days=5)

        if len(recent['values']) < 2:
            return 'NEUTRAL', 0, '数据不足'

        current_net = recent['values'][-1]
        prev_net = recent['values'][-2]
        change = current_net - prev_net

        # 计算历史波动范围
        avg_net = sum(recent['values']) / len(recent['values'])
        max_net = max(recent['values'])
        min_net = min(recent['values'])

        # 信号逻辑
        if len(recent['values']) >= 3:
            recent_changes = recent['changes'][1:]  # 排除第一个
            consecutive_increase = sum(1 for c in recent_changes if c > 0)
        else:
            consecutive_increase = 0

        signal = 'NEUTRAL'
        confidence = 50
        description = ''

        # 情况1: 净空单创近期新高 -> 可能见底信号
        if current_net >= max_net and change > 0:
            signal = 'BULLISH'
            confidence = 70
            description = f'净空单创{min_net/10000:.1f}日内新高({current_net/10000:.1f}万手)，机构或已大规模建仓'

        # 情况2: 净空单大幅增加(>5000手) -> 反向指标
        elif change > 5000:
            signal = 'BULLISH'
            confidence = 65
            description = f'净空单单日大增{change/10000:.1f}万手，可能预示短期底部'

        # 情况3: 连续3天净空单增加 -> 累积风险
        elif consecutive_increase >= 3:
            signal = 'BEARISH'
            confidence = 60
            description = f'净空单连续{consecutive_increase}天增加，机构看空情绪累积'

        # 情况4: 净空单大幅下降(>5000手) -> 机构减空
        elif change < -5000:
            signal = 'BEARISH'
            confidence = 55
            description = f'净空单减少{abs(change)/10000:.1f}万手，机构减持空头'

        # 情况5: 净空单在低位且下降
        elif current_net < avg_net and change < 0:
            signal = 'BEARISH'
            confidence = 55
            description = '净空单低于均值且继续下降，机构看多'

        else:
            description = f'净空单稳定({current_net/10000:.1f}万手)，无明显信号'

        return signal, confidence, description


def get_cffex_signal() -> Dict:
    """
    获取CFFEX信号的便捷函数

    Returns:
        {'signal': str, 'confidence': float, 'description': str,
         'net_short': int, 'change': int, 'dates': [...]}
    """
    cffex = CFFEXSignal()

    try:
        signal, confidence, description = cffex.generate_signal()

        recent = cffex.get_recent_net_short(days=5)

        return {
            'signal': signal,
            'confidence': confidence,
            'description': description,
            'net_short': recent['values'][-1] if recent['values'] else 0,
            'change': recent['changes'][-1] if recent['changes'] else 0,
            'dates': recent['dates'],
            'values': recent['values'],
            'changes': recent['changes']
        }
    except Exception as e:
        print(f"[CFFEX] 信号生成失败: {e}")
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'description': f'获取失败: {e}',
            'net_short': 0,
            'change': 0,
            'dates': [],
            'values': [],
            'changes': []
        }


def print_cffex_status():
    """打印CFFEX信号状态"""
    signal_data = get_cffex_signal()

    print("\n" + "=" * 60)
    print("机构持仓信号 (CFFEX中信期货)")
    print("=" * 60)

    if signal_data['dates']:
        print(f"\n最近5个交易日净空单:")
        for i, date in enumerate(signal_data['dates']):
            val = signal_data['values'][i]
            chg = signal_data['changes'][i]
            chg_str = f"{chg:+,.0f}" if chg != 0 else "-"
            print(f"  {date}: {val:>10,} 手 ({chg_str})")

    print(f"\n信号: {signal_data['signal']}")
    print(f"置信度: {signal_data['confidence']}%")
    print(f"描述: {signal_data['description']}")
    print("=" * 60)


if __name__ == '__main__':
    print_cffex_status()