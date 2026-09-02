#!/usr/bin/env python3
"""
Factor Discovery Agent - 因子挖掘Agent
从市场数据中发现有效预测因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.agents.base import Agent, Message


@dataclass
class FactorInfo:
    """因子信息"""
    name: str
    category: str  # momentum, trend, volatility, volume, fundamental
    ic: float      # Information Coefficient
    ir: float      # Information Ratio
    p_value: float
    description: str


class FactorDiscoveryAgent(Agent):
    """因子挖掘Agent"""

    def __init__(self):
        super().__init__("FactorDiscovery")
        self.discovered_factors: List[FactorInfo] = []
        self.factor_library: Dict[str, Any] = {}

    def process(self, message: Optional[Message] = None) -> Optional[Message]:
        """处理消息"""
        if message and message.content.get('action') == 'discover':
            return self.discover_factors(message.content)

        # 默认：发现新因子
        return self.discover_factors({})

    def discover_factors(self, context: Dict[str, Any]) -> Optional[Message]:
        """发现有效因子"""
        stock_data = context.get('stock_data', {})

        if not stock_data:
            return None

        print(f"[FactorDiscovery] 开始发现因子, 股票数量: {len(stock_data)}")

        # 预定义因子池
        factor_candidates = self._get_factor_candidates()

        valid_factors = []

        # 对每只股票计算因子IC
        for code, df in stock_data.items():
            if df is None or len(df) < 60:
                continue

            # 计算未来收益
            df = df.copy()
            df['future_return'] = df['close'].pct_change(5).shift(-5)

            for factor_name, factor_func in factor_candidates.items():
                try:
                    df[factor_name] = factor_func(df)

                    # 计算IC
                    ic = df[factor_name].corr(df['future_return'])
                    ir = ic / df[factor_name].std() if df[factor_name].std() > 0 else 0

                    if abs(ic) > 0.02:  # IC阈值
                        valid_factors.append({
                            'name': factor_name,
                            'ic': ic,
                            'ir': ir,
                            'stock': code
                        })
                except:
                    continue

        # 汇总因子有效性
        factor_summary = self._summarize_factors(valid_factors)

        print(f"[FactorDiscovery] 发现 {len(factor_summary)} 个有效因子")

        self.set_state('discovered_factors', factor_summary)
        self.set_state('last_discovery', len(factor_summary))

        return self.send('Orchestrator', {
            'action': 'discovery_complete',
            'factors': factor_summary
        }, 'RESPONSE')

    def _get_factor_candidates(self) -> Dict[str, callable]:
        """获取候选因子函数"""
        return {
            # 动量因子
            'momentum_5d': lambda df: df['close'].pct_change(5),
            'momentum_10d': lambda df: df['close'].pct_change(10),
            'momentum_20d': lambda df: df['close'].pct_change(20),
            'momentum_5_20_diff': lambda df: df['close'].pct_change(5) - df['close'].pct_change(20),

            # 趋势因子
            'ma5_ma20_ratio': lambda df: df['close'].rolling(5).mean() / df['close'].rolling(20).mean(),
            'price_ma20_ratio': lambda df: df['close'] / df['close'].rolling(20).mean(),
            'ma_bull_alignment': lambda df: (df['close'].rolling(5).mean() > df['close'].rolling(10).mean()).astype(int) +
                                          (df['close'].rolling(10).mean() > df['close'].rolling(20).mean()).astype(int),

            # 波动率因子
            'volatility_5d': lambda df: df['close'].pct_change().rolling(5).std(),
            'volatility_20d': lambda df: df['close'].pct_change().rolling(20).std(),
            'vol_ratio': lambda df: df['close'].pct_change().rolling(5).std() / df['close'].pct_change().rolling(20).std(),

            # 量价因子
            'volume_ratio': lambda df: df['volume'] / df['volume'].rolling(20).mean(),
            'volume_price_corr': lambda df: df['close'].rolling(10).corr(df['volume']),
            'money_flow': lambda df: df['close'] * df['volume'],

            # RSI
            'rsi_14': lambda df: self._calc_rsi(df['close'], 14),
            'rsi_5': lambda df: self._calc_rsi(df['close'], 5),

            # MACD
            'macd': lambda df: self._calc_macd(df['close'])[0],
            'macd_signal': lambda df: self._calc_macd(df['close'])[1],
            'macd_hist': lambda df: self._calc_macd(df['close'])[2],
        }

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calc_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

    def _summarize_factors(self, factor_results: List[Dict]) -> List[Dict]:
        """汇总因子有效性"""
        if not factor_results:
            return []

        # 按因子名分组，计算平均IC
        factor_ic = {}
        for r in factor_results:
            name = r['name']
            if name not in factor_ic:
                factor_ic[name] = []
            factor_ic[name].append(r['ic'])

        summary = []
        for name, ic_list in factor_ic.items():
            avg_ic = np.mean(ic_list)
            ir = np.mean(ic_list) / np.std(ic_list) if np.std(ic_list) > 0 else 0

            if abs(avg_ic) > 0.02:
                summary.append({
                    'name': name,
                    'avg_ic': float(avg_ic),
                    'ir': float(ir),
                    'valid_count': len(ic_list),
                    'direction': 'positive' if avg_ic > 0 else 'negative'
                })

        # 按|IC|排序
        summary.sort(key=lambda x: abs(x['avg_ic']), reverse=True)

        return summary[:10]  # 返回top10

    def get_top_factors(self, n: int = 5) -> List[Dict]:
        """获取top N因子"""
        factors = self.get_state('discovered_factors', [])
        return factors[:n]

    def register_factor_to_library(self, factor: FactorInfo):
        """注册因子到因子库"""
        self.factor_library[factor.name] = factor


if __name__ == '__main__':
    # 测试
    from src.data.data_pipeline import DataPipeline

    agent = FactorDiscoveryAgent()

    # 加载数据
    pipeline = DataPipeline()
    stock_data = {}

    for code in ['sh.600519', 'sh.600036', 'sh.688012']:
        df = pipeline.load_from_csv(code)
        if df is not None:
            stock_data[code] = df

    # 发现因子
    msg = agent.discover_factors({'stock_data': stock_data})

    if msg:
        print("\n发现的因子:")
        for f in msg.content.get('factors', [])[:5]:
            print(f"  {f['name']}: IC={f['avg_ic']:.3f}, IR={f['ir']:.3f}")
