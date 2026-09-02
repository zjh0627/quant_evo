#!/usr/bin/env python3
"""
因子计算模块
基于获取的数据计算各种技术因子和基本面因子
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class FactorCalculator:
    """因子计算器"""

    def __init__(self):
        pass

    def calculate_momentum(self, df: pd.DataFrame, periods: list = [5, 10, 20, 60]) -> pd.DataFrame:
        """动量因子"""
        result = df.copy()

        for period in periods:
            result[f'momentum_{period}d'] = result['close'].pct_change(period)

        # 相对强弱
        result['rs_5_20'] = result['momentum_5d'] - result['momentum_20d']

        return result

    def calculate_volatility(self, df: pd.DataFrame, periods: list = [5, 10, 20, 60]) -> pd.DataFrame:
        """波动率因子"""
        result = df.copy()

        for period in periods:
            result[f'volatility_{period}d'] = result['close'].pct_change().rolling(period).std()

        # 波动率比值
        result['vol_ratio_5_20'] = result['volatility_5d'] / result['volatility_20d']

        return result

    def calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """趋势因子 - 均线系统"""
        result = df.copy()

        # 移动平均线
        for period in [5, 10, 20, 60]:
            result[f'ma_{period}'] = result['close'].rolling(period).mean()

        # 均线多头排列
        result['ma_bull'] = (
            (result['ma_5'] > result['ma_10']) &
            (result['ma_10'] > result['ma_20']) &
            (result['ma_20'] > result['ma_60'])
        ).astype(int)

        # 价格与均线的关系
        result['price_ma20_ratio'] = result['close'] / result['ma_20']
        result['price_ma60_ratio'] = result['close'] / result['ma_60']

        # MACD
        exp1 = result['close'].ewm(span=12, adjust=False).mean()
        exp2 = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = exp1 - exp2
        result['signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['signal']

        return result

    def calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """量价因子"""
        result = df.copy()

        # 成交量移动平均
        result['volume_ma5'] = result['volume'].rolling(5).mean()
        result['volume_ma20'] = result['volume'].rolling(20).mean()

        # 量价配合
        result['volume_ratio'] = result['volume'] / result['volume_ma20']
        result['price_volume_corr'] = result['close'].rolling(10).corr(result['volume'])

        # 资金流
        result['money_flow'] = result['close'] * result['volume']
        result['money_flow_ma5'] = result['money_flow'].rolling(5).mean()

        return result

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """RSI 相对强弱指数"""
        result = df.copy()

        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))

        # 超买超卖
        result['rsi_overbought'] = (result['rsi'] > 70).astype(int)
        result['rsi_oversold'] = (result['rsi'] < 30).astype(int)

        return result

    def calculate_bollinger(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """布林带"""
        result = df.copy()

        result['bb_mid'] = result['close'].rolling(period).mean()
        result['bb_std'] = result['close'].rolling(period).std()
        result['bb_upper'] = result['bb_mid'] + std_dev * result['bb_std']
        result['bb_lower'] = result['bb_mid'] - std_dev * result['bb_std']
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_mid']
        result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])

        return result

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR 平均真实波幅"""
        result = df.copy()

        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr'] = true_range.rolling(period).mean()
        result['atr_ratio'] = result['atr'] / result['close']

        return result

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ADX 平均方向性指数 - 趋势强度指标"""
        result = df.copy()

        high_diff = result['high'].diff()
        low_diff = -result['low'].diff()

        # +DM 和 -DM
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # True Range
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # 平滑
        atr = tr.rolling(period).mean()
        plus_di = (plus_dm.rolling(period).mean() / atr) * 100
        minus_di = (minus_dm.rolling(period).mean() / atr) * 100

        # DX
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

        # ADX
        result['adx'] = dx.rolling(period).mean()
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        result['adx_signal'] = np.where(plus_di > minus_di, 'long', 'short')

        return result

    def calculate_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """资金流因子 - 基于成交量和价格"""
        result = df.copy()

        if 'volume' not in result.columns:
            return result

        # 典型价格
        tp = (result['high'] + result['low'] + result['close']) / 3
        money_flow = tp * result['volume']

        # 资金流累计
        result['money_flow'] = money_flow.rolling(period).sum()
        result['money_ratio'] = money_flow.rolling(period).sum() / money_flow.rolling(period).sum().shift()

        # 资金流入/流出判断
        price_up = result['close'] > result['close'].shift()
        result['money_inflow'] = money_flow.where(price_up, 0).rolling(period).sum()
        result['money_outflow'] = money_flow.where(~price_up, 0).rolling(period).sum()
        result['money_net'] = result['money_inflow'] - result['money_outflow']

        # 资金流强度
        result['money_strength'] = result['money_net'] / result['money_flow'] if result['money_flow'] != 0 else 0

        return result

    def calculate_cci(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """CCI 商品通道指数"""
        result = df.copy()

        tp = (result['high'] + result['low'] + result['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        result['cci'] = (tp - sma) / (0.015 * mad)

        return result

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        if df is None or len(df) < 60:
            return pd.DataFrame()

        result = df.copy()

        # 基础因子
        result = self.calculate_momentum(result)
        result = self.calculate_volatility(result)
        result = self.calculate_trend(result)
        result = self.calculate_volume(result)
        result = self.calculate_rsi(result)
        result = self.calculate_bollinger(result)
        result = self.calculate_atr(result)
        result = self.calculate_cci(result)

        return result

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        基于因子生成交易信号
        返回: 'BUY', 'SELL', 'HOLD'
        """
        if df is None or len(df) < 20:
            return 'HOLD'

        latest = df.iloc[-1]

        score = 0

        # 动量评分
        if latest.get('momentum_5d', 0) > 0.02:
            score += 1
        if latest.get('momentum_20d', 0) > 0:
            score += 1

        # 趋势评分
        if latest.get('ma_bull', 0) == 1:
            score += 2
        if latest.get('price_ma20_ratio', 1) > 1:
            score += 1

        # MACD
        if latest.get('macd', 0) > latest.get('signal', 0):
            score += 1
        if latest.get('macd_hist', 0) > 0:
            score += 1

        # RSI
        if latest.get('rsi', 50) < 40:
            score += 1  # 超卖，低位买入信号
        elif latest.get('rsi', 50) > 70:
            score -= 1  # 超买

        # 布林带
        if latest.get('bb_position', 0.5) < 0.2:
            score += 1  # 接近下轨

        # 综合评分
        if score >= 5:
            return 'BUY'
        elif score <= 1:
            return 'SELL'
        else:
            return 'HOLD'

    def get_composite_score(self, df: pd.DataFrame) -> float:
        """
        获取综合评分 (0-100)
        """
        if df is None or len(df) < 20:
            return 50.0

        latest = df.iloc[-1]
        score = 50.0

        # 动量因子 (+/- 15)
        momentum = latest.get('momentum_5d', 0) * 100
        score += min(max(momentum, -10), 10) * 1.5

        # 趋势因子 (+/- 20)
        if latest.get('ma_bull', 0) == 1:
            score += 10
        price_ma_ratio = latest.get('price_ma20_ratio', 1)
        if price_ma_ratio > 1:
            score += 5
        if price_ma_ratio > 1.1:
            score += 5

        # MACD (+/- 10)
        if latest.get('macd_hist', 0) > 0:
            score += 5
        if latest.get('macd', 0) > latest.get('signal', 0):
            score += 5

        # RSI (0-15)
        rsi = latest.get('rsi', 50)
        if rsi < 30:
            score += 15
        elif rsi < 40:
            score += 10
        elif rsi > 70:
            score -= 10
        elif rsi > 60:
            score -= 5

        # 布林带 (+/- 10)
        bb_pos = latest.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 10
        elif bb_pos > 0.8:
            score -= 10
        elif bb_pos < 0.3:
            score += 5
        elif bb_pos > 0.7:
            score -= 5

        return min(max(score, 0), 100)


def calculate_stock_factors(code: str, df: pd.DataFrame) -> Dict:
    """为单只股票计算因子"""
    calc = FactorCalculator()
    factors_df = calc.calculate_all_factors(df)

    if factors_df.empty:
        return {}

    latest = factors_df.iloc[-1]

    return {
        'code': code,
        'close': latest.get('close'),
        'momentum_5d': latest.get('momentum_5d', 0),
        'momentum_20d': latest.get('momentum_20d', 0),
        'volatility_20d': latest.get('volatility_20d', 0),
        'ma_bull': latest.get('ma_bull', 0),
        'rsi': latest.get('rsi', 50),
        'cci': latest.get('cci', 0),
        'signal': calc.generate_signal(factors_df),
        'score': calc.get_composite_score(factors_df)
    }


if __name__ == '__main__':
    # 测试
    import os
    from data_pipeline import DataPipeline

    pipeline = DataPipeline()
    df = pipeline.load_from_csv('sh.600519')

    if df is not None:
        calc = FactorCalculator()
        factors = calc.calculate_all_factors(df)
        print(f"\n贵州茅台因子计算完成: {len(factors)} 行")
        print(factors.tail(5))

        signal = calc.generate_signal(factors)
        score = calc.get_composite_score(factors)
        print(f"\n信号: {signal}, 评分: {score:.1f}")
