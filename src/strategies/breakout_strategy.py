#!/usr/bin/env python3
"""
突破策略
============

买入条件: 价格突破N日最高价
卖出条件: 价格跌破M日最低价 或 跌破买入价-止损

原理: 突破新高代表强势,顺势买入;跌破新低代表弱势,止损卖出
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class BreakoutStrategy:
    """突破策略"""

    def __init__(self,
                 breakout_period: int = 20,
                 stop_period: int = 10,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 use_trailing_stop: bool = True):
        """
        Args:
            breakout_period: 突破判断周期 (N日高点)
            stop_period: 止损周期 (M日低点)
            atr_period: ATR周期
            atr_multiplier: ATR止损倍数
            use_trailing_stop: 是否使用追踪止损
        """
        self.breakout_period = breakout_period
        self.stop_period = stop_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_trailing_stop = use_trailing_stop

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def calculate_highest(self, series: pd.Series, period: int) -> pd.Series:
        """计算周期最高值"""
        return series.rolling(period).max()

    def calculate_lowest(self, series: pd.Series, period: int) -> pd.Series:
        """计算周期最低值"""
        return series.rolling(period).min()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            df: 包含 'date', 'close', 'open', 'high', 'low' 列的DataFrame

        Returns:
            添加了 'signal', 'score', 'breakout_high', 'stop_low', 'atr' 列的DataFrame
        """
        result = df.copy()

        # 计算指标
        result['breakout_high'] = self.calculate_highest(result['high'], self.breakout_period)
        result['stop_low'] = self.calculate_lowest(result['low'], self.stop_period)
        result['atr'] = self.calculate_atr(result, self.atr_period)

        # 突破信号: 今日收盘价突破昨日N日最高价
        result['prev_breakout_high'] = result['breakout_high'].shift(1)
        result['breakout'] = result['close'] > result['prev_breakout_high']

        # 止损信号: 价格跌破M日最低价
        result['stop_triggered'] = result['close'] < result['stop_low']

        # 初始化信号
        result['signal'] = 'HOLD'
        result['score'] = 50.0

        # 突破买入
        buy_mask = result['breakout']
        result.loc[buy_mask, 'signal'] = 'BUY'
        result.loc[buy_mask, 'score'] = 65 + (result.loc[buy_mask, 'close'] / result.loc[buy_mask, 'prev_breakout_high'] - 1).clip(0, 0.05) * 1000

        # 止损卖出
        sell_mask = result['stop_triggered']
        result.loc[sell_mask, 'signal'] = 'SELL'
        result.loc[sell_mask, 'score'] = 70

        return result[['date', 'open', 'high', 'low', 'close', 'volume',
                       'signal', 'score', 'breakout_high', 'stop_low', 'atr']]

    def should_buy(self, close: float, breakout_high: float, atr: float = None) -> Tuple[bool, float]:
        """判断是否应该买入"""
        if pd.isna(breakout_high) or pd.isna(close):
            return False, 50.0

        if close > breakout_high:
            strength = (close / breakout_high - 1) * 100 if breakout_high > 0 else 0
            score = 65 + min(strength * 10, 35)
            return True, score

        return False, 50.0

    def should_sell(self, close: float, stop_low: float, entry_price: float = None,
                    atr: float = None, high_since_entry: float = None) -> Tuple[bool, str]:
        """判断是否应该卖出"""
        if pd.isna(stop_low) or pd.isna(close):
            return False, ''

        # 跌破止损低
        if close < stop_low:
            return True, 'STOP_LOSS'

        # ATR止损
        if atr and entry_price:
            atr_stop = entry_price - atr * self.atr_multiplier
            if close < atr_stop:
                return True, 'ATR_STOP'

        # 追踪止损 (从高点回落一定比例)
        if self.use_trailing_stop and high_since_entry:
            trailing_stop = high_since_entry * 0.95  # 从高点回落5%
            if close < trailing_stop:
                return True, 'TRAILING_STOP'

        return False, ''

    def get_stop_price(self, entry_price: float, atr: float = None) -> float:
        """获取止损价格"""
        if atr:
            return entry_price - atr * self.atr_multiplier
        return entry_price * 0.95  # 默认5%止损


class BreakoutStrategyGenerator:
    """突破策略生成器"""

    @staticmethod
    def create_short_term() -> BreakoutStrategy:
        """短期突破: 10日"""
        return BreakoutStrategy(
            breakout_period=10,
            stop_period=5,
            atr_period=10,
            atr_multiplier=1.5
        )

    @staticmethod
    def create_medium_term() -> BreakoutStrategy:
        """中期突破: 20日"""
        return BreakoutStrategy(
            breakout_period=20,
            stop_period=10,
            atr_period=14,
            atr_multiplier=2.0
        )

    @staticmethod
    def create_long_term() -> BreakoutStrategy:
        """长期突破: 50日"""
        return BreakoutStrategy(
            breakout_period=50,
            stop_period=20,
            atr_period=20,
            atr_multiplier=2.5
        )

    @staticmethod
    def create_aggressive() -> BreakoutStrategy:
        """激进突破: 高频交易"""
        return BreakoutStrategy(
            breakout_period=5,
            stop_period=3,
            atr_period=5,
            atr_multiplier=1.0,
            use_trailing_stop=False
        )


def generate_signals_for_stocks(
    stock_data: Dict[str, pd.DataFrame],
    strategy: BreakoutStrategy = None
) -> Dict[str, pd.DataFrame]:
    """为多只股票生成突破策略信号"""
    if strategy is None:
        strategy = BreakoutStrategyGenerator.create_medium_term()

    signals_dict = {}
    for code, df in stock_data.items():
        if df is None or len(df) < max(strategy.breakout_period, strategy.stop_period) + 5:
            continue
        try:
            signals = strategy.generate_signals(df)
            signals_dict[code] = signals
        except Exception as e:
            print(f"Error generating signals for {code}: {e}")
            continue

    return signals_dict


def main():
    """测试突破策略"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')
    from src.data.data_pipeline import DataPipeline

    print("=" * 60)
    print("突破策略回测")
    print("=" * 60)

    pipeline = DataPipeline()
    codes = ['sh.600519', 'sh.600036', 'sh.601318']
    stock_data = {}

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 60:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    strategy = BreakoutStrategyGenerator.create_medium_term()
    signals = generate_signals_for_stocks(stock_data, strategy)

    for code, df in signals.items():
        buy_count = (df['signal'] == 'BUY').sum()
        sell_count = (df['signal'] == 'SELL').sum()
        print(f"{code}: BUY={buy_count}, SELL={sell_count}, HOLD={len(df)-buy_count-sell_count}")

        recent = df[df['signal'] != 'HOLD'].tail(3)
        if len(recent) > 0:
            print(f"  最近信号:")
            for _, row in recent.iterrows():
                print(f"    {row['date']} {row['signal']} 收盘:{row['close']:.2f} 突破:{row['breakout_high']:.2f}")


if __name__ == '__main__':
    main()
