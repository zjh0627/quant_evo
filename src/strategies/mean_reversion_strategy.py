#!/usr/bin/env python3
"""
均值回归策略
============

买入条件: 20日收益率 < -8% (严重超跌)
卖出条件: 收益率回归到 -2% 或 10日内未盈利

原理: 价格严重偏离均值后会回归,逆势买入等待反弹
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MeanReversionStrategy:
    """均值回归策略"""

    def __init__(self,
                 lookback: int = 20,
                 buy_threshold: float = -0.08,
                 sell_threshold: float = -0.02,
                 max_hold_days: int = 10,
                 use_volatility: bool = True):
        """
        Args:
            lookback: 回看周期
            buy_threshold: 买入阈值 (负值表示下跌)
            sell_threshold: 卖出阈值 (回升到正值卖出)
            max_hold_days: 最大持仓天数
            use_volatility: 是否使用波动率调整阈值
        """
        self.lookback = lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_hold_days = max_hold_days
        self.use_volatility = use_volatility

    def calculate_returns(self, prices: pd.Series, period: int) -> pd.Series:
        """计算周期收益率"""
        return prices.pct_change(period)

    def calculate_rolling_mean(self, prices: pd.Series, period: int) -> pd.Series:
        """计算滚动均值"""
        return prices.rolling(period).mean()

    def calculate_z_score(self, prices: pd.Series) -> pd.Series:
        """计算Z分数 (价格偏离均值的标准差倍数)"""
        ma = prices.rolling(self.lookback).mean()
        std = prices.rolling(self.lookback).std()
        z = (prices - ma) / std.replace(0, np.nan)
        return z.fillna(0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            df: 包含 'date', 'close', 'open', 'high', 'low' 列的DataFrame

        Returns:
            添加了 'signal', 'score', 'return_20d', 'z_score' 列的DataFrame
        """
        result = df.copy()
        close = result['close']

        # 计算20日收益率
        result['return_20d'] = self.calculate_returns(close, 20)

        # 计算Z分数
        if self.use_volatility:
            result['z_score'] = self.calculate_z_score(close)
        else:
            result['z_score'] = result['return_20d'] / 0.1  # 简化为固定阈值

        # 初始化信号
        result['signal'] = 'HOLD'
        result['score'] = 50.0

        # 超跌买入: 收益率 < -8% 或 Z分数 < -2
        buy_threshold_adjusted = self.buy_threshold
        if self.use_volatility:
            buy_threshold_adjusted = result['z_score'] < -1.5
            result.loc[buy_threshold_adjusted, 'signal'] = 'BUY'
            result.loc[buy_threshold_adjusted, 'score'] = 70 + (-result.loc[buy_threshold_adjusted, 'z_score'] * 10).clip(0, 30)
        else:
            buy_condition = result['return_20d'] < self.buy_threshold
            result.loc[buy_condition, 'signal'] = 'BUY'
            result.loc[buy_condition, 'score'] = 70 + ((self.buy_threshold - result.loc[buy_condition, 'return_20d']) * 100).clip(0, 30)

        # 回归卖出: 收益率 > -2% 或 Z分数 > 0
        sell_threshold_adjusted = result['return_20d'] > self.sell_threshold
        result.loc[sell_threshold_adjusted & (result['signal'] == 'BUY'), 'signal'] = 'SELL'
        result.loc[sell_threshold_adjusted & (result['signal'] == 'SELL'), 'score'] = 65

        return result[['date', 'open', 'high', 'low', 'close', 'volume',
                       'signal', 'score', 'return_20d', 'z_score']]

    def should_buy(self, returns: float, z_score: float = None) -> Tuple[bool, float]:
        """判断是否应该买入"""
        if self.use_volatility and z_score is not None:
            if z_score < -1.5:
                score = 70 + (-z_score * 10)
                return True, score
        else:
            if returns < self.buy_threshold:
                score = 70 + (self.buy_threshold - returns) * 100
                return True, score
        return False, 50.0

    def should_sell(self, returns: float, hold_days: int, entry_return: float = None, z_score: float = None) -> Tuple[bool, str]:
        """判断是否应该卖出"""
        # 收益率回升
        if returns > self.sell_threshold:
            return True, 'PROFIT_TAKEN'

        # 超过最大持仓
        if hold_days >= self.max_hold_days:
            if entry_return and returns > entry_return:
                return True, 'BREAKEVEN'
            return True, 'MAX_HOLD'

        # Z分数回归
        if self.use_volatility and z_score is not None:
            if z_score > 0:
                return True, 'Z_SCORE_NORMALIZED'

        return False, ''


class MeanReversionStrategyGenerator:
    """均值回归策略生成器"""

    @staticmethod
    def create_conservative() -> MeanReversionStrategy:
        """保守策略 - 更深跌幅才买"""
        return MeanReversionStrategy(
            lookback=20,
            buy_threshold=-0.12,  # 跌12%才买
            sell_threshold=0.0,    # 回到原点卖
            max_hold_days=15,
            use_volatility=False
        )

    @staticmethod
    def create_aggressive() -> MeanReversionStrategy:
        """激进策略 - 更频繁交易"""
        return MeanReversionStrategy(
            lookback=10,
            buy_threshold=-0.05,  # 跌5%就买
            sell_threshold=-0.01,
            max_hold_days=5,
            use_volatility=False
        )

    @staticmethod
    def create_balanced() -> MeanReversionStrategy:
        """平衡策略"""
        return MeanReversionStrategy(
            lookback=20,
            buy_threshold=-0.08,
            sell_threshold=-0.02,
            max_hold_days=10,
            use_volatility=True
        )


def generate_signals_for_stocks(
    stock_data: Dict[str, pd.DataFrame],
    strategy: MeanReversionStrategy = None
) -> Dict[str, pd.DataFrame]:
    """为多只股票生成均值回归策略信号"""
    if strategy is None:
        strategy = MeanReversionStrategyGenerator.create_balanced()

    signals_dict = {}
    for code, df in stock_data.items():
        if df is None or len(df) < strategy.lookback + 5:
            continue
        try:
            signals = strategy.generate_signals(df)
            signals_dict[code] = signals
        except Exception as e:
            print(f"Error generating signals for {code}: {e}")
            continue

    return signals_dict


def main():
    """测试均值回归策略"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')
    from src.data.data_pipeline import DataPipeline

    print("=" * 60)
    print("均值回归策略回测")
    print("=" * 60)

    pipeline = DataPipeline()
    codes = ['sh.600519', 'sh.600036', 'sh.601318']
    stock_data = {}

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 60:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    strategy = MeanReversionStrategyGenerator.create_balanced()
    signals = generate_signals_for_stocks(stock_data, strategy)

    for code, df in signals.items():
        buy_count = (df['signal'] == 'BUY').sum()
        sell_count = (df['signal'] == 'SELL').sum()
        print(f"{code}: BUY={buy_count}, SELL={sell_count}, HOLD={len(df)-buy_count-sell_count}")

        recent = df[df['signal'] != 'HOLD'].tail(3)
        if len(recent) > 0:
            print(f"  最近信号:")
            for _, row in recent.iterrows():
                print(f"    {row['date']} {row['signal']} 20日收益:{row['return_20d']*100:.1f}%")


if __name__ == '__main__':
    main()
