#!/usr/bin/env python3
"""
RSI 反向策略
============

买入条件: RSI < 30 (超卖)
卖出条件: RSI > 70 (超买) 或 RSI > 50 (上升趋势破坏)
持仓周期: 5-10天

原理: 超卖时买入,超买时卖出,逆势而行
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class RSIStrategy:
    """RSI反向策略"""

    def __init__(self,
                 rsi_period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 sell_rsi: float = 50,
                 min_hold_days: int = 3,
                 max_hold_days: int = 15):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.sell_rsi = sell_rsi
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            df: 包含 'date', 'close', 'open', 'high', 'low' 列的DataFrame

        Returns:
            添加了 'signal', 'score', 'rsi' 列的DataFrame
        """
        result = df.copy()
        close = result['close']

        # 计算RSI
        result['rsi'] = self.calculate_rsi(close)

        # 初始化信号
        result['signal'] = 'HOLD'
        result['score'] = 50.0

        # RSI超卖 - 买入信号
        buy_condition = result['rsi'] < self.oversold
        result.loc[buy_condition, 'signal'] = 'BUY'
        result.loc[buy_condition, 'score'] = 70 + (self.oversold - result.loc[buy_condition, 'rsi'])

        # RSI超买 - 卖出信号
        sell_condition = result['rsi'] > self.overbought
        result.loc[sell_condition, 'signal'] = 'SELL'
        result.loc[sell_condition, 'score'] = 70 + (result.loc[sell_condition, 'rsi'] - self.overbought)

        return result[['date', 'open', 'high', 'low', 'close', 'volume', 'signal', 'score', 'rsi']]

    def should_buy(self, rsi: float) -> Tuple[bool, float]:
        """判断是否应该买入"""
        if rsi < self.oversold:
            score = 70 + (self.oversold - rsi)
            return True, score
        return False, 50.0

    def should_sell(self, rsi: float, hold_days: int, entry_rsi: float = None) -> Tuple[bool, str]:
        """
        判断是否应该卖出

        Args:
            rsi: 当前RSI
            hold_days: 持仓天数
            entry_rsi: 买入时的RSI

        Returns:
            (是否卖出, 卖出原因)
        """
        # RSI超过70超买
        if rsi > self.overbought:
            return True, 'RSI_OVERBOUGHT'

        # 持仓超过最大天数
        if hold_days >= self.max_hold_days:
            return True, 'MAX_HOLD'

        # RSI回到50以上且盈利 (止盈)
        if rsi > self.sell_rsi and hold_days >= self.min_hold_days:
            return True, 'RSI_PROFIT'

        # RSI从超卖回到50以上但价格没涨 (止损)
        if entry_rsi and entry_rsi < self.oversold and rsi > 50:
            if hold_days >= self.min_hold_days:
                return True, 'RSI_NORMALIZED'

        return False, ''


class RSIStrategyGenerator:
    """RSI策略生成器"""

    @staticmethod
    def create_conservative() -> RSIStrategy:
        """保守策略 - 更极端的超卖超买"""
        return RSIStrategy(
            rsi_period=14,
            oversold=25,      # 更低才买
            overbought=75,    # 更高才卖
            sell_rsi=55,
            min_hold_days=5,
            max_hold_days=20
        )

    @staticmethod
    def create_aggressive() -> RSIStrategy:
        """激进策略 - 更频繁交易"""
        return RSIStrategy(
            rsi_period=10,   # 更短周期
            oversold=35,      # 更高就买
            overbought=65,    # 更低就卖
            sell_rsi=45,
            min_hold_days=2,
            max_hold_days=8
        )

    @staticmethod
    def create_balanced() -> RSIStrategy:
        """平衡策略"""
        return RSIStrategy(
            rsi_period=14,
            oversold=30,
            overbought=70,
            sell_rsi=50,
            min_hold_days=3,
            max_hold_days=15
        )


def generate_signals_for_stocks(
    stock_data: Dict[str, pd.DataFrame],
    strategy: RSIStrategy = None
) -> Dict[str, pd.DataFrame]:
    """
    为多只股票生成RSI策略信号

    Args:
        stock_data: {code: dataframe} 股票数据
        strategy: RSI策略实例

    Returns:
        {code: signal_dataframe} 信号数据
    """
    if strategy is None:
        strategy = RSIStrategyGenerator.create_balanced()

    signals_dict = {}
    for code, df in stock_data.items():
        if df is None or len(df) < 30:
            continue
        try:
            signals = strategy.generate_signals(df)
            signals_dict[code] = signals
        except Exception as e:
            print(f"Error generating signals for {code}: {e}")
            continue

    return signals_dict


def main():
    """测试RSI策略"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')
    from src.data.data_pipeline import DataPipeline

    print("=" * 60)
    print("RSI策略回测")
    print("=" * 60)

    # 加载数据
    pipeline = DataPipeline()
    codes = ['sh.600519', 'sh.600036', 'sh.601318']
    stock_data = {}

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 60:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    # 生成信号
    strategy = RSIStrategyGenerator.create_balanced()
    signals = generate_signals_for_stocks(stock_data, strategy)

    # 统计信号
    for code, df in signals.items():
        buy_count = (df['signal'] == 'BUY').sum()
        sell_count = (df['signal'] == 'SELL').sum()
        print(f"{code}: BUY={buy_count}, SELL={sell_count}, HOLD={len(df)-buy_count-sell_count}")

        # 显示最近信号
        recent = df[df['signal'] != 'HOLD'].tail(3)
        if len(recent) > 0:
            print(f"  最近信号:")
            for _, row in recent.iterrows():
                print(f"    {row['date']} {row['signal']} RSI:{row['rsi']:.1f} 价格:{row['close']:.2f}")


if __name__ == '__main__':
    main()
