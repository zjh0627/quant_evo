#!/usr/bin/env python3
"""
均线交叉策略
============

买入条件: MA5 上穿 MA20 (金叉)
卖出条件: MA5 下穿 MA20 (死叉)

原理: 短期均线上穿长期均线表示上涨趋势形成,买入;反之下跌
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MACrossoverStrategy:
    """均线交叉策略"""

    def __init__(self,
                 fast_ma: int = 5,
                 slow_ma: int = 20,
                 signal_ma: int = 5,
                 use_signal: bool = False):
        """
        Args:
            fast_ma: 快速均线周期
            slow_ma: 慢速均线周期
            signal_ma: 信号线周期 (用于MACD类似策略)
            use_signal: 是否使用信号线过滤
        """
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.signal_ma = signal_ma
        self.use_signal = use_signal

    def calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """计算移动平均线"""
        return prices.rolling(period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            df: 包含 'date', 'close', 'open', 'high', 'low' 列的DataFrame

        Returns:
            添加了 'signal', 'score', 'ma_fast', 'ma_slow', 'cross' 列的DataFrame
        """
        result = df.copy()
        close = result['close']

        # 计算均线
        if self.use_signal:
            result['ma_fast'] = self.calculate_ema(close, self.fast_ma)
            result['ma_slow'] = self.calculate_ema(close, self.slow_ma)
            result['ma_signal'] = self.calculate_ema(result['ma_fast'] - result['ma_slow'], self.signal_ma)
        else:
            result['ma_fast'] = self.calculate_ma(close, self.fast_ma)
            result['ma_slow'] = self.calculate_ma(close, self.slow_ma)

        # 计算交叉信号
        result['diff'] = result['ma_fast'] - result['ma_slow']
        result['prev_diff'] = result['diff'].shift(1)

        # 金叉: fast上穿slow (diff从负变正或diff>0但前一天<0)
        result['golden_cross'] = (result['diff'] > 0) & (result['prev_diff'] <= 0)

        # 死叉: fast下穿slow (diff从正变负或diff<0但前一天>0)
        result['death_cross'] = (result['diff'] < 0) & (result['prev_diff'] >= 0)

        # 初始化信号
        result['signal'] = 'HOLD'
        result['score'] = 50.0
        result['cross'] = ''

        # 金叉买入
        buy_mask = result['golden_cross']
        result.loc[buy_mask, 'signal'] = 'BUY'
        result.loc[buy_mask, 'score'] = 70 + (result.loc[buy_mask, 'diff'] / result.loc[buy_mask, 'ma_slow'] * 100).clip(0, 30)
        result.loc[buy_mask, 'cross'] = 'GOLDEN'

        # 死叉卖出
        sell_mask = result['death_cross']
        result.loc[sell_mask, 'signal'] = 'SELL'
        result.loc[sell_mask, 'score'] = 70 - (result.loc[sell_mask, 'diff'] / result.loc[sell_mask, 'ma_slow'] * 100).clip(0, 30)
        result.loc[sell_mask, 'cross'] = 'DEATH'

        return result[['date', 'open', 'high', 'low', 'close', 'volume',
                       'signal', 'score', 'ma_fast', 'ma_slow', 'cross']]

    def should_buy(self, ma_fast: float, ma_slow: float, prev_fast: float, prev_slow: float) -> Tuple[bool, float]:
        """判断是否应该买入 (金叉)"""
        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return False, 50.0

        if prev_fast <= prev_slow and ma_fast > ma_slow:
            # 金叉
            strength = (ma_fast - ma_slow) / ma_slow * 100 if ma_slow > 0 else 0
            score = 70 + min(strength, 30)
            return True, score

        return False, 50.0

    def should_sell(self, ma_fast: float, ma_slow: float, prev_fast: float, prev_slow: float) -> Tuple[bool, str]:
        """判断是否应该卖出 (死叉)"""
        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return False, ''

        if prev_fast >= prev_slow and ma_fast < ma_slow:
            # 死叉
            return True, 'DEATH_CROSS'

        return False, ''


class MACrossoverStrategyGenerator:
    """均线交叉策略生成器"""

    @staticmethod
    def create_short_term() -> MACrossoverStrategy:
        """短期策略: MA5/MA10"""
        return MACrossoverStrategy(fast_ma=5, slow_ma=10)

    @staticmethod
    def create_medium_term() -> MACrossoverStrategy:
        """中期策略: MA5/MA20"""
        return MACrossoverStrategy(fast_ma=5, slow_ma=20)

    @staticmethod
    def create_long_term() -> MACrossoverStrategy:
        """长期策略: MA10/MA60"""
        return MACrossoverStrategy(fast_ma=10, slow_ma=60)

    @staticmethod
    def create_ema_crossover() -> MACrossoverStrategy:
        """EMA交叉策略"""
        return MACrossoverStrategy(fast_ma=12, slow_ma=26, use_signal=True)


def generate_signals_for_stocks(
    stock_data: Dict[str, pd.DataFrame],
    strategy: MACrossoverStrategy = None
) -> Dict[str, pd.DataFrame]:
    """
    为多只股票生成均线交叉策略信号

    Args:
        stock_data: {code: dataframe} 股票数据
        strategy: 均线交叉策略实例

    Returns:
        {code: signal_dataframe} 信号数据
    """
    if strategy is None:
        strategy = MACrossoverStrategyGenerator.create_medium_term()

    signals_dict = {}
    for code, df in stock_data.items():
        if df is None or len(df) < max(strategy.fast_ma, strategy.slow_ma) + 5:
            continue
        try:
            signals = strategy.generate_signals(df)
            signals_dict[code] = signals
        except Exception as e:
            print(f"Error generating signals for {code}: {e}")
            continue

    return signals_dict


def main():
    """测试均线交叉策略"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')
    from src.data.data_pipeline import DataPipeline

    print("=" * 60)
    print("均线交叉策略回测")
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
    strategy = MACrossoverStrategyGenerator.create_medium_term()
    signals = generate_signals_for_stocks(stock_data, strategy)

    # 统计信号
    for code, df in signals.items():
        buy_count = (df['signal'] == 'BUY').sum()
        sell_count = (df['signal'] == 'SELL').sum()
        print(f"{code}: BUY={buy_count}, SELL={sell_count}, HOLD={len(df)-buy_count-sell_count}")

        # 显示最近信号
        recent = df[df['signal'] != 'HOLD'].tail(5)
        if len(recent) > 0:
            print(f"  最近信号:")
            for _, row in recent.iterrows():
                print(f"    {row['date']} {row['signal']} {row['cross']} MA{strategy.fast_ma}:{row['ma_fast']:.2f} MA{strategy.slow_ma}:{row['ma_slow']:.2f}")


if __name__ == '__main__':
    main()
