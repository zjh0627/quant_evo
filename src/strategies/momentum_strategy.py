#!/usr/bin/env python3
"""
动量策略
============

买入条件: 60日动量排名Top 20% 且 20日动量 > 0
卖出条件: 60日动量排名 Bottom 30% 或 20日动量转负

原理: 趋势延续,强者恒强,追涨杀跌
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class MomentumStrategy:
    """动量策略"""

    def __init__(self,
                 long_period: int = 60,
                 short_period: int = 20,
                 top_percentile: float = 0.2,
                 bottom_percentile: float = 0.3,
                 min_momentum: float = 0.0):
        """
        Args:
            long_period: 长期动量周期
            short_period: 短期动量周期
            top_percentile: 买入前百分比 (top 20%)
            bottom_percentile: 卖出后百分比 (bottom 30%)
            min_momentum: 最小动量阈值
        """
        self.long_period = long_period
        self.short_period = short_period
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.min_momentum = min_momentum

    def calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """计算动量 (百分比变化)"""
        return prices.pct_change(period)

    def rank_stocks(self, momentum_series: pd.Series, min_periods: int = None) -> pd.Series:
        """对股票动量排名 (百分比排名)"""
        return momentum_series.rank(pct=True, ascending=False)

    def generate_signals(self, df: pd.DataFrame, cross_sectional: bool = False,
                        all_momentums: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            df: 包含 'date', 'close' 列的DataFrame
            cross_sectional: 是否使用横截面动量 (所有股票排序)
            all_momentums: 当cross_sectional=True时,传入所有股票的动量用于排序

        Returns:
            添加了 'signal', 'score', 'momentum_long', 'momentum_short', 'rank' 列的DataFrame
        """
        result = df.copy()
        close = result['close']

        # 计算动量
        result['momentum_long'] = self.calculate_momentum(close, self.long_period)
        result['momentum_short'] = self.calculate_momentum(close, self.short_period)

        # 初始化
        result['signal'] = 'HOLD'
        result['score'] = 50.0
        result['rank'] = np.nan

        if cross_sectional and all_momentums is not None:
            # 横截面动量: 股票间排序
            result['rank'] = all_momentums.rank(pct=True, ascending=False)

            # Top percentile 买入
            buy_mask = result['rank'] <= self.top_percentile
            result.loc[buy_mask, 'signal'] = 'BUY'
            result.loc[buy_mask, 'score'] = 70 + (self.top_percentile - result.loc[buy_mask, 'rank']) * 100

            # Bottom percentile 卖出
            sell_mask = result['rank'] >= (1 - self.bottom_percentile)
            result.loc[sell_mask, 'signal'] = 'SELL'
            result.loc[sell_mask, 'score'] = 70 + (result.loc[sell_mask, 'rank'] - (1 - self.bottom_percentile)) * 100
        else:
            # 时间序列动量: 绝对阈值
            buy_mask = (result['momentum_long'] > self.min_momentum) & \
                       (result['momentum_short'] > 0)
            result.loc[buy_mask, 'signal'] = 'BUY'
            result.loc[buy_mask, 'score'] = 60 + (result.loc[buy_mask, 'momentum_long'] * 100).clip(0, 40)

            sell_mask = (result['momentum_long'] < -self.min_momentum) | \
                        (result['momentum_short'] < 0)
            result.loc[sell_mask, 'signal'] = 'SELL'
            result.loc[sell_mask, 'score'] = 60 + (-result.loc[sell_mask, 'momentum_long'] * 100).clip(0, 40)

        return result[['date', 'open', 'high', 'low', 'close', 'volume',
                       'signal', 'score', 'momentum_long', 'momentum_short', 'rank']]

    def should_buy(self, momentum_long: float, momentum_short: float,
                   rank: float = None) -> Tuple[bool, float]:
        """判断是否应该买入"""
        if pd.isna(momentum_long) or pd.isna(momentum_short):
            return False, 50.0

        if rank is not None:
            if rank <= self.top_percentile:
                score = 70 + (self.top_percentile - rank) * 100
                return True, score
        else:
            if momentum_long > self.min_momentum and momentum_short > 0:
                score = 60 + min(momentum_long * 100, 40)
                return True, score

        return False, 50.0

    def should_sell(self, momentum_long: float, momentum_short: float,
                    rank: float = None) -> Tuple[bool, str]:
        """判断是否应该卖出"""
        if pd.isna(momentum_long) or pd.isna(momentum_short):
            return False, ''

        if rank is not None:
            if rank >= (1 - self.bottom_percentile):
                return True, 'BOTTOM_RANK'
        else:
            if momentum_long < -self.min_momentum or momentum_short < 0:
                return True, 'MOMENTUM_REVERSAL'

        return False, ''


class MomentumStrategyGenerator:
    """动量策略生成器"""

    @staticmethod
    def create_short_term() -> MomentumStrategy:
        """短期动量: 20日/5日"""
        return MomentumStrategy(long_period=20, short_period=5)

    @staticmethod
    def create_medium_term() -> MomentumStrategy:
        """中期动量: 60日/20日"""
        return MomentumStrategy(long_period=60, short_period=20)

    @staticmethod
    def create_long_term() -> MomentumStrategy:
        """长期动量: 120日/60日"""
        return MomentumStrategy(long_period=120, short_period=60)

    @staticmethod
    def create_cross_sectional() -> MomentumStrategy:
        """横截面动量"""
        return MomentumStrategy(
            long_period=60,
            short_period=20,
            top_percentile=0.2,
            bottom_percentile=0.3,
            min_momentum=0.0
        )


def generate_cross_sectional_signals(
    stock_data: Dict[str, pd.DataFrame],
    strategy: MomentumStrategy = None
) -> Dict[str, pd.DataFrame]:
    """
    为多只股票生成横截面动量策略信号
    所有股票在同一时间点排序,选择top N买入
    """
    if strategy is None:
        strategy = MomentumStrategyGenerator.create_cross_sectional()

    if len(stock_data) < 2:
        return generate_signals_for_stocks(stock_data, strategy)

    # 获取所有股票的日期对齐数据
    signals_dict = {}

    for code, df in stock_data.items():
        if df is None or len(df) < strategy.long_period + 5:
            continue
        try:
            # 计算动量
            df = df.copy()
            df['momentum_long'] = df['close'].pct_change(strategy.long_period)
            signals_dict[code] = df
        except Exception as e:
            print(f"Error processing {code}: {e}")
            continue

    # 对每个日期,计算横截面排名
    dates = None
    for code, df in signals_dict.items():
        dates = df['date'].tolist()
        break

    for date in dates:
        # 获取该日所有股票的动量
        momentums = {}
        for code, df in signals_dict.items():
            row = df[df['date'] == date]
            if len(row) > 0:
                momentums[code] = row['momentum_long'].values[0]

        if len(momentums) < 2:
            continue

        # 排序
        sorted_codes = sorted(momentums.keys(), key=lambda c: momentums[c], reverse=True)
        n = len(sorted_codes)
        top_n = max(1, int(n * strategy.top_percentile))

        for i, code in enumerate(sorted_codes):
            mask = signals_dict[code]['date'] == date
            if i < top_n:
                signals_dict[code].loc[mask, 'signal'] = 'BUY'
                signals_dict[code].loc[mask, 'score'] = 70 + (top_n - i) * 3
            elif i >= n - int(n * strategy.bottom_percentile):
                signals_dict[code].loc[mask, 'signal'] = 'SELL'
                signals_dict[code].loc[mask, 'score'] = 65
            else:
                signals_dict[code].loc[mask, 'signal'] = 'HOLD'
                signals_dict[code].loc[mask, 'score'] = 50

    return signals_dict


def generate_signals_for_stocks(
    stock_data: Dict[str, pd.DataFrame],
    strategy: MomentumStrategy = None
) -> Dict[str, pd.DataFrame]:
    """为多只股票生成动量策略信号"""
    if strategy is None:
        strategy = MomentumStrategyGenerator.create_medium_term()

    signals_dict = {}
    for code, df in stock_data.items():
        if df is None or len(df) < strategy.long_period + 5:
            continue
        try:
            signals = strategy.generate_signals(df, cross_sectional=False)
            signals_dict[code] = signals
        except Exception as e:
            print(f"Error generating signals for {code}: {e}")
            continue

    return signals_dict


def main():
    """测试动量策略"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')
    from src.data.data_pipeline import DataPipeline

    print("=" * 60)
    print("动量策略回测")
    print("=" * 60)

    pipeline = DataPipeline()
    codes = ['sh.600519', 'sh.600036', 'sh.601318', 'sh.600276']
    stock_data = {}

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 120:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    strategy = MomentumStrategyGenerator.create_medium_term()
    signals = generate_signals_for_stocks(stock_data, strategy)

    for code, df in signals.items():
        buy_count = (df['signal'] == 'BUY').sum()
        sell_count = (df['signal'] == 'SELL').sum()
        print(f"{code}: BUY={buy_count}, SELL={sell_count}, HOLD={len(df)-buy_count-sell_count}")

        recent = df[df['signal'] != 'HOLD'].tail(3)
        if len(recent) > 0:
            print(f"  最近信号:")
            for _, row in recent.iterrows():
                print(f"    {row['date']} {row['signal']} 60日动量:{row['momentum_long']*100:.1f}%")


if __name__ == '__main__':
    main()
