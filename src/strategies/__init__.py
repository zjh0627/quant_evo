#!/usr/bin/env python3
"""
策略模块
==========

可选策略:
- MLStrategy: 机器学习策略
- RSIStrategy: RSI反向策略
- MACrossoverStrategy: 均线交叉策略
- MeanReversionStrategy: 均值回归策略
- MomentumStrategy: 动量策略
- BreakoutStrategy: 突破策略
- MultiFactorStrategy: 多因子量化策略 (推荐)
"""

from .ml_strategy import MLStrategy, MLStrategyGenerator
from .rsi_strategy import RSIStrategy, RSIStrategyGenerator
from .ma_crossover_strategy import MACrossoverStrategy, MACrossoverStrategyGenerator
from .mean_reversion_strategy import MeanReversionStrategy, MeanReversionStrategyGenerator
from .momentum_strategy import MomentumStrategy, MomentumStrategyGenerator
from .breakout_strategy import BreakoutStrategy, BreakoutStrategyGenerator
from .multi_strategy import MultiStrategyCombiner, StrategyBacktester
from .multi_factor_strategy import MultiFactorStrategy, FactorCalculator, StockSignal

__all__ = [
    'MLStrategy', 'MLStrategyGenerator',
    'RSIStrategy', 'RSIStrategyGenerator',
    'MACrossoverStrategy', 'MACrossoverStrategyGenerator',
    'MeanReversionStrategy', 'MeanReversionStrategyGenerator',
    'MomentumStrategy', 'MomentumStrategyGenerator',
    'BreakoutStrategy', 'BreakoutStrategyGenerator',
    'MultiStrategyCombiner', 'StrategyBacktester',
    'MultiFactorStrategy', 'FactorCalculator', 'StockSignal'
]
