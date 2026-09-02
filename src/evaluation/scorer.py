#!/usr/bin/env python3
"""
评估框架
多维度评估策略表现
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EvaluationMetrics:
    """评估指标"""
    total_return: float      # 总收益率
    annual_return: float     # 年化收益率
    sharpe_ratio: float      # 夏普比率
    calmar_ratio: float       # 卡玛比率
    max_drawdown: float      # 最大回撤
    volatility: float        # 波动率
    win_rate: float           # 胜率
    profit_loss_ratio: float  # 盈亏比
    trade_count: int          # 交易次数

class StrategyEvaluator:
    """策略评估器"""

    @staticmethod
    def evaluate(backtest_result) -> EvaluationMetrics:
        """
        评估回测结果
        """
        if backtest_result is None:
            raise ValueError("回测结果为空")

        portfolio_values = backtest_result.portfolio_values
        trades = backtest_result.trades

        if not portfolio_values:
            raise ValueError("无组合价值数据")

        values = [p['value'] for p in portfolio_values]

        # 总收益率
        initial = backtest_result.initial_capital
        final = portfolio_values[-1]['value']
        total_return = (final - initial) / initial * 100

        # 年化收益率
        annual_return = backtest_result.annual_return

        # 最大回撤
        max_value = np.maximum.accumulate(values)
        drawdowns = (values - max_value) / max_value * 100
        max_drawdown = abs(np.min(drawdowns))

        # 波动率
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0

        # 夏普比率
        risk_free = 2.5
        if volatility > 0:
            sharpe_ratio = (annual_return - risk_free) / volatility
        else:
            sharpe_ratio = 0

        # 卡玛比率
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0

        # 胜率
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        winning = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        win_rate = winning / len(sell_trades) * 100 if sell_trades else 0

        # 盈亏比
        win_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        loss_trades = [t for t in sell_trades if t.get('pnl', 0) <= 0]

        avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in loss_trades])) if loss_trades else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        return EvaluationMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            trade_count=len(trades)
        )

    @staticmethod
    def calculate_composite_score(metrics: EvaluationMetrics) -> float:
        """
        计算综合评分 (0-100)
        """
        score = 0.0

        # 年化收益 (30分)
        score += min(metrics.annual_return / 20 * 30, 30)

        # 夏普比率 (25分)
        score += min(metrics.sharpe_ratio * 12.5, 25)

        # 最大回撤 (20分)
        if metrics.max_drawdown < 5:
            score += 20
        elif metrics.max_drawdown < 10:
            score += 15
        elif metrics.max_drawdown < 20:
            score += 10
        elif metrics.max_drawdown < 30:
            score += 5

        # 胜率 (15分)
        score += min(metrics.win_rate / 100 * 15, 15)

        # 交易次数 (10分) - 至少20次交易才算有效
        score += min(metrics.trade_count / 50 * 10, 10)

        return min(max(score, 0), 100)

    @staticmethod
    def meets_minimum_requirements(metrics: EvaluationMetrics) -> bool:
        """
        检查是否满足最低门槛
        """
        if metrics.calmar_ratio < 0.5:
            return False
        if metrics.max_drawdown > 30:
            return False
        if metrics.trade_count < 20:
            return False
        return True

    @staticmethod
    def print_evaluation(metrics: EvaluationMetrics):
        """打印评估结果"""
        score = StrategyEvaluator.calculate_composite_score(metrics)
        meets = StrategyEvaluator.meets_minimum_requirements(metrics)

        print(f"\n{'='*50}")
        print("策略评估")
        print(f"{'='*50}")
        print(f"总收益率:      {metrics.total_return:.2f}%")
        print(f"年化收益率:    {metrics.annual_return:.2f}%")
        print(f"夏普比率:      {metrics.sharpe_ratio:.3f}")
        print(f"卡玛比率:      {metrics.calmar_ratio:.3f}")
        print(f"最大回撤:      {metrics.max_drawdown:.2f}%")
        print(f"波动率:        {metrics.volatility:.2f}%")
        print(f"胜率:          {metrics.win_rate:.1f}%")
        print(f"盈亏比:        {metrics.profit_loss_ratio:.2f}")
        print(f"交易次数:      {metrics.trade_count}")
        print(f"{'='*50}")
        print(f"综合评分:      {score:.1f}")
        print(f"最低门槛:      {'通过 ✓' if meets else '未通过 ✗'}")
        print(f"{'='*50}")


if __name__ == '__main__':
    # 测试
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from backtest.engine import BacktestEngine
    from data.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    stock_data = {}

    for code in ['sh.600519', 'sh.600036', 'sh.601318']:
        df = pipeline.load_from_csv(code)
        if df is not None:
            stock_data[code] = df

    engine = BacktestEngine(initial_capital=1000000)
    result = engine.run(
        stock_data,
        start_date='2025-01-01',
        end_date='2025-06-30',
        rebalance_days=5,
        min_score=55
    )

    metrics = StrategyEvaluator.evaluate(result)
    StrategyEvaluator.print_evaluation(metrics)
