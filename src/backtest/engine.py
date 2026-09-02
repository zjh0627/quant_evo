#!/usr/bin/env python3
"""
回测引擎
基于因子信号进行回测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.technical import FactorCalculator

@dataclass
class BacktestResult:
    """回测结果"""
    initial_capital: float
    final_value: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    trades: List[Dict]
    portfolio_values: List[Dict]


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float = 1000000, commission: float = 0.0003):
        """
        Args:
            initial_capital: 初始资金
            commission: 交易佣金率 (默认万3)
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, stock_data: Dict[str, pd.DataFrame],
            start_date: str, end_date: str,
            rebalance_days: int = 5,
            min_score: float = 60) -> Optional[BacktestResult]:
        """
        运行回测

        Args:
            stock_data: {code: DataFrame} 股票数据
            start_date: 回测开始日期
            end_date: 回测结束日期
            rebalance_days: 调仓周期
            min_score: 最小入场评分

        Returns:
            BacktestResult
        """
        if not stock_data:
            return None

        # 合并所有日期
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())

        dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        if len(dates) < 20:
            return None

        # 初始化
        capital = self.initial_capital
        positions = {}  # {code: {'shares': int, 'cost': float, 'entry_date': str}}
        trades = []
        portfolio_values = []
        factor_calc = FactorCalculator()

        last_rebalance_idx = -1

        for i, date_str in enumerate(dates):
            # 计算当前组合价值
            daily_prices = {}
            for code, df in stock_data.items():
                row = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
                if not row.empty:
                    daily_prices[code] = row['close'].iloc[0]

            portfolio_value = capital
            for code, pos in positions.items():
                if code in daily_prices:
                    portfolio_value += pos['shares'] * daily_prices[code]

            portfolio_values.append({
                'date': date_str,
                'value': portfolio_value,
                'capital': capital,
                'positions_value': portfolio_value - capital,
                'positions': len(positions)
            })

            # 检查是否需要调仓
            should_rebalance = False
            if last_rebalance_idx < 0:
                should_rebalance = True
            elif i - last_rebalance_idx >= rebalance_days:
                should_rebalance = True

            if not should_rebalance:
                continue

            # === 调仓逻辑 ===
            buy_signals = []
            sell_signals = []

            # 检查持仓
            for code in list(positions.keys()):
                if code not in stock_data:
                    sell_signals.append(code)
                    continue

                df = stock_data[code]
                hist_df = df[df['date'] <= date_str].tail(60)

                if len(hist_df) >= 20:
                    factors = factor_calc.calculate_all_factors(hist_df)
                    signal = factor_calc.generate_signal(factors)

                    if signal == 'SELL':
                        sell_signals.append(code)

            # 执行卖出
            for code in sell_signals:
                if code in positions and code in daily_prices:
                    price = daily_prices[code]
                    shares = positions[code]['shares']
                    cost = positions[code]['cost']

                    revenue = shares * price * (1 - self.commission)
                    pnl = (price - cost) * shares
                    capital += revenue

                    trades.append({
                        'date': date_str,
                        'action': 'SELL',
                        'code': code,
                        'price': price,
                        'shares': shares,
                        'cost': cost,
                        'pnl': pnl
                    })

                    del positions[code]

            # 选股买入
            candidates = []
            for code, df in stock_data.items():
                if code in positions:
                    continue

                hist_df = df[df['date'] <= date_str].tail(60)
                if len(hist_df) < 20:
                    continue

                factors = factor_calc.calculate_all_factors(hist_df)
                score = factor_calc.get_composite_score(factors)
                signal = factor_calc.generate_signal(factors)

                if signal == 'BUY' and score >= min_score:
                    price = daily_prices.get(code)
                    if price is not None:
                        candidates.append({
                            'code': code,
                            'score': score,
                            'price': price
                        })

            # 按评分排序
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # 分配资金买入
            max_positions = 5
            per_position = capital / max_positions if positions else capital / min(max_positions, len(candidates) + 1)

            for candidate in candidates:
                if len(positions) >= max_positions:
                    break

                code = candidate['code']
                price = candidate['price']

                if price <= 0:
                    continue

                shares = int(per_position / price / 100) * 100  # 按手买

                if shares < 100:
                    continue

                cost = shares * price * (1 + self.commission)

                if cost > capital:
                    continue

                positions[code] = {
                    'shares': shares,
                    'cost': price,
                    'entry_date': date_str
                }
                capital -= cost

                trades.append({
                    'date': date_str,
                    'action': 'BUY',
                    'code': code,
                    'price': price,
                    'shares': shares,
                    'cost': price,
                    'pnl': 0
                })

            last_rebalance_idx = i

        # === 回测结束，强制平仓 ===
        final_prices = {}
        last_date = dates[-1]
        for code in list(positions.keys()):
            for df in stock_data.values():
                row = df[df['date'].dt.strftime('%Y-%m-%d') == last_date]
                if not row.empty:
                    final_prices[code] = row['close'].iloc[0]
                    break

        for code, pos in list(positions.items()):
            if code in final_prices:
                price = final_prices[code]
                revenue = pos['shares'] * price * (1 - self.commission)
                pnl = (price - pos['cost']) * pos['shares']
                capital += revenue

                trades.append({
                    'date': last_date,
                    'action': 'SELL',
                    'code': code,
                    'price': price,
                    'shares': pos['shares'],
                    'cost': pos['cost'],
                    'pnl': pnl
                })

                del positions[code]

        # 计算结果
        if not portfolio_values:
            return None

        final_value = portfolio_values[-1]['value']
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # 年化收益
        days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        annual_return = total_return / days * 365 if days > 0 else 0

        # 最大回撤
        values = [p['value'] for p in portfolio_values]
        max_value = np.maximum.accumulate(values)
        drawdowns = (values - max_value) / max_value * 100
        max_drawdown = np.min(drawdowns)

        # 夏普比率
        risk_free_rate = 2.5
        returns = np.diff(values) / values[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            excess_return = annual_return - risk_free_rate
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = excess_return / (volatility * 100) if volatility > 0 else 0
        else:
            sharpe_ratio = 0

        # 胜率
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        winning_trades = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        win_rate = winning_trades / len(sell_trades) * 100 if sell_trades else 0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=winning_trades,
            trades=trades,
            portfolio_values=portfolio_values
        )

    def print_result(self, result: BacktestResult):
        """打印回测结果"""
        if result is None:
            print("回测失败")
            return

        print(f"\n{'='*50}")
        print("回测结果")
        print(f"{'='*50}")
        print(f"初始资金:    {result.initial_capital:,.0f}")
        print(f"最终价值:    {result.final_value:,.0f}")
        print(f"总收益率:    {result.total_return:.2f}%")
        print(f"年化收益率:  {result.annual_return:.2f}%")
        print(f"最大回撤:    {result.max_drawdown:.2f}%")
        print(f"夏普比率:    {result.sharpe_ratio:.3f}")
        print(f"胜率:        {result.win_rate:.1f}%")
        print(f"总交易次数:  {result.total_trades}")
        print(f"盈利交易:    {result.winning_trades}")


if __name__ == '__main__':
    # 测试
    from data.data_pipeline import DataPipeline

    pipeline = DataPipeline()

    # 加载数据
    stock_data = {}
    codes = ['sh.600519', 'sh.600036', 'sh.601318']
    for code in codes:
        df = pipeline.load_from_csv(code)
        if df is not None:
            stock_data[code] = df

    if stock_data:
        engine = BacktestEngine(initial_capital=1000000)
        result = engine.run(
            stock_data,
            start_date='2025-01-01',
            end_date='2025-06-30',
            rebalance_days=5,
            min_score=55
        )

        engine.print_result(result)
