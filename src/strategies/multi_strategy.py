#!/usr/bin/env python3
"""
多策略组合器
============

将多个策略的信号进行组合:
1. 加权平均
2. 投票机制
3. 优先级筛选
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    weight: float = 1.0
    min_score: float = 60.0
    enabled: bool = True


class MultiStrategyCombiner:
    """多策略组合器"""

    def __init__(self, strategies: List[StrategyConfig] = None):
        """
        Args:
            strategies: 策略配置列表
        """
        self.strategies = strategies or [
            StrategyConfig(name='rsi', weight=1.0, min_score=60),
            StrategyConfig(name='ma_cross', weight=1.0, min_score=60),
            StrategyConfig(name='momentum', weight=1.0, min_score=60),
            StrategyConfig(name='mean_reversion', weight=1.0, min_score=60),
            StrategyConfig(name='breakout', weight=1.0, min_score=60),
        ]

    def combine_by_weighted_average(
        self,
        signals_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        加权平均组合

        Args:
            signals_dict: {策略名: 信号df} 各策略的信号

        Returns:
            组合后的信号
        """
        combined = {}

        # 获取所有股票代码
        all_codes = set()
        for strategy_name, df_dict in signals_dict.items():
            all_codes.update(df_dict.keys())

        for code in all_codes:
            # 收集各策略的分数
            scores = []
            weights = []
            dates = None

            for strategy_name, df_dict in signals_dict.items():
                if code not in df_dict:
                    continue

                df = df_dict[code]
                if dates is None:
                    dates = df['date'].tolist()

                # 检查信号是否有效
                if 'score' in df.columns:
                    valid = df['signal'] != 'HOLD'
                    scores.extend(df.loc[valid, 'score'].tolist())
                    weights.extend([self.get_strategy_weight(strategy_name)] * valid.sum())

            if not scores:
                continue

            # 加权平均
            result = pd.DataFrame({'date': dates})
            result['combined_score'] = np.average(scores, weights=weights) if weights else 50

            # 信号判定
            result['signal'] = 'HOLD'
            result.loc[result['combined_score'] > 65, 'signal'] = 'BUY'
            result.loc[result['combined_score'] < 40, 'signal'] = 'SELL'

            combined[code] = result

        return combined

    def combine_by_voting(
        self,
        signals_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        投票机制组合

        各策略投票,多数票决定信号
        """
        combined = {}

        all_codes = set()
        for strategy_name, df_dict in signals_dict.items():
            all_codes.update(df_dict.keys())

        for code in all_codes:
            votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            score_sum = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            dates = None

            for strategy_name, df_dict in signals_dict.items():
                if code not in df_dict:
                    continue

                df = df_dict[code]
                if dates is None:
                    dates = df['date'].tolist()

                for _, row in df.iterrows():
                    signal = row.get('signal', 'HOLD')
                    score = row.get('score', 50)
                    weight = self.get_strategy_weight(strategy_name)

                    votes[signal] += weight
                    score_sum[signal] += score * weight

            if not dates:
                continue

            result = pd.DataFrame({'date': dates})

            # 多数票决定
            result['signal'] = 'HOLD'
            for signal in ['BUY', 'SELL']:
                result.loc[votes[signal] > votes['HOLD'], 'signal'] = signal

            # 计算加权分数
            total_weight = sum(votes.values())
            result['combined_score'] = 50
            for signal in ['BUY', 'SELL']:
                if votes[signal] > 0:
                    result.loc[result['signal'] == signal, 'combined_score'] = \
                        score_sum[signal] / votes[signal]

            combined[code] = result

        return combined

    def combine_by_priority(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        priority_order: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        优先级组合

        按优先级顺序,第一个有效信号为准
        """
        if priority_order is None:
            priority_order = ['breakout', 'momentum', 'ma_cross', 'rsi', 'mean_reversion']

        combined = {}

        all_codes = set()
        for strategy_name, df_dict in signals_dict.items():
            all_codes.update(df_dict.keys())

        for code in all_codes:
            dates = None
            primary_signals = {}

            # 收集所有信号
            for strategy_name, df_dict in signals_dict.items():
                if code not in df_dict:
                    continue

                df = df_dict[code]
                if dates is None:
                    dates = df['date'].tolist()

                for _, row in df.iterrows():
                    date = row['date']
                    if date not in primary_signals:
                        primary_signals[date] = {
                            'signal': row.get('signal', 'HOLD'),
                            'score': row.get('score', 50),
                            'strategy': strategy_name
                        }

            if not dates:
                continue

            result = pd.DataFrame({'date': dates})

            # 按优先级选择信号
            signals = []
            scores = []
            for date in dates:
                signal_info = primary_signals.get(date, {'signal': 'HOLD', 'score': 50, 'strategy': ''})

                # 按优先级查找有效信号
                chosen = signal_info
                for pri in priority_order:
                    if signal_info['strategy'] == pri and signal_info['signal'] != 'HOLD':
                        chosen = signal_info
                        break

                signals.append(chosen['signal'])
                scores.append(chosen['score'])

            result['signal'] = signals
            result['combined_score'] = scores

            combined[code] = result

        return combined

    def get_strategy_weight(self, strategy_name: str) -> float:
        """获取策略权重"""
        for s in self.strategies:
            if s.name == strategy_name:
                return s.weight if s.enabled else 0
        return 1.0


class StrategyBacktester:
    """策略回测器"""

    def __init__(self, initial_capital: float = 1000000, commission: float = 0.0003):
        self.initial_capital = initial_capital
        self.commission = commission

    def backtest_single_strategy(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        stock_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        回测单个策略

        Returns:
            回测结果字典
        """
        trades = []
        positions = {}
        cash = self.initial_capital
        equity_curve = []

        # 获取所有日期
        all_dates = set()
        for code, df in stock_data.items():
            all_dates.update(df['date'].tolist())
        all_dates = sorted(list(all_dates))

        for date in all_dates:
            daily_value = cash

            # 检查卖出信号
            for code, pos in list(positions.items()):
                if code not in stock_data:
                    continue

                df = stock_data[code]
                row = df[df['date'] == date]
                if len(row) == 0:
                    continue

                price = row['close'].values[0]

                # 检查信号
                if code in signals_dict:
                    sig_df = signals_dict[code]
                    sig_row = sig_df[sig_df['date'] == date]
                    if len(sig_row) > 0:
                        signal = sig_row['signal'].values[0]
                        if signal == 'SELL':
                            # 卖出
                            proceeds = pos['shares'] * price * (1 - self.commission)
                            pnl = proceeds - pos['cost']
                            trades.append({
                                'date': date,
                                'action': 'SELL',
                                'code': code,
                                'price': price,
                                'shares': pos['shares'],
                                'pnl': pnl
                            })
                            cash += proceeds
                            del positions[code]

                daily_value += sum(p['shares'] * price for p in positions.values())

            # 检查买入信号
            for code, df in stock_data.items():
                if code in positions:
                    continue

                sig_df = signals_dict.get(code)
                if sig_df is None:
                    continue

                sig_row = sig_df[sig_df['date'] == date]
                if len(sig_row) == 0:
                    continue

                signal = sig_row['signal'].values[0]
                if signal == 'BUY':
                    price = df[df['date'] == date]['close'].values[0]
                    shares = int(cash * 0.1 / price / 100) * 100  # 10%仓位

                    if shares > 0:
                        cost = shares * price * (1 + self.commission)
                        if cost <= cash:
                            positions[code] = {
                                'shares': shares,
                                'entry_price': price,
                                'cost': cost,
                                'entry_date': date
                            }
                            cash -= cost
                            trades.append({
                                'date': date,
                                'action': 'BUY',
                                'code': code,
                                'price': price,
                                'shares': shares
                            })

            equity_curve.append({
                'date': date,
                'value': daily_value
            })

        # 计算统计
        equity_df = pd.DataFrame(equity_curve)
        equity_df['equity'] = equity_df['value'].cummax()

        total_return = (equity_df['value'].iloc[-1] - self.initial_capital) / self.initial_capital
        max_drawdown = ((equity_df['equity'] - equity_df['value']) / equity_df['equity']).max()

        winning_trades = [t for t in trades if t['action'] == 'SELL' and t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t['action'] == 'SELL' and t.get('pnl', 0) <= 0]

        return {
            'total_return': total_return,
            'final_value': equity_df['value'].iloc[-1],
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'trades': trades,
            'equity_curve': equity_df
        }


def main():
    """测试多策略组合"""
    from src.strategies.rsi_strategy import generate_signals_for_stocks as rsi_signals, RSIStrategyGenerator
    from src.strategies.momentum_strategy import generate_signals_for_stocks as momentum_signals, MomentumStrategyGenerator
    from src.data.data_pipeline import DataPipeline

    print("=" * 60)
    print("多策略组合回测")
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

    # 生成各策略信号
    rsi_signals_dict = {code: df for code, df in
                        rsi_signals(stock_data, RSIStrategyGenerator.create_balanced()).items()}
    momentum_signals_dict = {code: df for code, df in
                            momentum_signals(stock_data, MomentumStrategyGenerator.create_medium_term()).items()}

    # 组合
    combiner = MultiStrategyCombiner()
    combined = combiner.combine_by_voting({
        'rsi': rsi_signals_dict,
        'momentum': momentum_signals_dict
    })

    print(f"组合后信号统计:")
    for code, df in combined.items():
        buy_count = (df['signal'] == 'BUY').sum()
        sell_count = (df['signal'] == 'SELL').sum()
        print(f"  {code}: BUY={buy_count}, SELL={sell_count}")


if __name__ == '__main__':
    main()
