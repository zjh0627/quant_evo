#!/usr/bin/env python3
"""
Simulation Agent - 模拟Agent
运行回测、参数调优
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.agents.base import Agent, Message
from src.backtest.engine import BacktestEngine
from src.evaluation.scorer import StrategyEvaluator


class SimulationAgent(Agent):
    """模拟Agent"""

    def __init__(self):
        super().__init__("Simulation")
        self.backtester = BacktestEngine(initial_capital=1000000)
        self.evaluator = StrategyEvaluator()
        self.last_result = None

    def process(self, message: Optional[Message] = None) -> Optional[Message]:
        """处理消息"""
        if message:
            action = message.content.get('action')

            if action == 'backtest':
                return self.run_backtest(message.content)
            elif action == 'optimize':
                return self.optimize_params(message.content)

        return None

    def run_backtest(self, context: Dict[str, Any]) -> Optional[Message]:
        """运行回测"""
        stock_data = context.get('stock_data', {})
        params = context.get('params', {})
        start_date = context.get('start_date', '2025-01-01')
        end_date = context.get('end_date', '2025-06-30')

        print(f"[Simulation] 运行回测, 参数: {params}")

        if not stock_data:
            print("[Simulation] 无股票数据")
            return None

        try:
            # 运行回测
            result = self.backtester.run(
                stock_data,
                start_date=start_date,
                end_date=end_date,
                rebalance_days=params.get('rebalance_days', 5),
                min_score=params.get('min_score', 55)
            )

            if result is None:
                print("[Simulation] 回测失败")
                return None

            # 评估
            metrics = self.evaluator.evaluate(result)
            score = self.evaluator.calculate_composite_score(metrics)

            self.last_result = {
                'params': params,
                'metrics': {
                    'total_return': metrics.total_return,
                    'annual_return': metrics.annual_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'calmar_ratio': metrics.calmar_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'trade_count': metrics.trade_count
                },
                'score': score
            }

            self.set_state('last_backtest', self.last_result)

            print(f"[Simulation] 回测完成, 评分: {score:.1f}")

            return self.send('Orchestrator', {
                'action': 'backtest_complete',
                'params': params,
                'metrics': self.last_result['metrics'],
                'score': score
            }, 'RESPONSE')

        except Exception as e:
            print(f"[Simulation] 回测异常: {e}")
            return None

    def optimize_params(self, context: Dict[str, Any]) -> Optional[Message]:
        """优化参数"""
        stock_data = context.get('stock_data', {})
        base_params = context.get('base_params', {})
        target_metric = context.get('target', 'sharpe_ratio')
        n_iterations = context.get('iterations', 10)

        print(f"[Simulation] 优化参数, 目标: {target_metric}, 迭代: {n_iterations}")

        best_score = 0
        best_params = base_params.copy()
        all_results = []

        # 网格搜索
        param_ranges = {
            'min_score': [45, 50, 55, 60, 65],
            'rebalance_days': [3, 5, 7, 10],
            'max_positions': [3, 5, 7]
        }

        import itertools
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())

        for combo in itertools.product(*values):
            params = base_params.copy()
            for i, key in enumerate(keys):
                params[key] = combo[i]

            result = self.backtester.run(
                stock_data,
                start_date='2025-01-01',
                end_date='2025-06-30',
                rebalance_days=params.get('rebalance_days', 5),
                min_score=params.get('min_score', 55)
            )

            if result:
                metrics = self.evaluator.evaluate(result)
                score = self.evaluator.calculate_composite_score(metrics)

                all_results.append({
                    'params': params.copy(),
                    'score': score,
                    'metrics': {
                        'annual_return': metrics.annual_return,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown
                    }
                })

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

        # 排序结果
        all_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"[Simulation] 优化完成, 最佳评分: {best_score:.1f}")

        self.set_state('optimization_result', {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results[:10]  # 保存top10
        })

        return self.send('Orchestrator', {
            'action': 'optimization_complete',
            'best_params': best_params,
            'best_score': best_score,
            'top_results': all_results[:5]
        }, 'RESPONSE')

    def get_last_result(self) -> Optional[Dict]:
        """获取上次回测结果"""
        return self.last_result

    def run_walk_forward(self, stock_data: Dict[str, pd.DataFrame],
                        params: Dict,
                        train_period: int = 180,
                        test_period: int = 60) -> List[Dict]:
        """
        Walk-forward 分析
        滚动窗口验证策略稳定性
        """
        results = []

        # 获取所有日期
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())
        dates = sorted(all_dates)

        start_idx = train_period
        while start_idx + test_period <= len(dates):
            train_end = dates[start_idx - train_period]
            test_end = dates[start_idx]
            test_start = dates[start_idx]
            test_end_idx = min(start_idx + test_period, len(dates) - 1)
            test_end = dates[test_end_idx]

            # 训练
            train_result = self.backtester.run(
                stock_data,
                start_date=train_end,
                end_date=train_end,
                rebalance_days=params.get('rebalance_days', 5),
                min_score=params.get('min_score', 55)
            )

            # 测试
            test_result = self.backtester.run(
                stock_data,
                start_date=test_start,
                end_date=test_end,
                rebalance_days=params.get('rebalance_days', 5),
                min_score=params.get('min_score', 55)
            )

            if test_result:
                results.append({
                    'period': f"{test_start}~{test_end}",
                    'train_score': self.evaluator.calculate_composite_score(
                        self.evaluator.evaluate(train_result)) if train_result else 0,
                    'test_score': self.evaluator.calculate_composite_score(
                        self.evaluator.evaluate(test_result))
                })

            start_idx += test_period

        return results


if __name__ == '__main__':
    # 测试
    from data.data_pipeline import DataPipeline

    agent = SimulationAgent()

    # 加载数据
    pipeline = DataPipeline()
    stock_data = {}

    for code in ['sh.600519', 'sh.600036', 'sh.688012']:
        df = pipeline.load_from_csv(code)
        if df is not None:
            stock_data[code] = df

    if stock_data:
        params = {
            'min_score': 55,
            'rebalance_days': 5,
            'max_positions': 5
        }

        msg = agent.run_backtest({
            'stock_data': stock_data,
            'params': params,
            'start_date': '2025-01-01',
            'end_date': '2025-06-30'
        })

        if msg:
            print("\n回测结果:")
            for k, v in msg.content['metrics'].items():
                print(f"  {k}: {v}")
            print(f"  评分: {msg.content['score']:.1f}")
