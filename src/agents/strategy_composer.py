#!/usr/bin/env python3
"""
Strategy Composer Agent - 策略组合Agent
将有效因子组合成可交易策略
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.agents.base import Agent, Message


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    entry_conditions: List[Dict]  # 入场条件
    exit_conditions: List[Dict]  # 出场条件
    position_rules: Dict          # 仓位规则
    risk_rules: Dict             # 风控规则


class StrategyComposerAgent(Agent):
    """策略组合Agent"""

    def __init__(self):
        super().__init__("StrategyComposer")
        self.strategies: List[StrategyConfig] = []
        self.current_strategy: Optional[StrategyConfig] = None

    def process(self, message: Optional[Message] = None) -> Optional[Message]:
        """处理消息"""
        if message and message.content.get('action') == 'compose':
            return self.compose_strategy(message.content)

        return None

    def compose_strategy(self, context: Dict[str, Any]) -> Optional[Message]:
        """组合策略"""
        factors = context.get('factors', [])
        current_params = context.get('current_params', {})

        print(f"[StrategyComposer] 组合策略, 因子数量: {len(factors)}")

        # 基于因子组合策略
        strategy = self._build_strategy_from_factors(factors, current_params)

        self.current_strategy = strategy
        self.strategies.append(strategy)

        self.set_state('current_strategy', self._strategy_to_dict(strategy))
        self.set_state('strategy_count', len(self.strategies))

        return self.send('Orchestrator', {
            'action': 'strategy_composed',
            'strategy': self._strategy_to_dict(strategy)
        }, 'RESPONSE')

    def _build_strategy_from_factors(self, factors: List[Dict],
                                     current_params: Dict) -> StrategyConfig:
        """基于因子构建策略"""
        # 入场条件：使用top因子
        entry_conditions = []

        for factor in factors[:3]:
            if factor.get('direction') == 'positive':
                entry_conditions.append({
                    'type': 'factor_threshold',
                    'factor': factor['name'],
                    'operator': '>',
                    'threshold': 0
                })
            else:
                entry_conditions.append({
                    'type': 'factor_threshold',
                    'factor': factor['name'],
                    'operator': '<',
                    'threshold': 0
                })

        # 动量确认
        entry_conditions.append({
            'type': 'momentum_confirm',
            'momentum_5d': '>',
            'threshold': -0.02
        })

        # RSI过滤
        entry_conditions.append({
            'type': 'rsi_filter',
            'rsi': '<',
            'threshold': 70
        })

        # 出场条件
        exit_conditions = [
            {'type': 'stop_loss', 'pct': current_params.get('stop_loss_pct', 0.08)},
            {'type': 'take_profit', 'pct': current_params.get('take_profit_pct', 0.15)},
            {'type': 'time_based', 'max_days': current_params.get('max_hold_days', 10)},
            {'type': 'trend_reverse', 'ma_bull': 0}
        ]

        # 仓位规则
        position_rules = {
            'max_position_pct': current_params.get('max_position_pct', 0.2),
            'max_total_pct': current_params.get('max_total_pct', 0.8),
            'max_positions': current_params.get('max_positions', 5),
            'rebalance_days': current_params.get('rebalance_days', 5)
        }

        # 风控规则
        risk_rules = {
            'max_drawdown_pct': 0.15,
            'max_single_loss_pct': 0.10,
            'volatility_target': 0.15,
            'correlation_limit': 0.7
        }

        strategy = StrategyConfig(
            name=f"Strategy_{len(self.strategies) + 1}",
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_rules=position_rules,
            risk_rules=risk_rules
        )

        return strategy

    def _strategy_to_dict(self, strategy: StrategyConfig) -> Dict:
        """策略转字典"""
        return {
            'name': strategy.name,
            'entry_conditions': strategy.entry_conditions,
            'exit_conditions': strategy.exit_conditions,
            'position_rules': strategy.position_rules,
            'risk_rules': strategy.risk_rules
        }

    def generate_param_suggestions(self, current_params: Dict,
                                   backtest_result: Dict) -> Dict[str, Any]:
        """
        基于回测结果生成参数建议
        这是简化的规则版本，真正的LLM版本会分析原因
        """
        suggestions = {}
        annual_return = backtest_result.get('annual_return', 0)
        max_drawdown = backtest_result.get('max_drawdown', 0)
        sharpe = backtest_result.get('sharpe_ratio', 0)
        win_rate = backtest_result.get('win_rate', 0)

        # 收益太低 -> 放宽入场条件
        if annual_return < 10:
            if current_params.get('min_score', 50) > 45:
                suggestions['min_score'] = max(40, current_params['min_score'] - 5)

        # 回撤太大 -> 收紧止损
        if abs(max_drawdown) > 10:
            if current_params.get('stop_loss_pct', 0.05) < 0.10:
                suggestions['stop_loss_pct'] = current_params.get('stop_loss_pct', 0.05) * 0.8

        # 夏普太低 -> 减少交易频率
        if sharpe < 1.0:
            if current_params.get('rebalance_days', 5) < 10:
                suggestions['rebalance_days'] = min(15, current_params['rebalance_days'] + 3)

        # 胜率太低 -> 调整选股条件
        if win_rate < 40:
            if current_params.get('min_score', 50) < 60:
                suggestions['min_score'] = current_params['min_score'] + 5

        return suggestions

    def get_current_strategy(self) -> Optional[Dict]:
        """获取当前策略"""
        return self.get_state('current_strategy')


if __name__ == '__main__':
    # 测试
    agent = StrategyComposerAgent()

    factors = [
        {'name': 'momentum_5d', 'direction': 'positive', 'avg_ic': 0.05},
        {'name': 'rsi_14', 'direction': 'negative', 'avg_ic': -0.03},
        {'name': 'ma_bull_alignment', 'direction': 'positive', 'avg_ic': 0.04},
    ]

    params = {
        'min_score': 55,
        'rebalance_days': 5,
        'max_positions': 5,
        'stop_loss_pct': 0.08,
        'take_profit_pct': 0.15,
    }

    msg = agent.compose_strategy({'factors': factors, 'current_params': params})

    if msg:
        print("\n生成的策略:")
        print(json.dumps(msg.content.get('strategy'), indent=2, ensure_ascii=False))
