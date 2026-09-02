#!/usr/bin/env python3
"""
Risk Control Agent - 风控Agent
管理仓位、止损止盈、回撤控制
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.agents.base import Agent, Message


@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_return: float
    volatility: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    leverage: float


class RiskControlAgent(Agent):
    """风控Agent"""

    def __init__(self):
        super().__init__("RiskControl")
        self.max_drawdown_limit = 0.15  # 15%最大回撤
        self.max_position_pct = 0.20     # 单只最大仓位20%
        self.max_total_pct = 0.80        # 总仓位最大80%
        self.stop_loss_pct = 0.08        # 止损8%
        self.take_profit_pct = 0.15      # 止盈15%

    def process(self, message: Optional[Message] = None) -> Optional[Message]:
        """处理消息"""
        if message and message.content.get('action') == 'evaluate':
            return self.evaluate_risk(message.content)

        return None

    def evaluate_risk(self, context: Dict[str, Any]) -> Optional[Message]:
        """评估风险"""
        portfolio_value = context.get('portfolio_value', 1000000)
        positions = context.get('positions', [])
        cash = context.get('cash', portfolio_value)
        price_changes = context.get('price_changes', {})

        print(f"[RiskControl] 评估风险, 组合价值: {portfolio_value:,.0f}")

        # 计算风险指标
        metrics = self._calculate_risk_metrics(
            portfolio_value, cash, positions, price_changes
        )

        # 生成风控信号
        signals = self._generate_risk_signals(metrics, positions, price_changes)

        self.set_state('current_risk', {
            'max_drawdown': metrics.max_drawdown,
            'volatility': metrics.volatility,
            'leverage': metrics.leverage
        })

        return self.send('Orchestrator', {
            'action': 'risk_evaluated',
            'metrics': {
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'var_95': metrics.var_95,
                'leverage': metrics.leverage
            },
            'signals': signals
        }, 'RESPONSE')

    def _calculate_risk_metrics(self, portfolio_value: float, cash: float,
                                 positions: List[Dict], price_changes: Dict) -> RiskMetrics:
        """计算风险指标"""
        positions_value = portfolio_value - cash

        # 计算当日盈亏
        daily_pnl = 0
        for pos in positions:
            code = pos.get('code')
            if code in price_changes:
                change = price_changes[code]
                pos_value = pos.get('shares', 0) * pos.get('current_price', 0)
                daily_pnl += pos_value * change

        daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0

        # 计算波动率（简化版）
        volatility = 0.02  # 默认2%日波动

        # 计算最大回撤（简化）
        max_drawdown = abs(self.get_state('current_risk', {}).get('max_drawdown', 0))

        # 计算VaR (95%)
        var_95 = portfolio_value * volatility * 1.65

        # 计算杠杆
        leverage = positions_value / portfolio_value if portfolio_value > 0 else 0

        return RiskMetrics(
            portfolio_value=portfolio_value,
            cash=cash,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            leverage=leverage
        )

    def _generate_risk_signals(self, metrics: RiskMetrics,
                                positions: List[Dict],
                                price_changes: Dict) -> List[Dict]:
        """生成风控信号"""
        signals = []

        # 1. 回撤超限 -> 减仓
        if metrics.max_drawdown > self.max_drawdown_limit:
            signals.append({
                'type': 'reduce_position',
                'reason': 'max_drawdown_exceeded',
                'action': 'sell_all',
                'priority': 'high'
            })

        # 2. 波动率过高 -> 降低仓位
        if metrics.volatility > 0.03:  # 3%日波动
            signals.append({
                'type': 'reduce_position',
                'reason': 'high_volatility',
                'action': 'reduce_50pct',
                'priority': 'medium'
            })

        # 3. 检查个股止损
        for pos in positions:
            code = pos.get('code')
            if code in price_changes:
                change = price_changes[code]

                # 止损信号
                if change < -self.stop_loss_pct:
                    signals.append({
                        'type': 'stop_loss',
                        'code': code,
                        'reason': f'price_down_{abs(change)*100:.1f}%',
                        'action': 'sell',
                        'priority': 'high'
                    })

                # 止盈信号
                elif change > self.take_profit_pct:
                    signals.append({
                        'type': 'take_profit',
                        'code': code,
                        'reason': f'price_up_{abs(change)*100:.1f}%',
                        'action': 'sell_partial',
                        'priority': 'medium'
                    })

        # 4. 杠杆过高 -> 降低杠杆
        if metrics.leverage > self.max_total_pct:
            signals.append({
                'type': 'reduce_leverage',
                'reason': 'leverage_exceeded',
                'action': 'reduce_to_80pct',
                'priority': 'high'
            })

        return signals

    def calculate_position_size(self, stock_price: float, stop_loss_pct: float,
                                 account_value: float, risk_pct: float = 0.02) -> int:
        """
        计算仓位大小
        基于风险百分比模型
        """
        # 每笔交易风险金额
        risk_amount = account_value * risk_pct

        # 止损金额
        stop_amount = stock_price * stop_loss_pct

        # 股票数量
        shares = int(risk_amount / stop_amount / 100) * 100

        return max(100, shares)

    def check_position_limits(self, current_positions: int,
                              new_position_value: float,
                              total_value: float) -> bool:
        """检查仓位限制"""
        # 单只仓位限制
        max_single = total_value * self.max_position_pct
        if new_position_value > max_single:
            return False

        # 总仓位限制
        current_positions_value = total_value - self.get_state('cash', total_value * 0.2)
        if (current_positions_value + new_position_value) > total_value * self.max_total_pct:
            return False

        return True

    def update_limits(self, **kwargs):
        """更新风控参数"""
        if 'max_drawdown_limit' in kwargs:
            self.max_drawdown_limit = kwargs['max_drawdown_limit']
        if 'max_position_pct' in kwargs:
            self.max_position_pct = kwargs['max_position_pct']
        if 'stop_loss_pct' in kwargs:
            self.stop_loss_pct = kwargs['stop_loss_pct']
        if 'take_profit_pct' in kwargs:
            self.take_profit_pct = kwargs['take_profit_pct']


if __name__ == '__main__':
    # 测试
    agent = RiskControlAgent()

    positions = [
        {'code': 'sh.600519', 'shares': 100, 'entry_price': 1400, 'current_price': 1420},
        {'code': 'sh.600036', 'shares': 1000, 'entry_price': 35, 'current_price': 34},
    ]

    price_changes = {
        'sh.600519': 0.014,   # 涨1.4%
        'sh.600036': -0.028,  # 跌2.8%
    }

    msg = agent.evaluate_risk({
        'portfolio_value': 1050000,
        'cash': 300000,
        'positions': positions,
        'price_changes': price_changes
    })

    if msg:
        print("\n风控评估:")
        print(f"  最大回撤: {msg.content['metrics']['max_drawdown']:.2%}")
        print(f"  波动率: {msg.content['metrics']['volatility']:.2%}")
        print(f"  杠杆: {msg.content['metrics']['leverage']:.2%}")
        print(f"\n风控信号: {len(msg.content['signals'])}")
        for sig in msg.content['signals']:
            print(f"  - {sig}")
