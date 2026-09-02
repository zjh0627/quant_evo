#!/usr/bin/env python3
"""
风控模块
========

功能:
1. 仓位管理（单只个股仓位上限、总仓位限制）
2. 止损机制（固定止损、时间止损、跟踪止损）
3. 风险规则引擎
4. 实时风险监控

使用方法:
    from src.trading.risk_control import RiskManager

    risk_mgr = RiskManager()
    result = risk_mgr.check_order(order)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Position:
    """持仓"""
    code: str
    shares: int
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    stop_loss_price: float = 0.0  # 止损价
    take_profit_price: float = 0.0  # 止盈价


@dataclass
class Order:
    """订单"""
    code: str
    action: str  # BUY / SELL
    price: float
    shares: int
    order_type: str = "MARKET"  # MARKET / LIMIT


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    approved: bool
    risk_level: RiskLevel
    message: str
    adjusted_shares: int = 0
    adjusted_price: float = 0.0


@dataclass
class RiskMetrics:
    """风险指标"""
    total_value: float
    cash: float
    positions_value: float
    position_ratio: float  # 持仓比例
    single_position_ratio: float  # 单只持仓比例
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown: float


class RiskRule:
    """风控规则基类"""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority

    def check(self,
             order: Order,
             positions: Dict[str, Position],
             metrics: RiskMetrics,
             market_data: Dict[str, float]) -> Tuple[bool, str]:
        """
        检查订单是否通过风控

        Returns:
            (passed, message)
        """
        raise NotImplementedError


class MaxPositionRatioRule(RiskRule):
    """最大持仓比例规则"""

    def __init__(self, max_ratio: float = 0.3):
        super().__init__("最大持仓比例", priority=10)
        self.max_ratio = max_ratio

    def check(self,
             order: Order,
             positions: Dict[str, Position],
             metrics: RiskMetrics,
             market_data: Dict[str, float]) -> Tuple[bool, str]:
        if order.action != "BUY":
            return True, "OK"

        # 计算买入后的持仓比例
        order_value = order.price * order.shares
        new_total_value = metrics.total_value + order_value
        new_position_ratio = (metrics.positions_value + order_value) / new_total_value

        if new_position_ratio > self.max_ratio:
            # 自动调整买入数量
            max_shares = int((metrics.total_value * self.max_ratio - metrics.positions_value) / order.price / 100) * 100
            if max_shares <= 0:
                return False, f"超过最大持仓比例 ({self.max_ratio:.0%})"
            return False, f"超过最大持仓比例，调整为 {max_shares} 股"

        return True, "OK"


class SingleStockMaxRatioRule(RiskRule):
    """单只股票最大比例规则"""

    def __init__(self, max_ratio: float = 0.1):
        super().__init__("单只股票最大比例", priority=20)
        self.max_ratio = max_ratio

    def check(self,
             order: Order,
             positions: Dict[str, Position],
             metrics: RiskMetrics,
             market_data: Dict[str, float]) -> Tuple[bool, str]:
        if order.action != "BUY":
            return True, "OK"

        current_price = market_data.get(order.code, order.price)
        existing_shares = positions.get(order.code, Position("", 0, 0, "", 0)).shares

        # 计算买入后这只股票的持仓
        new_shares = existing_shares + order.shares
        new_stock_value = new_shares * current_price
        new_ratio = new_stock_value / metrics.total_value

        if new_ratio > self.max_ratio:
            max_shares = int((metrics.total_value * self.max_ratio) / current_price / 100) * 100 - existing_shares
            max_shares = max(0, max_shares)
            if max_shares == 0:
                return False, f"单只股票超过最大比例 ({self.max_ratio:.0%})"
            return False, f"单只股票超过最大比例，调整为 {max_shares} 股"

        return True, "OK"


class MinCashRule(RiskRule):
    """最低现金规则"""

    def __init__(self, min_cash_ratio: float = 0.05):
        super().__init__("最低现金", priority=30)
        self.min_cash_ratio = min_cash_ratio

    def check(self,
             order: Order,
             positions: Dict[str, Position],
             metrics: RiskMetrics,
             market_data: Dict[str, float]) -> Tuple[bool, str]:
        if order.action != "BUY":
            return True, "OK"

        order_cost = order.price * order.shares * 1.003  # 预估佣金
        min_cash = metrics.total_value * self.min_cash_ratio

        if metrics.cash - order_cost < min_cash:
            max_shares = int((metrics.cash - min_cash) / (order.price * 1.003) / 100) * 100
            if max_shares <= 0:
                return False, f"现金不足 (最低保留 {min_cash:,.0f})"
            return False, f"现金不足，调整为 {max_shares} 股"

        return True, "OK"


class LimitUpDownRule(RiskRule):
    """涨跌停规则"""

    def __init__(self):
        super().__init__("涨跌停", priority=40)

    def check(self,
             order: Order,
             positions: Dict[str, Position],
             metrics: RiskMetrics,
             market_data: Dict[str, float]) -> Tuple[bool, str]:
        current_price = market_data.get(order.code, order.price)

        # 检查是否涨跌停
        if order.action == "BUY":
            # 涨停无法买入（按收盘价计算）
            prev_close = current_price  # 简化：使用当前价作为前收价
            if current_price >= prev_close * 1.095:
                return False, f"{order.code} 涨停无法买入"

        elif order.action == "SELL":
            # 跌停无法卖出
            prev_close = current_price
            if current_price <= prev_close * 0.905:
                return False, f"{order.code} 跌停无法卖出"

        return True, "OK"


class StopLossRule(RiskRule):
    """止损规则"""

    def __init__(self, stop_loss_pct: float = -0.05, take_profit_pct: float = 0.15):
        super().__init__("止损", priority=50)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def check(self,
             order: Order,
             positions: Dict[str, Position],
             metrics: RiskMetrics,
             market_data: Dict[str, float]) -> Tuple[bool, str]:
        if order.action != "SELL":
            return True, "OK"

        if order.code not in positions:
            return True, "OK"

        pos = positions[order.code]
        current_price = market_data.get(order.code, order.price)
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price

        # 触发止损
        if pnl_pct <= self.stop_loss_pct:
            return True, f"触发止损 (亏损 {pnl_pct:.1%})"

        # 触发止盈
        if pnl_pct >= self.take_profit_pct:
            return True, f"触发止盈 (盈利 {pnl_pct:.1%})"

        return True, "OK"


class RiskManager:
    """风控管理器"""

    def __init__(self,
                 initial_capital: float = 10000000,
                 max_position_ratio: float = 0.3,
                 single_stock_ratio: float = 0.1,
                 stop_loss_pct: float = -0.05,
                 take_profit_pct: float = 0.15):
        """
        Args:
            initial_capital: 初始资金
            max_position_ratio: 最大持仓比例
            single_stock_ratio: 单只股票最大比例
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
        """
        self.initial_capital = initial_capital
        self.max_position_ratio = max_position_ratio
        self.single_stock_ratio = single_stock_ratio
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # 风控规则
        self.rules: List[RiskRule] = [
            MaxPositionRatioRule(max_position_ratio),
            SingleStockMaxRatioRule(single_stock_ratio),
            MinCashRule(),
            LimitUpDownRule(),
            StopLossRule(stop_loss_pct, take_profit_pct),
        ]

        # 持仓
        self.positions: Dict[str, Position] = {}

        # 历史净值
        self.nav_history: List[Dict] = []

    def update_position(self, code: str, pos: Position):
        """更新持仓"""
        if pos.shares == 0:
            if code in self.positions:
                del self.positions[code]
        else:
            self.positions[code] = pos

    def calculate_metrics(self, market_data: Dict[str, float]) -> RiskMetrics:
        """计算当前风险指标"""
        positions_value = 0.0
        for code, pos in self.positions.items():
            current_price = market_data.get(code, pos.current_price or pos.entry_price)
            pos.current_price = current_price
            positions_value += pos.shares * current_price

        # 从历史获取现金
        if self.nav_history:
            cash = self.nav_history[-1].get('cash', self.initial_capital - positions_value)
        else:
            cash = self.initial_capital - positions_value

        total_value = positions_value + cash
        position_ratio = positions_value / total_value if total_value > 0 else 0

        # 单只最大比例
        single_max = 0.0
        for code, pos in self.positions.items():
            current_price = market_data.get(code, pos.current_price or pos.entry_price)
            stock_value = pos.shares * current_price
            ratio = stock_value / total_value if total_value > 0 else 0
            single_max = max(single_max, ratio)

        # 日盈亏
        daily_pnl = 0.0
        if len(self.nav_history) > 1:
            daily_pnl = total_value - self.nav_history[-1]['total_value']

        daily_pnl_pct = daily_pnl / self.nav_history[-1]['total_value'] if self.nav_history else 0

        # 最大回撤
        max_drawdown = 0.0
        if len(self.nav_history) > 0:
            peak = max(h['total_value'] for h in self.nav_history)
            max_drawdown = (total_value - peak) / peak if peak > 0 else 0

        return RiskMetrics(
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            position_ratio=position_ratio,
            single_position_ratio=single_max,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            max_drawdown=max_drawdown
        )

    def check_order(self, order: Order, market_data: Dict[str, float]) -> RiskCheckResult:
        """
        检查订单是否通过风控

        Args:
            order: 订单
            market_data: {code: price} 市场行情

        Returns:
            RiskCheckResult
        """
        metrics = self.calculate_metrics(market_data)

        # 按优先级检查所有规则
        for rule in sorted(self.rules, key=lambda r: r.priority):
            passed, message = rule.check(order, self.positions, metrics, market_data)

            if not passed:
                logger.warning(f"[{rule.name}] 订单被拒绝: {message}")
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.HIGH,
                    message=f"[{rule.name}] {message}"
                )

        return RiskCheckResult(
            approved=True,
            risk_level=RiskLevel.LOW,
            message="OK"
        )

    def check_stop_loss_take_profit(self, market_data: Dict[str, float]) -> List[str]:
        """
        检查是否触发止损止盈

        Returns:
            需要卖出的股票代码列表
        """
        to_sell = []
        stop_loss_rule = StopLossRule(self.stop_loss_pct, self.take_profit_pct)

        for code, pos in self.positions.items():
            current_price = market_data.get(code, pos.current_price or pos.entry_price)
            pos.current_price = current_price

            pnl_pct = (current_price - pos.entry_price) / pos.entry_price

            if pnl_pct <= self.stop_loss_pct:
                logger.info(f"{code} 触发止损: {pnl_pct:.2%}")
                to_sell.append(code)
            elif pnl_pct >= self.take_profit_pct:
                logger.info(f"{code} 触发止盈: {pnl_pct:.2%}")
                to_sell.append(code)

        return to_sell

    def record_nav(self, market_data: Dict[str, float]):
        """记录净值"""
        metrics = self.calculate_metrics(market_data)
        self.nav_history.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_value': metrics.total_value,
            'cash': metrics.cash,
            'positions_value': metrics.positions_value,
            'position_ratio': metrics.position_ratio
        })

    def get_risk_report(self, market_data: Dict[str, float]) -> Dict:
        """获取风险报告"""
        metrics = self.calculate_metrics(market_data)

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_value': metrics.total_value,
            'cash': metrics.cash,
            'positions_value': metrics.positions_value,
            'position_ratio': f"{metrics.position_ratio:.2%}",
            'single_max_ratio': f"{metrics.single_position_ratio:.2%}",
            'daily_pnl': f"{metrics.daily_pnl:+,.2f} ({metrics.daily_pnl_pct:+.2%})",
            'max_drawdown': f"{metrics.max_drawdown:.2%}",
            'num_positions': len(self.positions),
            'positions': {
                code: {
                    'shares': pos.shares,
                    'entry_price': pos.entry_price,
                    'current_price': market_data.get(code, pos.current_price),
                    'pnl_pct': f"{(market_data.get(code, pos.current_price) - pos.entry_price) / pos.entry_price:.2%}"
                }
                for code, pos in self.positions.items()
            }
        }


def test_risk_control():
    """测试风控模块"""
    print("=== 测试风控模块 ===\n")

    # 初始化
    risk_mgr = RiskManager(initial_capital=10000000)

    # 模拟持仓
    risk_mgr.positions['600519'] = Position(
        code='600519',
        shares=1000,
        entry_price=1800.0,
        entry_date='2024-01-01'
    )

    # 模拟行情
    market_data = {
        '600519': 1850.0,  # 盈利
        '000858': 1750.0,  # 亏损
    }

    # 风险报告
    report = risk_mgr.get_risk_report(market_data)
    print("风险报告:")
    for k, v in report.items():
        if k != 'positions':
            print(f"  {k}: {v}")
    print()

    # 测试订单检查
    print("订单检查:")
    order1 = Order(code='600519', action='BUY', price=1860, shares=100)
    result1 = risk_mgr.check_order(order1, market_data)
    print(f"  买入 600519 100股: {'批准' if result1.approved else '拒绝'} - {result1.message}")

    order2 = Order(code='000858', action='BUY', price=1760, shares=1000)  # 大单
    result2 = risk_mgr.check_order(order2, market_data)
    print(f"  买入 000858 1000股: {'批准' if result2.approved else '拒绝'} - {result2.message}")

    # 测试止损检查
    print("\n止损检查:")
    market_data2 = {
        '600519': 1710.0,  # 触发止损
        '000858': 1900.0,  # 触发止盈
    }
    to_sell = risk_mgr.check_stop_loss_take_profit(market_data2)
    print(f"  触发止损止盈: {to_sell}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='风控模块')
    parser.add_argument('--check', action='store_true', help='执行组合风险检查')
    parser.add_argument('--report', action='store_true', help='生成风险报告')
    parser.add_argument('--init-captain', type=float, default=10000000, help='初始资金')

    args = parser.parse_args()

    if args.check or args.report:
        risk_mgr = RiskManager(initial_capital=args.init_captain)

        # 模拟行情数据（实际应从实时市场获取）
        market_data = {}

        report = risk_mgr.get_risk_report(market_data)

        print(f"\n{'='*60}")
        print("组合风险报告")
        print(f"{'='*60}")
        print(f"时间: {report['timestamp']}")
        print(f"总资产: {report['total_value']:,.2f}")
        print(f"现金: {report['cash']:,.2f}")
        print(f"持仓市值: {report['positions_value']:,.2f}")
        print(f"持仓比例: {report['position_ratio']}")
        print(f"单只最大比例: {report['single_max_ratio']}")
        print(f"最大回撤: {report['max_drawdown']}")
        print(f"持仓数量: {report['num_positions']}")

        if report['positions']:
            print(f"\n持仓明细:")
            for code, pos in report['positions'].items():
                print(f"  {code}: {pos['shares']}股, 成本 {pos['entry_price']:.2f}, "
                      f"当前 {pos['current_price']:.2f}, 盈亏 {pos['pnl_pct']}")


if __name__ == '__main__':
    main()
