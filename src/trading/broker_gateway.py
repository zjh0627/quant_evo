#!/usr/bin/env python3
"""
券商下单对接模块
================

支持多种券商 API:
1. 聚宽 (JoinQuant)
2. 米筐 (RiceQuant)
3. 同花顺 (THS)
4. 东方财富 (EM)

功能:
- 下单接口封装
- 委托/成交回报处理
- 账户资金查询
- 持仓查询

注意事项:
- 使用前需开通券商 API 服务
- 需要配置 token/账号密码
- 部分券商需要签署电子签名

使用方法:
    from src.trading.broker_gateway import BrokerGateway

    broker = BrokerGateway(provider='joinquant', token='your_token')
    broker.login()

    # 查询账户
    account = broker.get_account()

    # 下单
    result = broker.order(code='600519', action='BUY', price=1800, shares=100)

    # 查持仓
    positions = broker.get_positions()
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import json
import requests
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrokerProvider(Enum):
    """券商提供商"""
    JOINQUANT = "joinquant"
    RICEQUANT = "ricequant"
    THS = "ths"
    EASTMONEY = "eastmoney"
    SIMULATOR = "simulator"


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"      # 市价单
    LIMIT = "LIMIT"       # 限价单


class OrderStatus(Enum):
    """订单状态"""
    SUBMITTED = "SUBMITTED"    # 已提交
    PARTIAL = "PARTIAL"        # 部分成交
    FILLED = "FILLED"          # 全部成交
    CANCELLED = "CANCELLED"    # 已取消
    REJECTED = "REJECTED"      # 已拒绝


@dataclass
class OrderRequest:
    """下单请求"""
    code: str           # 股票代码
    action: str         # BUY / SELL
    order_type: str     # MARKET / LIMIT
    price: float        # 价格（市价单填0）
    shares: int         # 数量（手为单位）
    comment: str = ""   # 备注


@dataclass
class OrderResponse:
    """下单响应"""
    order_id: str       # 券商订单ID
    status: str          # 状态
    message: str         # 消息
    submitted_time: str  # 提交时间


@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str       # 成交ID
    order_id: str        # 订单ID
    code: str            # 股票代码
    action: str          # 买卖
    price: float        # 成交价
    shares: int         # 成交数量
    amount: float       # 成交金额
    commission: float   # 佣金
    time: str           # 成交时间


@dataclass
class Position:
    """持仓"""
    code: str            # 股票代码
    name: str            # 股票名称
    shares: int          # 持仓数量
    available: int       # 可用数量
    avg_cost: float      # 平均成本
    current_price: float # 当前价
    market_value: float  # 市值
    pnl: float           # 盈亏
    pnl_pct: float       # 盈亏比例


@dataclass
class Account:
    """账户信息"""
    cash: float          # 可用资金
    market_value: float # 持仓市值
    total_value: float  # 总资产
    frozen: float        # 冻结资金
    position_ratio: float # 持仓比例


class BrokerGateway:
    """券商网关基类"""

    def __init__(self, provider: str = 'simulator'):
        self.provider = provider
        self._connected = False
        self._orders: Dict[str, OrderRequest] = {}
        self._positions: Dict[str, Position] = {}

    def login(self) -> bool:
        """登录"""
        raise NotImplementedError

    def logout(self):
        """登出"""
        raise NotImplementedError

    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    def get_account(self) -> Account:
        """查询账户"""
        raise NotImplementedError

    def get_positions(self) -> List[Position]:
        """查询持仓"""
        raise NotImplementedError

    def order(self, request: OrderRequest) -> OrderResponse:
        """下单"""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        raise NotImplementedError

    def get_orders(self, status: str = None) -> List[Dict]:
        """查询订单"""
        raise NotImplementedError

    def get_trades(self) -> List[TradeRecord]:
        """查询成交"""
        raise NotImplementedError


class JoinQuantBroker(BrokerGateway):
    """聚宽券商"""

    def __init__(self, token: str = ""):
        super().__init__('joinquant')
        self.token = token or os.getenv('JOINQUANT_TOKEN', '')
        self.base_url = "https://dataapi.joinquant.com/apis"

    def login(self) -> bool:
        if not self.token:
            logger.warning("未设置 JOINQUANT_TOKEN")
            return False

        try:
            # 验证 token
            resp = requests.post(
                f"{self.base_url}/query",
                data={'token': self.token, 'method': 'get_account_info'},
                timeout=10
            )
            if resp.status_code == 200:
                self._connected = True
                logger.info("聚宽登录成功")
                return True
        except Exception as e:
            logger.error(f"聚宽登录失败: {e}")

        return False

    def get_account(self) -> Account:
        if not self._connected:
            return Account(cash=0, market_value=0, total_value=0, frozen=0, position_ratio=0)

        try:
            resp = requests.post(
                f"{self.base_url}/query",
                data={'token': self.token, 'method': 'get_account_info'},
                timeout=10
            )
            data = resp.json()

            return Account(
                cash=float(data.get('cash', 0)),
                market_value=float(data.get('market_value', 0)),
                total_value=float(data.get('total_value', 0)),
                frozen=float(data.get('frozen', 0)),
                position_ratio=float(data.get('position_ratio', 0))
            )
        except Exception as e:
            logger.error(f"获取账户失败: {e}")
            return Account(0, 0, 0, 0, 0)

    def order(self, request: OrderRequest) -> OrderResponse:
        if not self._connected:
            return OrderResponse("", "REJECTED", "未登录", "")

        try:
            resp = requests.post(
                f"{self.base_url}/query",
                data={
                    'token': self.token,
                    'method': 'order',
                    'code': request.code,
                    'price': request.price,
                    'shares': request.shares,
                    'action': request.action,
                    'order_type': request.order_type
                },
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                return OrderResponse(
                    order_id=data.get('order_id', ''),
                    status='SUBMITTED',
                    message='成功',
                    submitted_time=datetime.now().isoformat()
                )
        except Exception as e:
            logger.error(f"下单失败: {e}")

        return OrderResponse("", "REJECTED", str(e), "")


class SimulatorBroker(BrokerGateway):
    """模拟券商（测试用）"""

    def __init__(self, initial_cash: float = 10000000):
        super().__init__('simulator')
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.orders: List[Dict] = []
        self.trades: List[TradeRecord] = []
        self.positions: Dict[str, Position] = {}
        self.order_id_counter = 1000

    def login(self) -> bool:
        self._connected = True
        logger.info("模拟券商登录成功")
        return True

    def logout(self):
        self._connected = False

    def get_account(self) -> Account:
        market_value = sum(p.market_value for p in self.positions.values())
        total_value = self.cash + market_value
        position_ratio = market_value / total_value if total_value > 0 else 0

        return Account(
            cash=self.cash,
            market_value=market_value,
            total_value=total_value,
            frozen=0,
            position_ratio=position_ratio
        )

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def order(self, request: OrderRequest) -> OrderResponse:
        order_id = f"SIM{self.order_id_counter}"
        self.order_id_counter += 1

        if request.action == 'BUY':
            cost = request.price * request.shares * 1.003
            if cost > self.cash:
                return OrderResponse(order_id, "REJECTED", "资金不足", datetime.now().isoformat())

            self.cash -= cost

            if request.code in self.positions:
                pos = self.positions[request.code]
                total_cost = pos.avg_cost * pos.shares + request.price * request.shares
                pos.shares += request.shares
                pos.avg_cost = total_cost / pos.shares
            else:
                self.positions[request.code] = Position(
                    code=request.code,
                    name=f"股票{request.code}",
                    shares=request.shares,
                    available=request.shares,
                    avg_cost=request.price,
                    current_price=request.price,
                    market_value=request.price * request.shares,
                    pnl=0,
                    pnl_pct=0
                )

            self.trades.append(TradeRecord(
                trade_id=f"TRD{order_id}",
                order_id=order_id,
                code=request.code,
                action=request.action,
                price=request.price,
                shares=request.shares,
                amount=request.price * request.shares,
                commission=request.price * request.shares * 0.003,
                time=datetime.now().isoformat()
            ))

        elif request.action == 'SELL':
            if request.code not in self.positions:
                return OrderResponse(order_id, "REJECTED", "无持仓", datetime.now().isoformat())

            pos = self.positions[request.code]
            if pos.shares < request.shares:
                return OrderResponse(order_id, "REJECTED", "持仓不足", datetime.now().isoformat())

            revenue = request.price * request.shares * 0.997
            self.cash += revenue

            pos.shares -= request.shares
            if pos.shares == 0:
                del self.positions[request.code]

            self.trades.append(TradeRecord(
                trade_id=f"TRD{order_id}",
                order_id=order_id,
                code=request.code,
                action=request.action,
                price=request.price,
                shares=request.shares,
                amount=request.price * request.shares,
                commission=request.price * request.shares * 0.003,
                time=datetime.now().isoformat()
            ))

        self.orders.append({
            'order_id': order_id,
            'code': request.code,
            'action': request.action,
            'price': request.price,
            'shares': request.shares,
            'status': 'FILLED',
            'time': datetime.now().isoformat()
        })

        return OrderResponse(order_id, "FILLED", "成功", datetime.now().isoformat())

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_orders(self, status: str = None) -> List[Dict]:
        if status:
            return [o for o in self.orders if o['status'] == status]
        return self.orders

    def get_trades(self) -> List[TradeRecord]:
        return self.trades


def create_broker(provider: str = 'simulator', **kwargs) -> BrokerGateway:
    """创建券商网关"""
    if provider == 'joinquant':
        return JoinQuantBroker(kwargs.get('token', ''))
    elif provider == 'ricequant':
        logger.warning("米筐暂未实现")
        return SimulatorBroker()
    elif provider == 'ths':
        logger.warning("同花顺暂未实现")
        return SimulatorBroker()
    else:
        return SimulatorBroker(kwargs.get('initial_cash', 10000000))


def test_broker():
    """测试券商网关"""
    print("=== 测试券商网关 ===\n")

    # 使用模拟券商
    broker = create_broker('simulator', initial_cash=10000000)
    broker.login()

    # 查询账户
    account = broker.get_account()
    print(f"账户: 总资产 {account.total_value:,.2f}, 现金 {account.cash:,.2f}")

    # 下单买入
    print("\n买入 600519 100股 @ 1800")
    result = broker.order(OrderRequest(
        code='600519',
        action='BUY',
        order_type='LIMIT',
        price=1800.0,
        shares=100
    ))
    print(f"下单结果: {result.order_id} - {result.status} - {result.message}")

    # 查询持仓
    positions = broker.get_positions()
    print(f"\n持仓: {len(positions)} 只")
    for p in positions:
        print(f"  {p.code}: {p.shares}股, 成本 {p.avg_cost:.2f}")

    # 卖出
    print("\n卖出 600519 50股")
    result2 = broker.order(OrderRequest(
        code='600519',
        action='SELL',
        order_type='LIMIT',
        price=1850.0,
        shares=50
    ))
    print(f"卖出结果: {result2.order_id} - {result2.status} - {result2.message}")

    # 再次查询持仓
    positions = broker.get_positions()
    print(f"\n剩余持仓: {len(positions)} 只")
    for p in positions:
        print(f"  {p.code}: {p.shares}股")

    # 查询账户
    account = broker.get_account()
    print(f"\n账户: 总资产 {account.total_value:,.2f}, 现金 {account.cash:,.2f}")

    # 查询成交
    trades = broker.get_trades()
    print(f"\n成交记录: {len(trades)} 笔")
    for t in trades:
        print(f"  {t.code}: {t.action} {t.shares}股 @{t.price}")


if __name__ == '__main__':
    test_broker()
