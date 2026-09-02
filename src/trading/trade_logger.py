#!/usr/bin/env python3
"""
交易日志模块
============

记录每一笔信号、委托、成交，用于复盘分析。

功能:
1. 信号日志表设计
2. 委托/成交日志表设计
3. 复盘查询接口
4. 绩效统计

使用方法:
    from src.trading.trade_logger import TradeLogger

    logger = TradeLogger()
    logger.log_signal(signal)
    logger.log_order(order)
    logger.log_trade(trade)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"      # 待报
    SUBMITTED = "SUBMITTED"  # 已提交
    PARTIAL = "PARTIAL"      # 部分成交
    FILLED = "FILLED"        # 全部成交
    CANCELLED = "CANCELLED"  # 已取消
    REJECTED = "REJECTED"    # 已拒绝


class SignalStatus(Enum):
    """信号状态"""
    PENDING = "PENDING"    # 待执行
    EXECUTED = "EXECUTED"  # 已执行
    EXPIRED = "EXPIRED"    # 已过期
    CANCELLED = "CANCELLED"  # 已取消


@dataclass
class SignalRecord:
    """信号记录"""
    signal_id: str
    timestamp: str
    code: str
    signal_type: str  # BUY / SELL
    score: float
    price: float
    reason: str
    status: str = SignalStatus.PENDING.value
    order_id: str = ""
    executed_price: float = 0.0
    executed_shares: int = 0
    executed_time: str = ""


@dataclass
class OrderRecord:
    """委托记录"""
    order_id: str
    signal_id: str
    timestamp: str
    code: str
    action: str  # BUY / SELL
    price: float
    shares: int
    order_type: str  # MARKET / LIMIT
    status: str = OrderStatus.PENDING.value
    submitted_time: str = ""
    filled_shares: int = 0
    avg_fill_price: float = 0.0
    filled_time: str = ""
    cancelled_time: str = ""
    reject_reason: str = ""


@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str
    order_id: str
    timestamp: str
    code: str
    action: str  # BUY / SELL
    price: float
    shares: int
    amount: float  # 成交金额
    commission: float  # 佣金
    slippage: float  # 滑点成本


@dataclass
class DailySummary:
    """每日汇总"""
    date: str
    signals_generated: int = 0
    signals_executed: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    total_buy_amount: float = 0.0
    total_sell_amount: float = 0.0
    total_commission: float = 0.0
    net_value: float = 0.0
    positions_count: int = 0


class TradeLogger:
    """交易日志管理器"""

    def __init__(self, db_path: str = "data/trading_logs.db"):
        """
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self._conn = None

        # 创建目录
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)

        # 初始化数据库
        self._init_db()

        # 内存缓存
        self.signals: List[SignalRecord] = []
        self.orders: List[OrderRecord] = []
        self.trades: List[TradeRecord] = []

    def _get_conn(self):
        """获取数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 信号表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                code TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                score REAL,
                price REAL,
                reason TEXT,
                status TEXT,
                order_id TEXT,
                executed_price REAL,
                executed_shares INTEGER,
                executed_time TEXT
            )
        """)

        # 委托表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                signal_id TEXT,
                timestamp TEXT NOT NULL,
                code TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL,
                shares INTEGER,
                order_type TEXT,
                status TEXT,
                submitted_time TEXT,
                filled_shares INTEGER,
                avg_fill_price REAL,
                filled_time TEXT,
                cancelled_time TEXT,
                reject_reason TEXT
            )
        """)

        # 成交表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                code TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL,
                shares INTEGER,
                amount REAL,
                commission REAL,
                slippage REAL
            )
        """)

        # 每日汇总表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                signals_generated INTEGER,
                signals_executed INTEGER,
                orders_submitted INTEGER,
                orders_filled INTEGER,
                orders_cancelled INTEGER,
                total_buy_amount REAL,
                total_sell_amount REAL,
                total_commission REAL,
                net_value REAL,
                positions_count INTEGER
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_code ON signals(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_code ON orders(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_code ON trades(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")

        conn.commit()

    def log_signal(self, signal: SignalRecord):
        """记录信号"""
        self.signals.append(signal)

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.signal_id,
            signal.timestamp,
            signal.code,
            signal.signal_type,
            signal.score,
            signal.price,
            signal.reason,
            signal.status,
            signal.order_id,
            signal.executed_price,
            signal.executed_shares,
            signal.executed_time
        ))
        conn.commit()

    def log_order(self, order: OrderRecord):
        """记录委托"""
        self.orders.append(order)

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id,
            order.signal_id,
            order.timestamp,
            order.code,
            order.action,
            order.price,
            order.shares,
            order.order_type,
            order.status,
            order.submitted_time,
            order.filled_shares,
            order.avg_fill_price,
            order.filled_time,
            order.cancelled_time,
            order.reject_reason
        ))
        conn.commit()

    def log_trade(self, trade: TradeRecord):
        """记录成交"""
        self.trades.append(trade)

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id,
            trade.order_id,
            trade.timestamp,
            trade.code,
            trade.action,
            trade.price,
            trade.shares,
            trade.amount,
            trade.commission,
            trade.slippage
        ))
        conn.commit()

    def update_signal_status(self, signal_id: str, status: str, order_id: str = "",
                            executed_price: float = 0, executed_shares: int = 0):
        """更新信号状态"""
        for s in self.signals:
            if s.signal_id == signal_id:
                s.status = status
                s.order_id = order_id
                s.executed_price = executed_price
                s.executed_shares = executed_shares
                s.executed_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if status == SignalStatus.EXECUTED.value else ""
                break

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE signals SET status = ?, order_id = ?, executed_price = ?,
            executed_shares = ?, executed_time = ? WHERE signal_id = ?
        """, (status, order_id, executed_price, executed_shares,
              datetime.now().strftime('%Y-%m-%d %H:%M:%S') if status == SignalStatus.EXECUTED.value else "",
              signal_id))
        conn.commit()

    def update_order_status(self, order_id: str, status: str, filled_shares: int = 0,
                           avg_fill_price: float = 0):
        """更新委托状态"""
        for o in self.orders:
            if o.order_id == order_id:
                o.status = status
                if status == OrderStatus.SUBMITTED.value and not o.submitted_time:
                    o.submitted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                elif status == OrderStatus.FILLED.value:
                    o.filled_shares = filled_shares
                    o.avg_fill_price = avg_fill_price
                    o.filled_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                break

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE orders SET status = ?, filled_shares = ?, avg_fill_price = ?,
            filled_time = ? WHERE order_id = ?
        """, (status, filled_shares, avg_fill_price,
              datetime.now().strftime('%Y-%m-%d %H:%M:%S') if status == OrderStatus.FILLED.value else "",
              order_id))
        conn.commit()

    def get_trades(self,
                   start_date: str = None,
                   end_date: str = None,
                   code: str = None) -> pd.DataFrame:
        """查询成交记录"""
        conn = self._get_conn()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if code:
            query += " AND code = ?"
            params.append(code)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_orders(self,
                   start_date: str = None,
                   end_date: str = None,
                   code: str = None,
                   status: str = None) -> pd.DataFrame:
        """查询委托记录"""
        conn = self._get_conn()

        query = "SELECT * FROM orders WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if code:
            query += " AND code = ?"
            params.append(code)
        if status:
            query += " AND status = ?"
            params.append(status)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_daily_summary(self, date: str = None) -> pd.DataFrame:
        """查询每日汇总"""
        conn = self._get_conn()

        if date:
            df = pd.read_sql_query("SELECT * FROM daily_summary WHERE date = ?", conn, params=[date])
        else:
            df = pd.read_sql_query("SELECT * FROM daily_summary ORDER BY date DESC", conn)

        conn.close()
        return df

    def calculate_performance(self,
                             start_date: str = None,
                             end_date: str = None) -> Dict:
        """计算绩效"""
        trades_df = self.get_trades(start_date, end_date)

        if trades_df.empty:
            return {}

        # 买入总额
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']

        total_buy = buy_trades['amount'].sum()
        total_sell = sell_trades['amount'].sum()
        total_commission = trades_df['commission'].sum()

        # 盈利统计
        # 按股票分组，计算每只股票的盈亏
        stock_pnls = []
        for code in trades_df['code'].unique():
            code_trades = trades_df[trades_df['code'] == code].sort_values('timestamp')

            buy_shares = 0
            buy_cost = 0
            pnl = 0

            for _, t in code_trades.iterrows():
                if t['action'] == 'BUY':
                    buy_shares += t['shares']
                    buy_cost += t['amount'] + t['commission']
                else:  # SELL
                    sell_amount = t['amount'] - t['commission']
                    avg_buy_cost = buy_cost / buy_shares if buy_shares > 0 else 0
                    pnl += sell_amount - avg_buy_cost * t['shares']
                    buy_shares -= t['shares']
                    buy_cost = avg_buy_cost * buy_shares

            if buy_shares == 0:  # 完全平仓
                stock_pnls.append({'code': code, 'pnl': pnl})

        winning = len([x for x in stock_pnls if x['pnl'] > 0])
        losing = len([x for x in stock_pnls if x['pnl'] <= 0])

        return {
            'total_buy_amount': total_buy,
            'total_sell_amount': total_sell,
            'total_commission': total_commission,
            'net_pnl': total_sell - total_buy - total_commission,
            'total_trades': len(trades_df),
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': winning / (winning + losing) if (winning + losing) > 0 else 0
        }


def test_trade_logger():
    """测试交易日志"""
    print("=== 测试交易日志 ===\n")

    # 创建日志管理器
    logger = TradeLogger(":memory:")  # 使用内存数据库

    # 记录信号
    signal1 = SignalRecord(
        signal_id="SIG001",
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        code="600519",
        signal_type="BUY",
        score=85.0,
        price=1800.0,
        reason="RSI 超卖"
    )
    logger.log_signal(signal1)
    print(f"记录信号: {signal1.signal_id}")

    # 记录委托
    order1 = OrderRecord(
        order_id="ORD001",
        signal_id="SIG001",
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        code="600519",
        action="BUY",
        price=1800.0,
        shares=100,
        order_type="MARKET"
    )
    logger.log_order(order1)
    print(f"记录委托: {order1.order_id}")

    # 记录成交
    trade1 = TradeRecord(
        trade_id="TRD001",
        order_id="ORD001",
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        code="600519",
        action="BUY",
        price=1800.5,
        shares=100,
        amount=180050.0,
        commission=54.0,
        slippage=50.0
    )
    logger.log_trade(trade1)
    print(f"记录成交: {trade1.trade_id}")

    # 更新状态
    logger.update_signal_status("SIG001", SignalStatus.EXECUTED.value, "ORD001", 1800.5, 100)
    logger.update_order_status("ORD001", OrderStatus.FILLED.value, 100, 1800.5)
    print("更新状态完成")

    # 查询
    trades = logger.get_trades()
    print(f"\n成交记录: {len(trades)} 条")
    print(trades)

    # 绩效
    perf = logger.calculate_performance()
    print(f"\n绩效: {perf}")


if __name__ == '__main__':
    test_trade_logger()
