#!/usr/bin/env python3
"""
实时信号数据类
==============

定义实时行情和信号的 数据结构
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List


@dataclass
class RealtimeQuote:
    """实时行情"""
    code: str           # 股票代码 "sh.600519"
    name: str           # 股票名称
    current: float      # 当前价
    open: float         # 开盘价
    high: float         # 最高价
    low: float          # 最低价
    volume: float       # 成交量
    amount: float       # 成交额
    timestamp: datetime # 数据时间戳
    pct_change: float   # 涨跌幅 %

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealtimeSignal:
    """实时信号"""
    code: str
    name: str
    score: float            # 综合评分 0-100
    signal: str             # "BUY" / "SELL" / "HOLD"
    confidence: float       # 置信度 0-1
    reason: str             # 信号原因

    # 技术指标
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    boll_position: float = 0.5
    kdj_k: float = 50.0
    kdj_d: float = 50.0
    kdj_j: float = 50.0

    # 均线
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0

    # 动量
    momentum_1d: float = 0.0
    momentum_5d: float = 0.0

    # 情绪
    volume_ratio: float = 1.0
    turnover_rate: float = 0.0

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return d


@dataclass
class TradeSignal:
    """交易信号（用于执行层）"""
    code: str
    action: str               # "BUY" / "SELL" / "HOLD"
    score: float             # 当前评分
    threshold_buy: float      # 买入阈值
    threshold_sell: float     # 卖出阈值
    confidence: float         # 置信度 0-1
    reason: str               # 信号原因
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return d


def serialize_signals(signals: List[RealtimeSignal]) -> dict:
    """序列化信号列表为 JSON 兼容格式"""
    return {
        'signals': [s.to_dict() for s in signals],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
