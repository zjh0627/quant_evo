#!/usr/bin/env python3
"""
信号生成服务
============

Python 常驻服务，每分钟运行策略产生买卖信号。

功能:
1. 定时任务调度器（支持 crontab 风格配置）
2. 策略逻辑执行（基于技术指标）
3. 信号输出与记录（DolphinDB + 文件）
4. WebSocket 实时推送

使用方法:
    from src.trading.signal_service import SignalService

    service = SignalService()
    service.start()
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: str
    code: str
    signal: SignalType
    score: float
    price: float
    reason: str
    factors: Dict[str, float]


class StrategyBase:
    """策略基类"""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        生成信号

        Args:
            data: K线数据，需包含 open/high/low/close/volume

        Returns:
            List[TradingSignal]
        """
        raise NotImplementedError


class RSIMomentumStrategy(StrategyBase):
    """RSI 动量策略"""

    def __init__(self, rsi_period: int = 14, rsi_buy: float = 30, rsi_sell: float = 70):
        super().__init__("RSI_Momentum")
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []

        if len(data) < self.rsi_period + 1:
            return signals

        close = data['close']
        rsi = self.calculate_rsi(close, self.rsi_period)

        # RSI 超卖买入
        if rsi.iloc[-1] < self.rsi_buy:
            signals.append(TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                code=data['code'].iloc[-1] if 'code' in data.columns else 'UNKNOWN',
                signal=SignalType.BUY,
                score=100 - rsi.iloc[-1],  # RSI 越低评分越高
                price=close.iloc[-1],
                reason=f"RSI 超卖 ({rsi.iloc[-1]:.1f} < {self.rsi_buy})",
                factors={'rsi': rsi.iloc[-1]}
            ))

        # RSI 超买卖出
        elif rsi.iloc[-1] > self.rsi_sell:
            signals.append(TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                code=data['code'].iloc[-1] if 'code' in data.columns else 'UNKNOWN',
                signal=SignalType.SELL,
                score=rsi.iloc[-1],  # RSI 越高评分越高
                price=close.iloc[-1],
                reason=f"RSI 超买 ({rsi.iloc[-1]:.1f} > {self.rsi_sell})",
                factors={'rsi': rsi.iloc[-1]}
            ))

        return signals


class MACDTrendStrategy(StrategyBase):
    """MACD 趋势策略"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD_Trend")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate_macd(self, close: pd.Series) -> tuple:
        """计算 MACD"""
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal_period).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []

        if len(data) < self.slow + 1:
            return signals

        close = data['close']
        macd, macd_signal, macd_hist = self.calculate_macd(close)

        # MACD 金叉买入
        if macd.iloc[-2] < macd_signal.iloc[-2] and macd.iloc[-1] > macd_signal.iloc[-1]:
            signals.append(TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                code=data['code'].iloc[-1] if 'code' in data.columns else 'UNKNOWN',
                signal=SignalType.BUY,
                score=min(abs(macd_hist.iloc[-1]) * 100, 100),
                price=close.iloc[-1],
                reason=f"MACD 金叉 (hist={macd_hist.iloc[-1]:.4f})",
                factors={'macd': macd.iloc[-1], 'macd_signal': macd_signal.iloc[-1], 'macd_hist': macd_hist.iloc[-1]}
            ))

        # MACD 死叉卖出
        elif macd.iloc[-2] > macd_signal.iloc[-2] and macd.iloc[-1] < macd_signal.iloc[-1]:
            signals.append(TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                code=data['code'].iloc[-1] if 'code' in data.columns else 'UNKNOWN',
                signal=SignalType.SELL,
                score=min(abs(macd_hist.iloc[-1]) * 100, 100),
                price=close.iloc[-1],
                reason=f"MACD 死叉 (hist={macd_hist.iloc[-1]:.4f})",
                factors={'macd': macd.iloc[-1], 'macd_signal': macd_signal.iloc[-1], 'macd_hist': macd_hist.iloc[-1]}
            ))

        return signals


class BollingerBandStrategy(StrategyBase):
    """布林带策略"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("BollingerBand")
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []

        if len(data) < self.period + 1:
            return signals

        close = data['close']
        mid = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = mid + std * self.std_dev
        lower = mid - std * self.std_dev

        # 价格触及下轨买入
        if close.iloc[-1] <= lower.iloc[-1]:
            signals.append(TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                code=data['code'].iloc[-1] if 'code' in data.columns else 'UNKNOWN',
                signal=SignalType.BUY,
                score=80,
                price=close.iloc[-1],
                reason=f"触及布林下轨 ({close.iloc[-1]:.2f} <= {lower.iloc[-1]:.2f})",
                factors={'bb_upper': upper.iloc[-1], 'bb_mid': mid.iloc[-1], 'bb_lower': lower.iloc[-1]}
            ))

        # 价格触及上轨卖出
        elif close.iloc[-1] >= upper.iloc[-1]:
            signals.append(TradingSignal(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                code=data['code'].iloc[-1] if 'code' in data.columns else 'UNKNOWN',
                signal=SignalType.SELL,
                score=80,
                price=close.iloc[-1],
                reason=f"触及布林上轨 ({close.iloc[-1]:.2f} >= {upper.iloc[-1]:.2f})",
                factors={'bb_upper': upper.iloc[-1], 'bb_mid': mid.iloc[-1], 'bb_lower': lower.iloc[-1]}
            ))

        return signals


class CompositeStrategy(StrategyBase):
    """复合策略 - 结合多个指标"""

    def __init__(self, min_score: float = 60):
        super().__init__("Composite")
        self.min_score = min_score
        self.strategies = [
            RSIMomentumStrategy(),
            MACDTrendStrategy(),
            BollingerBandStrategy()
        ]

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        all_signals = []

        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                all_signals.extend(signals)
            except Exception as e:
                logger.warning(f"策略 {strategy.name} 执行失败: {e}")

        # 合并同方向信号
        buy_signals = [s for s in all_signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in all_signals if s.signal == SignalType.SELL]

        if not buy_signals and not sell_signals:
            return []

        # 如果多个指标同时发出买入信号，提高评分
        if len(buy_signals) >= 2:
            for s in buy_signals:
                s.score = min(s.score + 20, 100)

        if len(sell_signals) >= 2:
            for s in sell_signals:
                s.score = min(s.score + 20, 100)

        # 返回最高评分的信号
        all_signals.sort(key=lambda x: x.score, reverse=True)

        return all_signals[:3]  # 最多返回3个信号


class SignalService:
    """信号生成服务"""

    def __init__(self,
                 data_provider: Callable = None,
                 output_path: str = "signals",
                 push_interval: int = 60):
        """
        Args:
            data_provider: 数据提供函数，返回 {code: DataFrame}
            output_path: 信号输出目录
            push_interval: 推送间隔（秒）
        """
        self.data_provider = data_provider
        self.output_path = output_path
        self.push_interval = push_interval

        self.strategies: List[StrategyBase] = []
        self.running = False
        self.thread = None

        # 信号缓存
        self.latest_signals: List[TradingSignal] = []
        self.signal_history: List[TradingSignal] = []

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        # 默认策略
        self.add_strategy(CompositeStrategy())

    def add_strategy(self, strategy: StrategyBase):
        """添加策略"""
        self.strategies.append(strategy)
        logger.info(f"添加策略: {strategy.name}")

    def scan_and_generate(self) -> List[TradingSignal]:
        """扫描市场并生成信号"""
        if self.data_provider is None:
            logger.warning("未设置数据提供器")
            return []

        try:
            data_dict = self.data_provider()
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return []

        all_signals = []

        for code, df in data_dict.items():
            if df is None or len(df) < 30:
                continue

            df = df.copy()
            df['code'] = code

            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(df)
                    all_signals.extend(signals)
                except Exception as e:
                    logger.warning(f"策略 {strategy.name} 执行失败 ({code}): {e}")

        # 更新缓存
        self.latest_signals = all_signals
        self.signal_history.extend(all_signals)

        return all_signals

    def save_signals(self, signals: List[TradingSignal]):
        """保存信号到文件"""
        if not signals:
            return

        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{self.output_path}/signals_{date_str}.json"

        # 读取现有信号
        existing = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    existing = json.load(f)
            except:
                pass

        # 添加新信号
        for s in signals:
            existing.append({
                'timestamp': s.timestamp,
                'code': s.code,
                'signal': s.signal.value,
                'score': s.score,
                'price': s.price,
                'reason': s.reason,
                'factors': s.factors
            })

        # 保存
        with open(filename, 'w') as f:
            json.dump(existing[-1000:], f, indent=2)  # 保留最近1000条

        logger.info(f"保存 {len(signals)} 条信号到 {filename}")

    def _run_loop(self):
        """运行主循环"""
        logger.info("信号服务启动")

        while self.running:
            try:
                # 生成信号
                signals = self.scan_and_generate()

                # 保存信号
                self.save_signals(signals)

                # 输出日志
                if signals:
                    buy_count = len([s for s in signals if s.signal == SignalType.BUY])
                    sell_count = len([s for s in signals if s.signal == SignalType.SELL])
                    logger.info(f"信号扫描: 买入 {buy_count}, 卖出 {sell_count}")

            except Exception as e:
                logger.error(f"信号生成失败: {e}")

            # 等待下一次执行
            time.sleep(self.push_interval)

        logger.info("信号服务停止")

    def start(self):
        """启动服务"""
        if self.running:
            logger.warning("服务已在运行")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止服务"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_latest_signals(self) -> List[TradingSignal]:
        """获取最新信号"""
        return self.latest_signals

    def get_signal_summary(self) -> Dict:
        """获取信号汇总"""
        if not self.latest_signals:
            return {'buy': 0, 'sell': 0, 'hold': 0}

        return {
            'buy': len([s for s in self.latest_signals if s.signal == SignalType.BUY]),
            'sell': len([s for s in self.latest_signals if s.signal == SignalType.SELL]),
            'hold': len([s for s in self.latest_signals if s.signal == SignalType.HOLD]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def demo_data_provider():
    """演示数据提供器"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')

    stocks = ['600519', '000858', '000001']
    data_dict = {}

    for code in stocks:
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(60) * 2)

        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(60) * 0.5,
            'high': prices + np.abs(np.random.randn(60) * 2),
            'low': prices - np.abs(np.random.randn(60) * 2),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 60)
        })

        data_dict[code] = df

    return data_dict


def test_signal_service():
    """测试信号服务"""
    print("=== 测试信号生成服务 ===")

    # 使用演示数据
    service = SignalService(data_provider=demo_data_provider)

    # 添加策略
    service.add_strategy(RSIMomentumStrategy())
    service.add_strategy(MACDTrendStrategy())

    # 手动执行一次扫描
    signals = service.scan_and_generate()

    print(f"\n生成信号数: {len(signals)}")

    for s in signals:
        print(f"  {s.code}: {s.signal.value} ({s.score:.1f}分) - {s.reason}")

    # 保存测试
    service.save_signals(signals)

    # 汇总
    summary = service.get_signal_summary()
    print(f"\n信号汇总: {summary}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='信号生成服务')
    parser.add_argument('--once', action='store_true', help='单次执行信号生成（用于 crontab）')
    parser.add_argument('--output', type=str, default='signals', help='信号输出目录')

    args = parser.parse_args()

    if args.once:
        # 单次执行模式（用于 crontab）
        service = SignalService(output_path=args.output)
        service.add_strategy(RSIMomentumStrategy())
        service.add_strategy(MACDTrendStrategy())
        service.add_strategy(BollingerBandStrategy())

        signals = service.scan_and_generate()
        service.save_signals(signals)

        summary = service.get_signal_summary()
        print(f"信号生成完成: {summary}")
    else:
        test_signal_service()
