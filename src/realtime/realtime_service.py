#!/usr/bin/env python3
"""
实时信号服务
============

主调度服务：每分钟获取行情、生成信号、广播推送
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Optional

from .data_types import RealtimeSignal, serialize_signals
from .quote_provider import get_default_provider, load_stock_pool
from .signal_generator import RealtimeSignalGenerator

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSignalService:
    """
    实时信号服务主类

    功能:
    1. 每分钟获取实时行情
    2. 计算技术指标和评分
    3. 生成交易信号
    4. 缓存信号到文件
    5. 通过 WebSocket 广播（如果启用）
    """

    def __init__(self, interval: int = 60):
        """
        Args:
            interval: 更新间隔（秒），默认60秒
        """
        self.interval = interval
        self.quote_provider = get_default_provider()
        self.signal_generator = RealtimeSignalGenerator()
        self.running = False
        self._thread: Optional[threading.Thread] = None

        # 缓存
        self._latest_signals: Dict[str, RealtimeSignal] = {}
        self._latest_quotes = {}
        self._last_update: Optional[datetime] = None

        # 回调函数
        self._callbacks = []

        # 股票池
        self.codes = []

    def initialize(self, codes: list = None):
        """初始化服务"""
        # 加载股票池
        if codes is None:
            self.codes = load_stock_pool()
        else:
            self.codes = codes

        logger.info(f"初始化实时信号服务: {len(self.codes)} 只股票")

        # 加载历史数据
        self.signal_generator.load_historical_data(self.codes)

        # 确保缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)

    def start(self):
        """启动服务"""
        if self.running:
            logger.warning("服务已在运行")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("实时信号服务已启动")

    def stop(self):
        """停止服务"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("实时信号服务已停止")

    def register_callback(self, callback):
        """注册回调函数，每次更新后调用"""
        self._callbacks.append(callback)

    def _run_loop(self):
        """主循环"""
        while self.running:
            try:
                # 检查交易时间
                if not self.quote_provider.is_market_open():
                    logger.debug("非交易时间，休眠30秒")
                    time.sleep(30)
                    continue

                # 执行一次更新
                self._update()

                # 等待下一次
                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"主循环异常: {e}")
                time.sleep(10)

    def _update(self):
        """执行一次更新"""
        try:
            start_time = datetime.now()

            # 1. 获取实时行情
            logger.info("获取实时行情...")
            quotes = self.quote_provider.fetch_quotes(self.codes)
            self._latest_quotes = quotes

            if not quotes:
                logger.warning("未获取到行情数据")
                return

            # 2. 生成信号
            logger.info("生成信号...")
            signals = self.signal_generator.generate_signals(quotes)
            self._latest_signals = signals
            self._last_update = datetime.now()

            # 3. 保存到缓存
            self._save_to_cache(signals)

            # 4. 通知回调
            for callback in self._callbacks:
                try:
                    callback(signals)
                except Exception as e:
                    logger.warning(f"回调函数异常: {e}")

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"更新完成: {len(signals)} 只信号, 耗时 {elapsed:.2f}秒")

        except Exception as e:
            logger.error(f"更新失败: {e}")

    def _save_to_cache(self, signals):
        """保存信号到缓存文件"""
        try:
            cache_file = f'{CACHE_DIR}/realtime_signals.json'
            data = serialize_signals(list(signals.values()))

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def get_latest_signals(self) -> Dict[str, RealtimeSignal]:
        """获取最新信号"""
        return self._latest_signals

    def get_latest_quotes(self):
        """获取最新行情"""
        return self._latest_quotes

    def get_last_update(self) -> Optional[datetime]:
        """获取最后更新时间"""
        return self._last_update

    def force_update(self):
        """强制立即更新"""
        self._update()


# 全局单例
_instance: Optional[RealtimeSignalService] = None


def get_instance() -> RealtimeSignalService:
    """获取全局单例"""
    global _instance
    if _instance is None:
        _instance = RealtimeSignalService()
    return _instance


def start_service(codes: list = None):
    """启动服务（便捷函数）"""
    service = get_instance()
    service.initialize(codes)
    service.start()
    return service


def stop_service():
    """停止服务（便捷函数）"""
    global _instance
    if _instance:
        _instance.stop()
        _instance = None


if __name__ == '__main__':
    # 测试服务
    print("测试实时信号服务...")

    service = RealtimeSignalService(interval=60)  # 60秒更新
    service.initialize(['sh.600519', 'sz.000858', 'sh.600085'])

    # 强制执行一次更新
    service.force_update()

    # 打印结果
    signals = service.get_latest_signals()
    print(f"\n获取到 {len(signals)} 只信号:")
    for code, signal in signals.items():
        print(f"  {code}: {signal.name} - {signal.signal} ({signal.score:.1f})")
        print(f"    原因: {signal.reason}")
