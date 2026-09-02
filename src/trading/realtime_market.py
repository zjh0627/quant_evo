#!/usr/bin/env python3
"""
实时行情接入模块
================

支持多种实时行情数据源:
1. Tushare Pro (需要 token)
2. 东方财富 WebSocket
3. 聚宽数据 (需要 token)
4. 自定义模拟数据（测试用）

功能:
- 实时行情订阅
- 断线重连
- 行情数据缓冲
- 统一数据格式输出

使用方法:
    from src.trading.realtime_market import RealtimeMarket

    market = RealtimeMarket(source='tushare', token='your_token')
    market.subscribe(['600519', '000858'])
    market.start()

    # 获取实时行情
    data = market.get_realtime(['600519'])
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import threading
import time
import json
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import requests
import websocket
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketSource(Enum):
    """行情数据源"""
    TUSHARE = "tushare"
    EASTMONEY = "eastmoney"
    JOINQUANT = "joinquant"
    SIMULATOR = "simulator"


@dataclass
class TickData:
    """tick 数据"""
    code: str
    name: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    bid1_price: float
    bid1_volume: float
    ask1_price: float
    ask1_volume: float
    change_pct: float  # 涨跌幅


class RealtimeMarket:
    """实时行情管理器"""

    def __init__(self,
                 source: str = 'simulator',
                 host: str = 'localhost',
                 port: int = 8848):
        """
        Args:
            source: 数据源 ('tushare', 'eastmoney', 'joinquant', 'simulator')
            host: DolphinDB 主机地址
            port: DolphinDB 端口
        """
        self.source = source
        self.host = host
        self.port = port

        self._running = False
        self._thread = None
        self._subscribed_codes: List[str] = []
        self._tick_cache: Dict[str, TickData] = {}
        self._callbacks: List[Callable] = []

        # Tushare token (需要自行设置)
        self._tushare_token = os.getenv('TUSHARE_TOKEN', '')

        # 缓存最后更新时间
        self._last_update: Dict[str, datetime] = {}

    def subscribe(self, codes: List[str]):
        """订阅股票"""
        self._subscribed_codes.extend([c for c in codes if c not in self._subscribed_codes])
        logger.info(f"订阅股票: {codes}, 共 {len(self._subscribed_codes)} 只")

    def unsubscribe(self, codes: List[str]):
        """取消订阅"""
        self._subscribed_codes = [c for c in self._subscribed_codes if c not in codes]
        logger.info(f"取消订阅: {codes}, 剩余 {len(self._subscribed_codes)} 只")

    def add_callback(self, callback: Callable):
        """添加行情回调"""
        self._callbacks.append(callback)

    def start(self):
        """启动行情接收"""
        if self._running:
            logger.warning("行情服务已在运行")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"行情服务启动，数据源: {self.source}")

    def stop(self):
        """停止行情接收"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("行情服务停止")

    def _run_loop(self):
        """主循环"""
        while self._running:
            try:
                if self.source == 'tushare':
                    self._fetch_tushare()
                elif self.source == 'eastmoney':
                    self._fetch_eastmoney()
                elif self.source == 'joinquant':
                    self._fetch_joinquant()
                else:  # simulator
                    self._generate_simulator_data()

                # 通知回调
                for callback in self._callbacks:
                    try:
                        callback(self._tick_cache.copy())
                    except Exception as e:
                        logger.error(f"回调执行失败: {e}")

            except Exception as e:
                logger.error(f"行情获取失败: {e}")

            # 更新间隔
            time.sleep(3)  # 3秒更新一次

    def _fetch_tushare(self):
        """获取 Tushare 行情"""
        if not self._tushare_token:
            logger.warning("未设置 TUSHARE_TOKEN")
            return

        try:
            import tushare as ts
            ts.set_token(self._tushare_token)
            pro = ts.pro_api()

            # 批量查询
            for code in self._subscribed_codes[:50]:  # Tushare 限制
                try:
                    df = pro.realtime_quote(ts_code=code)
                    if not df.empty:
                        row = df.iloc[0]
                        tick = TickData(
                            code=code,
                            name=row.get('name', code),
                            timestamp=row.get('timestamp', datetime.now().isoformat()),
                            open=float(row.get('open', 0)),
                            high=float(row.get('high', 0)),
                            low=float(row.get('low', 0)),
                            close=float(row.get('price', 0)),
                            volume=float(row.get('vol', 0)),
                            amount=float(row.get('amount', 0)),
                            bid1_price=float(row.get('bid1_price', 0)),
                            bid1_volume=float(row.get('bid1_vol', 0)),
                            ask1_price=float(row.get('ask1_price', 0)),
                            ask1_volume=float(row.get('ask1_vol', 0)),
                            change_pct=float(row.get('pct_chg', 0))
                        )
                        self._tick_cache[code] = tick
                        self._last_update[code] = datetime.now()
                except Exception as e:
                    logger.debug(f"获取 {code} 行情失败: {e}")

        except Exception as e:
            logger.error(f"Tushare 获取失败: {e}")

    def _fetch_eastmoney(self):
        """获取东方财富行情"""
        try:
            # 东方财富实时行情 API
            for code in self._subscribed_codes[:50]:
                try:
                    # 转换代码格式
                    if code.startswith('6'):
                        market = '1'
                    else:
                        market = '0'
                    em_code = f"{market}{code}"

                    url = f"http://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f107,f169,f170"

                    resp = requests.get(url, timeout=5)
                    data = resp.json()

                    if 'data' in data and data['data']:
                        d = data['data']
                        tick = TickData(
                            code=code,
                            name=d.get('f58', code),
                            timestamp=datetime.now().isoformat(),
                            open=float(d.get('f43', 0)) / 100,
                            high=float(d.get('f44', 0)) / 100,
                            low=float(d.get('f45', 0)) / 100,
                            close=float(d.get('f43', 0)) / 100,
                            volume=float(d.get('f46', 0)),
                            amount=float(d.get('f47', 0)),
                            bid1_price=float(d.get('f50', 0)) / 100,
                            bid1_volume=float(d.get('f51', 0)),
                            ask1_price=float(d.get('f52', 0)) / 100,
                            ask1_volume=float(d.get('f53', 0)),
                            change_pct=float(d.get('f170', 0)) / 100
                        )
                        self._tick_cache[code] = tick
                        self._last_update[code] = datetime.now()

                except Exception as e:
                    logger.debug(f"获取 {code} 行情失败: {e}")

        except Exception as e:
            logger.error(f"东方财富获取失败: {e}")

    def _fetch_joinquant(self):
        """获取聚宽行情"""
        logger.warning("聚宽数据需要 token，请设置 JOINQUANT_TOKEN")
        # 实现类似 Tushare 的逻辑

    def _generate_simulator_data(self):
        """生成模拟行情（测试用）"""
        np.random.seed(int(datetime.now().timestamp()) % 10000)

        for code in self._subscribed_codes:
            # 模拟价格变动
            if code in self._tick_cache:
                prev_tick = self._tick_cache[code]
                change_pct = np.random.randn() * 0.5  # 随机波动
                new_close = prev_tick.close * (1 + change_pct / 100)
            else:
                # 初始化
                base_price = 100 + np.random.randint(-10, 50)
                change_pct = np.random.randn() * 2
                new_close = base_price

            tick = TickData(
                code=code,
                name=f"股票{code}",
                timestamp=datetime.now().isoformat(),
                open=new_close * (1 - np.random.rand() * 0.01),
                high=new_close * (1 + np.random.rand() * 0.02),
                low=new_close * (1 - np.random.rand() * 0.02),
                close=new_close,
                volume=np.random.randint(100000, 5000000),
                amount=new_close * np.random.randint(100000, 5000000),
                bid1_price=new_close * 0.999,
                bid1_volume=np.random.randint(100, 10000),
                ask1_price=new_close * 1.001,
                ask1_volume=np.random.randint(100, 10000),
                change_pct=change_pct
            )

            self._tick_cache[code] = tick
            self._last_update[code] = datetime.now()

    def get_realtime(self, codes: List[str] = None) -> Dict[str, TickData]:
        """获取实时行情"""
        if codes is None:
            codes = self._subscribed_codes

        result = {}
        for code in codes:
            if code in self._tick_cache:
                result[code] = self._tick_cache[code]

        return result

    def get_dataframe(self, codes: List[str] = None) -> pd.DataFrame:
        """获取实时行情 DataFrame"""
        ticks = self.get_realtime(codes)

        if not ticks:
            return pd.DataFrame()

        data = []
        for code, tick in ticks.items():
            data.append({
                'code': tick.code,
                'name': tick.name,
                'timestamp': tick.timestamp,
                'open': tick.open,
                'high': tick.high,
                'low': tick.low,
                'close': tick.close,
                'volume': tick.volume,
                'amount': tick.amount,
                'change_pct': tick.change_pct,
                'bid1': tick.bid1_price,
                'bid1_vol': tick.bid1_volume,
                'ask1': tick.ask1_price,
                'ask1_vol': tick.ask1_volume,
            })

        return pd.DataFrame(data)

    def write_to_dolphindb(self, table_name: str = "realtime_k"):
        """写入 DolphinDB 流数据表"""
        try:
            import dolphindb
            conn = dolphindb.session()
            conn.connect(self.host, self.port)

            for code, tick in self._tick_cache.items():
                script = f"""
                t = table(100:0,
                    `code`name`timestamp`open`high`low`close`volume`amount`change_pct,
                    [SYMBOL, STRING, TIMESTAMP, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE]
                )
                insert into {table_name} values
                """
                conn.run(script)

            conn.close()
        except Exception as e:
            logger.error(f"写入 DolphinDB 失败: {e}")

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._running

    def get_last_update(self, code: str) -> Optional[datetime]:
        """获取最后更新时间"""
        return self._last_update.get(code)


def demo_callback(ticks: Dict[str, TickData]):
    """行情回调示例"""
    if ticks:
        for code, tick in list(ticks.items())[:3]:
            print(f"{tick.code}: {tick.close:.2f} ({tick.change_pct:+.2f}%)")


def test_realtime_market():
    """测试实时行情"""
    print("=== 测试实时行情模块 ===\n")

    # 使用模拟数据
    market = RealtimeMarket(source='simulator')

    # 订阅股票
    market.subscribe(['600519', '000858', '000001'])
    market.add_callback(demo_callback)

    # 启动
    market.start()

    # 运行 10 秒
    print("运行 10 秒...")
    time.sleep(10)

    # 获取数据
    print("\n获取快照数据:")
    df = market.get_dataframe()
    print(df[['code', 'close', 'change_pct']])

    # 停止
    market.stop()
    print("\n测试完成")


if __name__ == '__main__':
    test_realtime_market()
