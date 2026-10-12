#!/usr/bin/env python3
"""
行情提供者
==========

从 akshare 获取实时行情数据
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import os
import json
import logging
from datetime import datetime, time
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import pandas as pd

from .data_types import RealtimeQuote

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteProvider(ABC):
    """行情提供者基类"""

    @abstractmethod
    def fetch_quotes(self, codes: List[str]) -> Dict[str, RealtimeQuote]:
        """获取实时行情"""
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """检查是否开盘"""
        pass


class AkshareProvider(QuoteProvider):
    """akshare 行情提供者"""

    def __init__(self):
        self._cache: Dict[str, RealtimeQuote] = {}
        self._last_fetch: Optional[datetime] = None
        self._cache_seconds = 30  # 缓存30秒

    def is_market_open(self) -> bool:
        """检查是否在交易时间"""
        now = datetime.now()
        current_time = now.time()
        current_weekday = now.weekday()

        # 周末休市
        if current_weekday >= 5:
            return False

        # 9:30-11:30 (上午)
        morning_start = time(9, 30)
        morning_end = time(11, 30)

        # 13:00-15:00 (下午)
        afternoon_start = time(13, 0)
        afternoon_end = time(15, 0)

        if morning_start <= current_time <= morning_end:
            return True
        if afternoon_start <= current_time <= afternoon_end:
            return True

        return False

    def fetch_quotes(self, codes: List[str]) -> Dict[str, RealtimeQuote]:
        """获取实时行情"""
        # 检查缓存
        if self._cache and self._last_fetch:
            elapsed = (datetime.now() - self._last_fetch).total_seconds()
            if elapsed < self._cache_seconds:
                return self._cache

        try:
            import akshare as ak

            # 批量获取A股实时行情
            df = ak.stock_zh_a_spot_em()

            quotes = {}
            for code in codes:
                # 转换代码格式: sh.600519 -> 600519
                code_num = code.replace('sh.', '').replace('sz.', '').replace('bj.', '')
                market = 'SH' if code.startswith('sh') else 'SZ' if code.startswith('sz') else 'BJ'

                # 在 DataFrame 中查找
                symbol = f'{market}{code_num}'
                row = df[df['代码'] == code_num]

                if len(row) > 0:
                    r = row.iloc[0]
                    try:
                        quote = RealtimeQuote(
                            code=code,
                            name=str(r.get('名称', code)),
                            current=float(r.get('最新价', 0)),
                            open=float(r.get('开盘', 0)),
                            high=float(r.get('最高', 0)),
                            low=float(r.get('最低', 0)),
                            volume=float(r.get('成交量', 0)),
                            amount=float(r.get('成交额', 0)),
                            timestamp=datetime.now(),
                            pct_change=float(r.get('涨跌幅', 0))
                        )
                        quotes[code] = quote
                    except (ValueError, TypeError) as e:
                        logger.warning(f"解析行情失败 {code}: {e}")
                        continue

            # 更新缓存
            self._cache = quotes
            self._last_fetch = datetime.now()

            logger.info(f"获取行情成功: {len(quotes)}/{len(codes)} 只")
            return quotes

        except Exception as e:
            logger.error(f"获取行情失败: {e}")
            # 返回缓存（如果有）
            return self._cache if self._cache else {}

    def fetch_single_quote(self, code: str) -> Optional[RealtimeQuote]:
        """获取单只股票行情"""
        quotes = self.fetch_quotes([code])
        return quotes.get(code)


class CachedQuoteProvider(QuoteProvider):
    """带缓存的行情提供者"""

    def __init__(self, provider: QuoteProvider, cache_file: str):
        self.provider = provider
        self.cache_file = cache_file
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def is_market_open(self) -> bool:
        return self.provider.is_market_open()

    def fetch_quotes(self, codes: List[str]) -> Dict[str, RealtimeQuote]:
        # 尝试从缓存加载
        cached = self._load_cache()
        if cached:
            return cached

        # 从 provider 获取
        quotes = self.provider.fetch_quotes(codes)
        self._save_cache(quotes)
        return quotes

    def _load_cache(self) -> Optional[Dict[str, RealtimeQuote]]:
        """从文件加载缓存"""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            quotes = {}
            for q in data.get('quotes', []):
                q['timestamp'] = datetime.strptime(q['timestamp'], '%Y-%m-%d %H:%M:%S')
                quotes[q['code']] = RealtimeQuote(**q)

            # 检查缓存是否过期（超过5分钟）
            cache_time = datetime.strptime(data.get('timestamp', ''), '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - cache_time).total_seconds() > 300:
                return None  # 缓存过期

            return quotes
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None

    def _save_cache(self, quotes: Dict[str, RealtimeQuote]):
        """保存到文件"""
        try:
            data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quotes': [q.to_dict() for q in quotes.values()]
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")


def get_default_provider() -> QuoteProvider:
    """获取默认行情提供者"""
    akshare_provider = AkshareProvider()
    cache_file = f'{CACHE_DIR}/realtime_quotes.json'
    return CachedQuoteProvider(akshare_provider, cache_file)


def load_stock_pool() -> List[str]:
    """加载股票池"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('stocks', [])[:100]  # 最多100只
    return []


if __name__ == '__main__':
    # 测试行情获取
    print("测试行情获取...")

    provider = get_default_provider()
    codes = load_stock_pool()[:10]  # 测试前10只

    print(f"股票池: {codes}")
    print(f"开盘状态: {provider.is_market_open()}")

    if provider.is_market_open():
        quotes = provider.fetch_quotes(codes)
        print(f"\n获取到 {len(quotes)} 只股票行情:")
        for code, quote in quotes.items():
            print(f"  {code}: {quote.name} = {quote.current} ({quote.pct_change:+.2f}%)")
    else:
        print("当前非交易时间，跳过测试")
