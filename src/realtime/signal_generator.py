#!/usr/bin/env python3
"""
实时信号生成器
==============

基于实时行情和历史数据生成交易信号
复用 calculate_enhanced_features() 和 MultiStrategySignal 评分逻辑
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict

import pandas as pd
import numpy as np

from .data_types import RealtimeQuote, RealtimeSignal
from .quote_provider import get_default_provider, load_stock_pool

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'

# 复用现有的因子计算和信号生成
from src.simulation.evolving_portfolio import (
    calculate_enhanced_features,
    MultiStrategySignal,
    DEFAULT_PARAMS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSignalGenerator:
    """实时信号生成器"""

    def __init__(self, params: dict = None):
        self.params = params or DEFAULT_PARAMS.copy()
        self.quote_provider = get_default_provider()
        self.multi_strategy = MultiStrategySignal(use_hybrid=False)
        self._historical_data: Dict[str, pd.DataFrame] = {}

    def load_historical_data(self, codes: List[str]):
        """加载历史K线数据"""
        self._historical_data = {}

        for code in codes:
            # 尝试多种文件格式
            # sh.600085 -> kline_sh_600085.parquet 或 kline_sh.600085.parquet
            code_fmt1 = code.replace('.', '_')  # sh_600085
            code_fmt2 = code  # sh.600085

            parquet_file1 = f'{CACHE_DIR}/kline_{code_fmt1}.parquet'
            parquet_file2 = f'{CACHE_DIR}/kline_{code_fmt2}.parquet'
            csv_file = f'{DATA_DIR}/raw/kline_{code_fmt1}.csv'

            df = None
            if os.path.exists(parquet_file1):
                df = pd.read_parquet(parquet_file1)
            elif os.path.exists(parquet_file2):
                df = pd.read_parquet(parquet_file2)
            elif os.path.exists(csv_file):
                df = pd.read_csv(csv_file)

            if df is not None and len(df) >= 20:
                # 统一列名
                if 'vol' in df.columns and 'volume' not in df.columns:
                    df['volume'] = df['vol']
                if 'pctChg' in df.columns and 'pct_change' not in df.columns:
                    df['pct_change'] = df['pctChg']

                # 确保有 date 列
                if 'date' not in df.columns:
                    if 'trade_date' in df.columns:
                        if isinstance(df['trade_date'].iloc[0], int):
                            df['date'] = pd.to_datetime(df['trade_date'].astype(str)).dt.strftime('%Y-%m-%d')
                        else:
                            df['date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                    else:
                        continue

                # 确保有 close 列
                if 'close' not in df.columns:
                    continue

                self._historical_data[code] = df

        logger.info(f"加载历史数据: {len(self._historical_data)}/{len(codes)} 只")

    def generate_signals(
        self,
        quotes: Dict[str, RealtimeQuote],
        params: dict = None
    ) -> Dict[str, RealtimeSignal]:
        """
        生成实时信号

        Args:
            quotes: 实时行情字典
            params: 可选的参数覆盖

        Returns:
            {code: RealtimeSignal}
        """
        if params is None:
            params = self.params

        signals = {}
        buy_threshold = params.get('buy_threshold', 55)
        sell_threshold = params.get('sell_threshold', 42)

        for code, quote in quotes.items():
            # 获取历史数据
            df = self._historical_data.get(code)
            if df is None or len(df) < 20:
                continue

            try:
                # 追加最新行情到历史数据
                df_updated = self._append_quote(df, quote)

                # 计算增强特征
                df_enhanced = calculate_enhanced_features(df_updated)

                # 获取最新指标
                if len(df_enhanced) < 1:
                    continue

                latest = df_enhanced.iloc[-1]

                # 计算综合评分
                score = self._calculate_score(latest, params)

                # 生成信号
                if score >= buy_threshold:
                    signal_type = 'BUY'
                elif score <= sell_threshold:
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'

                # 计算置信度
                confidence = self._calculate_confidence(score, latest)

                # 生成信号原因
                reason = self._generate_reason(latest, score, signal_type)

                # 构建信号对象
                signal = RealtimeSignal(
                    code=code,
                    name=quote.name,
                    score=score,
                    signal=signal_type,
                    confidence=confidence,
                    reason=reason,
                    timestamp=datetime.now(),

                    # 技术指标
                    rsi_14=latest.get('rsi', 50.0),
                    macd=latest.get('macd', 0.0),
                    macd_signal=latest.get('macd_signal', 0.0),
                    macd_hist=latest.get('macd_hist', 0.0),
                    boll_position=latest.get('bb_position', 0.5),
                    kdj_k=latest.get('kdj_k', 50.0),
                    kdj_d=latest.get('kdj_d', 50.0),
                    kdj_j=latest.get('kdj_j', 50.0),

                    # 均线
                    ma5=latest.get('ma5', 0.0),
                    ma10=latest.get('ma10', 0.0),
                    ma20=latest.get('ma20', 0.0),
                    ma60=latest.get('ma60', 0.0),

                    # 动量
                    momentum_1d=latest.get('return_1d', 0.0) * 100,
                    momentum_5d=latest.get('momentum_5d', 0.0) * 100,

                    # 情绪
                    volume_ratio=latest.get('volume_ratio', 1.0),
                    turnover_rate=latest.get('turn', 0.0),
                )

                signals[code] = signal

            except Exception as e:
                logger.warning(f"生成信号失败 {code}: {e}")
                continue

        logger.info(f"生成信号: {len(signals)} 只")
        return signals

    def _append_quote(self, df: pd.DataFrame, quote: RealtimeQuote) -> pd.DataFrame:
        """将实时行情追加到历史数据"""
        df = df.copy()

        today = datetime.now().strftime('%Y-%m-%d')

        # 检查最后一条是否是今天
        if len(df) > 0:
            last_date = df['date'].iloc[-1]
            if last_date == today:
                # 更新最后一行
                df.iloc[-1, df.columns.get_loc('close')] = quote.current
                df.iloc[-1, df.columns.get_loc('high')] = max(df.iloc[-1]['high'], quote.high)
                df.iloc[-1, df.columns.get_loc('low')] = min(df.iloc[-1]['low'], quote.low)
                df.iloc[-1, df.columns.get_loc('volume')] = quote.volume
                return df

        # 添加新行 - 确保所有列都存在
        new_row = {}
        for col in df.columns:
            if col == 'date':
                new_row[col] = today
            elif col == 'code':
                new_row[col] = quote.code.replace('.', '_')
            elif col == 'open':
                new_row[col] = quote.open
            elif col == 'high':
                new_row[col] = quote.high
            elif col == 'low':
                new_row[col] = quote.low
            elif col == 'close':
                new_row[col] = quote.current
            elif col in ['volume', 'vol']:
                new_row[col] = quote.volume
            elif col == 'amount':
                new_row[col] = quote.amount
            else:
                # 其他列用前一行或0填充
                new_row[col] = df[col].iloc[-1] if col in df.columns and len(df) > 0 else None

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    def _calculate_score(self, latest: pd.Series, params: dict) -> float:
        """计算综合评分 - 复用 MultiStrategySignal 评分逻辑"""
        score = 50.0

        # 趋势因子 (默认权重 0.20)
        trend_weight = params.get('trend_weight', 0.20)
        if latest.get('ma_bull', 0) == 1:
            score += 20 * trend_weight * 2  # 均线多头 +10

        if latest.get('macd_hist', 0) > 0:
            score += 5 * trend_weight * 2  # MACD柱为正

        if latest.get('macd', 0) > latest.get('macd_signal', 0):
            score += 5 * trend_weight * 2  # MACD金叉

        # 均值回归因子 (默认权重 0.30)
        mean_rev_weight = params.get('mean_rev_weight', 0.30)
        rsi = latest.get('rsi', 50)
        if rsi < 30:
            score += 15 * mean_rev_weight * 2  # RSI超卖
        elif rsi < 40:
            score += 8 * mean_rev_weight * 2
        elif rsi > 70:
            score -= 10 * mean_rev_weight * 2  # RSI超买
        elif rsi > 60:
            score -= 5 * mean_rev_weight * 2

        bb_pos = latest.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 10 * mean_rev_weight * 2  # 布林下轨
        elif bb_pos > 0.8:
            score -= 10 * mean_rev_weight * 2  # 布林上轨

        # 动量因子 (默认权重 0.25)
        momentum_weight = params.get('momentum_weight', 0.25)
        mom_5d = latest.get('momentum_5d', 0)
        if mom_5d > 0.05:
            score += 10 * momentum_weight * 2  # 正动量
        elif mom_5d < -0.05:
            score -= 10 * momentum_weight * 2  # 负动量

        mom_20d = latest.get('momentum_20d', 0)
        if mom_20d > 0.10:
            score += 5 * momentum_weight * 2  # 20日强势
        elif mom_20d < -0.10:
            score -= 5 * momentum_weight * 2  # 20日弱势

        # 新因子 (默认权重 0.25)
        new_factor_weight = params.get('new_factor_weight', 0.25)
        volume_ratio = latest.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 5 * new_factor_weight * 2  # 放量
        elif volume_ratio < 0.5:
            score -= 5 * new_factor_weight * 2  # 缩量

        # KDJ 指标
        kdj_j = latest.get('kdj_j', 50)
        if kdj_j < 20:
            score += 5  # KDJ超卖
        elif kdj_j > 80:
            score -= 5  # KDJ超买

        # ATR 比率（波动率）
        atr_ratio = latest.get('atr_ratio', 0)
        if atr_ratio > 0.05:
            score += 3  # 高波动

        return min(max(score, 0), 100)

    def _calculate_confidence(self, score: float, latest: pd.Series) -> float:
        """计算置信度"""
        # 基于信号强度
        distance_from_mid = abs(score - 50) / 50

        # 基于指标一致性
        consistency = 1.0
        if latest.get('ma_bull', 0) == 1 and latest.get('macd_hist', 0) > 0:
            consistency = 1.2  # 多指标确认

        confidence = distance_from_mid * consistency
        return min(confidence, 1.0)

    def _generate_reason(self, latest: pd.Series, score: float, signal_type: str) -> str:
        """生成信号原因"""
        reasons = []

        if signal_type == 'BUY':
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                reasons.append(f"RSI超卖({rsi:.0f})")
            elif rsi < 40:
                reasons.append(f"RSI偏低({rsi:.0f})")

            if latest.get('macd_hist', 0) > 0:
                reasons.append("MACD柱为正")

            if latest.get('macd', 0) > latest.get('macd_signal', 0):
                reasons.append("MACD金叉")

            if latest.get('bb_position', 0.5) < 0.2:
                reasons.append("触及布林下轨")

            if latest.get('ma_bull', 0) == 1:
                reasons.append("均线多头排列")

            if latest.get('momentum_5d', 0) > 0.05:
                reasons.append("正动量")

        elif signal_type == 'SELL':
            rsi = latest.get('rsi', 50)
            if rsi > 70:
                reasons.append(f"RSI超买({rsi:.0f})")
            elif rsi > 60:
                reasons.append(f"RSI偏高({rsi:.0f})")

            if latest.get('macd_hist', 0) < 0:
                reasons.append("MACD柱为负")

            if latest.get('macd', 0) < latest.get('macd_signal', 0):
                reasons.append("MACD死叉")

            if latest.get('bb_position', 0.5) > 0.8:
                reasons.append("触及布林上轨")

        return " + ".join(reasons) if reasons else ""


def generate_signals_for_stock_pool(
    codes: List[str] = None,
    params: dict = None
) -> Dict[str, RealtimeSignal]:
    """
    为股票池生成信号

    Args:
        codes: 股票代码列表，None 时使用默认股票池
        params: 策略参数

    Returns:
        {code: RealtimeSignal}
    """
    if codes is None:
        codes = load_stock_pool()

    generator = RealtimeSignalGenerator(params)

    # 加载历史数据
    generator.load_historical_data(codes)

    # 获取实时行情
    quotes = generator.quote_provider.fetch_quotes(codes)

    # 生成信号
    return generator.generate_signals(quotes, params)


if __name__ == '__main__':
    # 测试信号生成
    print("测试实时信号生成...")

    codes = load_stock_pool()[:10]
    print(f"股票池: {codes}")

    signals = generate_signals_for_stock_pool(codes)

    print(f"\n生成信号 {len(signals)} 只:")
    for code, signal in sorted(signals.items(), key=lambda x: -x[1].score)[:5]:
        print(f"  {code}: {signal.name}")
        print(f"    评分: {signal.score:.1f}, 信号: {signal.signal}")
        print(f"    原因: {signal.reason}")
        print()
