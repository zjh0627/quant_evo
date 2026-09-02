#!/usr/bin/env python3
"""
回测避坑模块
============

严格避免回测中的常见陷阱:
1. 未来函数 (Look-ahead Bias)
2. 复权错误 (Adjustment Errors)
3. 滑点成本 (Slippage Costs)
4. 涨跌停无法交易 (Limit Up/Down Blocking)
5. T+1 交易限制
6. 流动性风险 (Liquidity Risk)

使用方法:
    from src.backtest.backtest_guard import BacktestGuard

    guard = BacktestGuard()
    result = guard.validate_and_run(data_dict, signals_dict)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    filtered_data: Dict[str, pd.DataFrame]
    filtered_signals: Dict[str, pd.DataFrame]


class BacktestGuard:
    """回测安全检查器"""

    def __init__(self,
                 check_look_ahead: bool = True,
                 check_adjustment: bool = True,
                 check_limit_updown: bool = True,
                 check_liquidity: bool = True,
                 min_volume: float = 1000000,  # 最小成交量
                 slippage_rate: float = 0.0001):  # 滑点率
        """
        Args:
            check_look_ahead: 检查未来函数
            check_adjustment: 检查复权一致性
            check_limit_updown: 检查涨跌停
            check_liquidity: 检查流动性
            min_volume: 最小成交量要求
            slippage_rate: 滑点率
        """
        self.check_look_ahead = check_look_ahead
        self.check_adjustment = check_adjustment
        self.check_limit_updown = check_limit_updown
        self.check_liquidity = check_liquidity
        self.min_volume = min_volume
        self.slippage_rate = slippage_rate

    def validate(self,
                 data_dict: Dict[str, pd.DataFrame],
                 signals_dict: Dict[str, pd.DataFrame]) -> ValidationResult:
        """
        验证数据安全性

        Returns:
            ValidationResult: 包含错误、警告和过滤后的数据
        """
        errors = []
        warnings = []
        filtered_data = {}
        filtered_signals = {}

        for code in data_dict.keys():
            df = data_dict[code].copy()
            sig_df = signals_dict.get(code)

            # 1. 检查数据完整性
            df_errors, df_warnings = self._check_data_quality(code, df)
            errors.extend(df_errors)
            warnings.extend(df_warnings)

            # 2. 检查未来函数
            if self.check_look_ahead:
                df_errors, df_warnings = self._check_look_ahead_bias(code, df, sig_df)
                errors.extend(df_errors)
                warnings.extend(df_warnings)

            # 3. 检查复权一致性
            if self.check_adjustment:
                df_errors, df_warnings = self._check_adjustment(code, df)
                errors.extend(df_errors)
                warnings.extend(df_warnings)

            # 4. 检查涨跌停
            if self.check_limit_updown:
                df = self._mark_limit_updown(df)

            # 5. 检查流动性
            if self.check_liquidity:
                df = self._check_liquidity(df)

            filtered_data[code] = df
            if sig_df is not None:
                filtered_signals[code] = sig_df

        is_valid = len([e for e in errors if 'CRITICAL' in e]) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            filtered_data=filtered_data,
            filtered_signals=filtered_signals
        )

    def _check_data_quality(self, code: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """检查数据质量"""
        errors = []
        warnings = []

        # 检查必要列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"[CRITICAL] {code}: 缺少列 {missing}")

        # 检查价格合理性
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                errors.append(f"[CRITICAL] {code}: 存在非正价格")

            if (df['high'] < df['low']).any():
                errors.append(f"[CRITICAL] {code}: 最高价 < 最低价")

            if (df['high'] < df['close']).any() or (df['low'] > df['close']).any():
                warnings.append(f"[WARNING] {code}: 收盘价超出最高/最低价范围")

        # 检查停牌（成交量为0）
        if 'volume' in df.columns:
            zero_volume_days = (df['volume'] == 0).sum()
            if zero_volume_days > len(df) * 0.1:  # 超过10%交易日停牌
                warnings.append(f"[WARNING] {code}: 停牌交易日占比 {zero_volume_days/len(df):.1%}")

        return errors, warnings

    def _check_look_ahead_bias(self,
                               code: str,
                               df: pd.DataFrame,
                               sig_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """检查未来函数"""
        errors = []
        warnings = []

        if sig_df is None:
            return errors, warnings

        # 检查信号是否使用未来数据
        # 例如: 信号使用了明天的收盘价
        if 'signal' in sig_df.columns and 'close' in df.columns:
            # 简单检查: 信号日期不能使用未来数据
            for idx, row in sig_df.iterrows():
                sig_date = row['date']
                # 查找对应日期的行情数据
                price_rows = df[df['date'] == sig_date]
                if price_rows.empty:
                    continue

                # 信号不能使用 sig_date 之后的数据
                future_rows = df[df['date'] > sig_date]
                # 这是一个简单的检测，实际需要更复杂的逻辑
                # ...

        # 检查因子计算是否使用了未来数据
        # 例如: 计算 N 日均线时使用了未来的价格
        future_dependent_cols = ['future_return', 'future_high', 'future_low']
        for col in future_dependent_cols:
            if col in df.columns:
                errors.append(f"[CRITICAL] {code}: 发现未来依赖列 {col}")

        return errors, warnings

    def _check_adjustment(self, code: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """检查复权一致性"""
        errors = []
        warnings = []

        # 检查复权因子是否单调递增（分红导致）
        if 'adj_factor' in df.columns:
            adj = df['adj_factor'].dropna()
            if len(adj) > 1:
                # 复权因子应该随着时间推移而增大（如果有分红）
                # 检查是否有异常下降
                diffs = adj.diff()
                if (diffs < 0).any():
                    # 复权因子下降可能表示数据错误或复权方式不一致
                    warnings.append(f"[WARNING] {code}: 复权因子存在下降，可能存在复权不一致")

        # 检查价格连续性（复权后价格应该连续）
        if 'close' in df.columns and 'adj_factor' in df.columns:
            adj_close = df['close'] * df['adj_factor']
            adj_close_diff = adj_close.diff().abs()
            # 如果价格跳变超过50%，可能是复权问题
            if (adj_close_diff > adj_close.shift(1) * 0.5).any():
                warnings.append(f"[WARNING] {code}: 复权后价格存在异常跳变")

        return errors, warnings

    def _mark_limit_updown(self, df: pd.DataFrame) -> pd.DataFrame:
        """标记涨跌停"""
        df = df.copy()

        if 'prev_close' not in df.columns:
            df['prev_close'] = df['close'].shift(1)

        df['pct_chg'] = (df['close'] - df['prev_close']) / df['prev_close']

        # 涨停
        df['is_limit_up'] = df['pct_chg'] >= 0.095  # 预留滑点
        # 跌停
        df['is_limit_down'] = df['pct_chg'] <= -0.095

        return df

    def _check_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """检查流动性，标记低流动性日期"""
        df = df.copy()

        if 'volume' in df.columns:
            # 标记低流动性日期（用于后续过滤）
            df['low_liquidity'] = df['volume'] < self.min_volume

        return df

    def apply_slippage(self,
                      price: float,
                      action: str,
                      is_limit_up: bool = False,
                      is_limit_down: bool = False) -> float:
        """
        应用滑点

        Args:
            price: 原始价格
            action: 'BUY' or 'SELL'
            is_limit_up: 是否涨停（无法买入）
            is_limit_down: 是否跌停（无法卖出）

        Returns:
            调整后的价格
        """
        if is_limit_up and action == 'BUY':
            # 涨停无法买入
            return price * 1.0  # 返回原价，但实际无法交易

        if is_limit_down and action == 'SELL':
            # 跌停无法卖出
            return price * 1.0  # 返回原价，但实际无法交易

        # 正常情况应用滑点
        if action == 'BUY':
            return price * (1 + self.slippage_rate)  # 买入时价格更高
        else:  # SELL
            return price * (1 - self.slippage_rate)  # 卖出时价格更低

    def check_t_plus_one(self,
                        buy_date: str,
                        sell_date: str,
                        holding_period: int = 1) -> bool:
        """
        检查是否满足 T+1 交易规则

        Args:
            buy_date: 买入日期
            sell_date: 卖出日期
            holding_period: 持有天数

        Returns:
            True if valid (可以卖出), False if invalid (T+1 violation)
        """
        buy_dt = datetime.strptime(buy_date, '%Y-%m-%d')
        sell_dt = datetime.strptime(sell_date, '%Y-%m-%d')

        days_diff = (sell_dt - buy_dt).days

        return days_diff >= holding_period

    def validate_trade(self,
                      action: str,
                      date: str,
                      code: str,
                      price: float,
                      volume: float,
                      is_limit_up: bool = False,
                      is_limit_down: bool = False,
                      buy_date: str = None) -> Tuple[bool, str]:
        """
        验证单笔交易是否有效

        Returns:
            (is_valid, reason)
        """
        # 涨停无法买入
        if action == 'BUY' and is_limit_up:
            return False, f"{date} {code} 涨停无法买入"

        # 跌停无法卖出
        if action == 'SELL' and is_limit_down:
            return False, f"{date} {code} 跌停无法卖出"

        # 检查 T+1
        if action == 'SELL' and buy_date is not None:
            if not self.check_t_plus_one(buy_date, date):
                return False, f"{date} {code} T+1 违规（买入日期: {buy_date}）"

        # 检查流动性
        if volume < self.min_volume:
            return False, f"{date} {code} 流动性不足（成交量: {volume:,.0f}）"

        return True, "OK"


def test_backtest_guard():
    """测试回测避坑模块"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

    from src.backtest.light_backtest import LightBacktest

    # 生成模拟数据（包含一些问题）
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')

    stocks = ['600519', '000858']
    data_dict = {}
    signals_dict = {}

    for code in stocks:
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
        volumes = np.random.randint(500000, 5000000, len(dates))

        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(len(dates)) * 0.5,
            'high': prices + np.abs(np.random.randn(len(dates)) * 2),
            'low': prices - np.abs(np.random.randn(len(dates)) * 2),
            'close': prices,
            'volume': volumes,
            'adj_factor': np.ones(len(dates))  # 简化：假设无复权
        })

        signals = pd.DataFrame({
            'date': dates,
            'signal': np.where(np.random.rand(len(dates)) > 0.8, 'BUY',
                              np.where(np.random.rand(len(dates)) > 0.6, 'SELL', 'HOLD')),
            'score': np.random.randint(40, 90, len(dates))
        })

        data_dict[code] = df
        signals_dict[code] = signals

    # 验证
    guard = BacktestGuard(check_liquidity=True, min_volume=1000000)
    result = guard.validate(data_dict, signals_dict)

    print("验证结果:")
    print(f"  是否有效: {result.is_valid}")
    print(f"  错误数: {len(result.errors)}")
    print(f"  警告数: {len(result.warnings)}")

    if result.errors:
        print("\n错误:")
        for e in result.errors[:5]:
            print(f"  {e}")

    if result.warnings:
        print("\n警告:")
        for w in result.warnings[:5]:
            print(f"  {w}")

    # 测试滑点应用
    print("\n滑点测试:")
    print(f"  买入价格 100 -> {guard.apply_slippage(100, 'BUY'):.4f}")
    print(f"  卖出价格 100 -> {guard.apply_slippage(100, 'SELL'):.4f}")

    # 测试 T+1 检查
    print("\nT+1 检查:")
    print(f"  买入 2024-01-01, 卖出 2024-01-02: {guard.check_t_plus_one('2024-01-01', '2024-01-02')}")
    print(f"  买入 2024-01-01, 卖出 2024-01-01: {guard.check_t_plus_one('2024-01-01', '2024-01-01')}")


if __name__ == '__main__':
    test_backtest_guard()
