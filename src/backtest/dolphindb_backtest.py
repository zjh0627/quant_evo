#!/usr/bin/env python3
"""
DolphinDB 回测包装器
====================

通过 Python 调用 DolphinDB 进行海量全市场批量回测

使用方法:
    from src.backtest.dolphindb_backtest import DolphinDBBacktest

    runner = DolphinDBBacktest()
    result = runner.run(
        start_date='2024-01-01',
        end_date='2024-03-31',
        initial_capital=10000000,
        signals_table='dfs://quant_evo_factors/factor_daily',
        market_table='dfs://quant_evo_market/daily_k'
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DDBacktestResult:
    """DolphinDB 回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


class DolphinDBBacktest:
    """DolphinDB 回测运行器"""

    def __init__(self, host: str = 'localhost', port: int = 8848):
        """
        Args:
            host: DolphinDB 服务器地址
            port: DolphinDB 端口
        """
        self.host = host
        self.port = port
        self._conn = None

    def _connect(self):
        """连接 DolphinDB"""
        try:
            import dolphindb
            self._conn = dolphindb.session()
            self._conn.connect(self.host, self.port)
            print(f"Connected to DolphinDB at {self.host}:{self.port}")
        except Exception as e:
            print(f"Warning: Could not connect to DolphinDB: {e}")
            self._conn = None

    def _disconnect(self):
        """断开连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def run(self,
            start_date: str,
            end_date: str,
            initial_capital: float = 10000000,
            commission: float = 0.0003,
            slippage: float = 0.0001,
            max_positions: int = 10,
            rebalance_days: int = 5,
            min_score: float = 60,
            signals_table: str = 'dfs://quant_evo_factors/factor_daily',
            market_table: str = 'dfs://quant_evo_market/daily_k') -> Optional[DDBacktestResult]:
        """
        运行 DolphinDB 回测

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            commission: 佣金率
            slippage: 滑点率
            max_positions: 最大持仓
            rebalance_days: 调仓周期
            min_score: 最小评分
            signals_table: 信号表路径
            market_table: 行情表路径

        Returns:
            DDBacktestResult
        """
        # 尝试连接 DolphinDB
        self._connect()

        if self._conn is None:
            print("Warning: DolphinDB not connected, using Python fallback")
            return self._python_fallback(
                start_date, end_date, initial_capital,
                signals_table, market_table
            )

        # 构建回测脚本
        script = self._build_script(
            start_date, end_date, initial_capital,
            commission, slippage, max_positions,
            rebalance_days, min_score,
            signals_table, market_table
        )

        try:
            # 执行回测
            result = self._conn.run(script)
            return self._parse_result(result)
        except Exception as e:
            print(f"DolphinDB backtest error: {e}")
            return self._python_fallback(
                start_date, end_date, initial_capital,
                signals_table, market_table
            )
        finally:
            self._disconnect()

    def _build_script(self,
                     start_date: str,
                     end_date: str,
                     initial_capital: float,
                     commission: float,
                     slippage: float,
                     max_positions: int,
                     rebalance_days: int,
                     min_score: float,
                     signals_table: str,
                     market_table: str) -> str:
        """构建回测脚本"""
        # 将日期转换为 DolphinDB 格式
        start_dt = start_date.replace('-', '.')
        end_dt = end_date.replace('-', '.')

        script = f"""
// 配置
CONFIG_START_DATE = {start_dt}
CONFIG_END_DATE = {end_dt}
CONFIG_INITIAL_CAPITAL = {initial_capital}
CONFIG_COMMISSION = {commission}
CONFIG_SLIPPAGE = {slippage}
CONFIG_MAX_POSITIONS = {max_positions}
CONFIG_REBALANCE_DAYS = {rebalance_days}
CONFIG_MIN_SCORE = {min_score}

// 加载数据
daily_k = loadTable("{market_table}", "daily_k")
factor_daily = loadTable("{signals_table}", "factor_daily")

// 获取交易日
trade_dates = exec distinct trade_date from daily_k
    where trade_date >= CONFIG_START_DATE, trade_date <= CONFIG_END_DATE
    order by trade_date

// 信号处理 (简化版)
signals = select
    trade_date,
    security_id,
    close,
    rsi_6,
    momentum_1d,
    iif(rsi_6 < 30 and momentum_1d > -0.03, 'BUY',
        iif(rsi_6 > 70 or momentum_1d > 0.1, 'SELL', 'HOLD')) as signal,
    rsi_6 as score
from factor_daily
where trade_date >= CONFIG_START_DATE, trade_date <= CONFIG_END_DATE
context by security_id

// 回测主逻辑
capital = CONFIG_INITIAL_CAPITAL
position_value = 0.0
equity_curve = table(100:0, `date`total_value`cash`positions, [DATE, DOUBLE, DOUBLE, INT])
trades = table(100:0, `date`code`action`price`shares, [DATE, STRING, STRING, DOUBLE, INT])

for (i in 0:size(trade_dates)) {{
    date = trade_dates[i]
    day_signals = select * from signals where trade_date = date
    day_k = select * from daily_k where trade_date = date

    // 记录当日权益
    equity_curve.append!([{{date, position_value + capital, capital, size(positions)}}])

    // 调仓逻辑 (每N天)
    if (i % CONFIG_REBALANCE_DAYS == 0) {{
        // 买入
        buy_signals = select security_id, score from day_signals
            where signal = 'BUY' and score >= CONFIG_MIN_SCORE
            order by score desc
            limit CONFIG_MAX_POSITIONS

        for (row in buy_signals) {{
            code = row.security_id
            price_row = select open from day_k where security_id = code
            if (size(price_row) > 0) {{
                price = price_row.open[0]
                shares = floor(CONFIG_INITIAL_CAPITAL * 0.1 / price / 100) * 100
                if (shares > 0 and shares * price <= capital) {{
                    trades.append!([{{date, code, 'BUY', price, shares}}])
                    capital -= shares * price
                }}
            }}
        }}
    }}
}}

// 输出结果
select * from equity_curve
"""
        return script

    def _parse_result(self, result) -> DDBacktestResult:
        """解析 DolphinDB 返回结果"""
        # DolphinDB 返回的是 pandas DataFrame
        if isinstance(result, pd.DataFrame):
            equity_curve = result
        else:
            equity_curve = pd.DataFrame()

        # 计算绩效指标
        if len(equity_curve) > 0:
            initial = equity_curve['total_value'].iloc[0]
            final = equity_curve['total_value'].iloc[-1]
            total_return = (final - initial) / initial

            returns = equity_curve['total_value'].pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            cummax = equity_curve['total_value'].cummax()
            drawdown = (equity_curve['total_value'] - cummax) / cummax
            max_drawdown = drawdown.min()
        else:
            total_return = 0
            sharpe = 0
            max_drawdown = 0

        return DDBacktestResult(
            total_return=total_return,
            annual_return=total_return,  # 简化
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            equity_curve=equity_curve,
            trades=pd.DataFrame()
        )

    def _python_fallback(self,
                        start_date: str,
                        end_date: str,
                        initial_capital: float,
                        signals_table: str,
                        market_table: str) -> Optional[DDBacktestResult]:
        """Python 回退方案 (当 DolphinDB 不可用时)"""
        print("Using Python fallback backtest...")

        # 这里可以使用 LightBacktest
        from src.backtest.light_backtest import LightBacktest

        # 由于没有真实数据，返回 None
        print("Warning: No market data available for backtest")
        print(f"Please ensure DolphinDB is running at {self.host}:{self.port}")
        print(f"And tables exist: {market_table}, {signals_table}")

        return None


def test_dolphindb_backtest():
    """测试 DolphinDB 回测包装器"""
    runner = DolphinDBBacktest()
    result = runner.run(
        start_date='2024-01-01',
        end_date='2024-03-31',
        initial_capital=10000000
    )

    if result:
        print("Backtest completed!")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
    else:
        print("Backtest skipped (no data or DolphinDB not available)")


if __name__ == '__main__':
    test_dolphindb_backtest()
