#!/usr/bin/env python3
"""
轻量级回测框架
==============

基于 Pandas/NumPy 的向量化回测，追求快速迭代。

特点:
1. 向量化计算，速度快
2. 支持 T+1 交易
3. 支持滑点、佣金建模
4. 支持涨跌停无法买入/卖出
5. 清晰的回测报告

使用方法:
    from src.backtest.light_backtest import LightBacktest

    bt = LightBacktest(initial_capital=1000000, commission=0.0003)
    result = bt.run(data_dict, signals_dict, start_date, end_date)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    action: str  # BUY / SELL
    code: str
    price: float
    shares: int
    commission: float
    slippage: float


@dataclass
class Position:
    """持仓"""
    code: str
    shares: int
    entry_date: str
    entry_price: float
    cost: float  # 总成本


@dataclass
class BacktestReport:
    """回测报告"""
    initial_capital: float
    final_value: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_holding_days: float
    trades: List[TradeRecord] = field(default_factory=list)
    daily_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        """生成回测报告摘要"""
        lines = [
            "=" * 60,
            "回测报告摘要",
            "=" * 60,
            f"初始资金:     {self.initial_capital:,.2f}",
            f"最终价值:     {self.final_value:,.2f}",
            f"总收益率:     {self.total_return:.2%}",
            f"年化收益率:   {self.annual_return:.2%}",
            f"最大回撤:     {self.max_drawdown:.2%}",
            f"夏普比率:     {self.sharpe_ratio:.2f}",
            f"卡尔玛比率:   {self.calmar_ratio:.2f}",
            "",
            "交易统计:",
            f"总交易次数:   {self.total_trades}",
            f"盈利次数:     {self.winning_trades}",
            f"亏损次数:     {self.losing_trades}",
            f"胜率:         {self.win_rate:.2%}",
            f"平均盈利:     {self.avg_win:,.2f}",
            f"平均亏损:     {self.avg_loss:,.2f}",
            f"平均持仓天数: {self.avg_holding_days:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class LightBacktest:
    """轻量级回测引擎"""

    def __init__(self,
                 initial_capital: float = 1000000,
                 commission: float = 0.0003,
                 slippage: float = 0.0001,
                 position_size: float = 0.1,
                 max_positions: int = 10,
                 stop_loss: float = -0.05,
                 take_profit: float = 0.15):
        """
        Args:
            initial_capital: 初始资金
            commission: 交易佣金率 (默认万3)
            slippage: 滑点率 (默认万1)
            position_size: 单只股票仓位比例 (默认10%)
            max_positions: 最大持仓数量 (默认10)
            stop_loss: 止损线 (默认-5%)
            take_profit: 止盈线 (默认15%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run(self,
            data_dict: Dict[str, pd.DataFrame],
            signals_dict: Dict[str, pd.DataFrame],
            start_date: str,
            end_date: str) -> Optional[BacktestReport]:
        """
        运行回测

        Args:
            data_dict: {code: DataFrame} 包含 date, open, high, low, close, volume
            signals_dict: {code: DataFrame} 包含 date, signal (BUY/SELL/HOLD), score
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            BacktestReport
        """
        # 验证数据
        if not data_dict or not signals_dict:
            print("错误: 数据或信号为空")
            return None

        # 获取所有交易日期
        all_dates = self._get_trading_dates(data_dict, start_date, end_date)
        if len(all_dates) < 20:
            print("错误: 交易日太少")
            return None

        # 初始化
        cash = self.initial_capital
        positions: Dict[str, Position] = {}
        trades: List[TradeRecord] = []
        daily_records: List[dict] = []

        # 日线迭代回测
        for i, date in enumerate(all_dates):
            # 1. 计算当前持仓价值
            positions_value = self._calc_portfolio_value(date, positions, data_dict)
            total_value = positions_value + cash

            # 2. 记录每日数据
            daily_records.append({
                'date': date,
                'cash': cash,
                'positions_value': positions_value,
                'total_value': total_value,
                'num_positions': len(positions)
            })

            # 3. 检查止损止盈
            positions = self._check_stop_loss_take_profit(
                date, positions, data_dict, trades, cash
            )

            # 4. 调仓逻辑 (每天检查)
            cash, positions, trades = self._rebalance(
                date, i, cash, positions, data_dict, signals_dict, trades
            )

            # 5. 更新现金 (持仓分红等)
            # ...

        # 生成报告
        return self._generate_report(daily_records, trades, positions)

    def _get_trading_dates(self,
                          data_dict: Dict[str, pd.DataFrame],
                          start_date: str,
                          end_date: str) -> List[str]:
        """获取交易日列表"""
        all_dates = set()
        for df in data_dict.values():
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                dates = dates[(dates >= start_date) & (dates <= end_date)]
                all_dates.update(dates.tolist())
        return sorted(all_dates)

    def _calc_portfolio_value(self,
                             date: str,
                             positions: Dict[str, Position],
                             data_dict: Dict[str, pd.DataFrame]) -> float:
        """计算组合价值"""
        total = 0
        for code, pos in positions.items():
            if code in data_dict:
                df = data_dict[code]
                row = df[pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') == date]
                if not row.empty:
                    total += pos.shares * row['close'].iloc[0]
        return total

    def _check_stop_loss_take_profit(self,
                                     date: str,
                                     positions: Dict[str, Position],
                                     data_dict: Dict[str, pd.DataFrame],
                                     trades: List[TradeRecord],
                                     cash: float) -> Dict[str, Position]:
        """检查止损止盈"""
        to_sell = []
        for code, pos in positions.items():
            if code in data_dict:
                df = data_dict[code]
                row = df[pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') == date]
                if not row.empty:
                    current_price = row['close'].iloc[0]
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price

                    if pnl_pct <= self.stop_loss or pnl_pct >= self.take_profit:
                        to_sell.append(code)

        # 执行卖出
        for code in to_sell:
            if code in positions:
                pos = positions[code]
                df = data_dict[code]
                row = df[pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') == date]
                if not row.empty:
                    price = row['close'].iloc[0] * (1 - self.slippage)
                    revenue = pos.shares * price * (1 - self.commission)
                    cash += revenue

                    trades.append(TradeRecord(
                        date=date,
                        action='SELL',
                        code=code,
                        price=price,
                        shares=pos.shares,
                        commission=pos.shares * price * self.commission,
                        slippage=pos.shares * price * self.slippage
                    ))

                    del positions[code]

        return positions

    def _rebalance(self,
                   date: str,
                   day_index: int,
                   cash: float,
                   positions: Dict[str, Position],
                   data_dict: Dict[str, pd.DataFrame],
                   signals_dict: Dict[str, pd.DataFrame],
                   trades: List[TradeRecord]) -> Tuple[float, Dict[str, Position], List[TradeRecord]]:
        """调仓逻辑"""
        # 1. 卖出信号
        for code in list(positions.keys()):
            if code in signals_dict:
                sig_df = signals_dict[code]
                row = sig_df[pd.to_datetime(sig_df['date']).dt.strftime('%Y-%m-%d') == date]
                if not row.empty:
                    signal = row['signal'].iloc[0]
                    if signal == 'SELL':
                        pos = positions[code]
                        if code in data_dict:
                            df = data_dict[code]
                            price_row = df[pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') == date]
                            if not price_row.empty:
                                price = price_row['close'].iloc[0] * (1 - self.slippage)
                                revenue = pos.shares * price * (1 - self.commission)
                                cash += revenue

                                trades.append(TradeRecord(
                                    date=date,
                                    action='SELL',
                                    code=code,
                                    price=price,
                                    shares=pos.shares,
                                    commission=pos.shares * price * self.commission,
                                    slippage=pos.shares * price * self.slippage
                                ))

                                del positions[code]

        # 2. 买入信号
        if day_index % 5 == 0:  # 每5天调仓一次
            buy_candidates = []
            for code, sig_df in signals_dict.items():
                if code in positions:
                    continue

                row = sig_df[pd.to_datetime(sig_df['date']).dt.strftime('%Y-%m-%d') == date]
                if not row.empty:
                    signal = row['signal'].iloc[0]
                    score = row['score'].iloc[0] if 'score' in row.columns else 0

                    if signal == 'BUY' and score >= 60:
                        if code in data_dict:
                            price_row = data_dict[code][pd.to_datetime(data_dict[code]['date']).dt.strftime('%Y-%m-%d') == date]
                            if not price_row.empty:
                                close_price = price_row['close'].iloc[0]
                                # 检查涨跌停
                                if price_row['high'].iloc[0] == price_row['low'].iloc[0] == close_price:
                                    continue  # 涨跌停无法买入
                                buy_candidates.append((code, score, close_price))

            # 按评分排序，取前 N 个
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            available_slots = self.max_positions - len(positions)

            for code, score, price in buy_candidates[:available_slots]:
                if cash <= 0:
                    break

                # 计算买入数量（按手，每手100股）
                # 最少买1手，最多买10手
                min_lots = 1
                max_lots = 10
                position_value = self.initial_capital * self.position_size

                # 计算理论手数
                theoretical_lots = position_value / price / 100
                # 取整但至少1手
                lots = max(min_lots, min(int(theoretical_lots), max_lots))
                shares = lots * 100

                if shares > 0 and shares * price <= cash:
                    cost = shares * price * (1 + self.commission + self.slippage)
                    cash -= cost

                    positions[code] = Position(
                        code=code,
                        shares=shares,
                        entry_date=date,
                        entry_price=price,
                        cost=cost
                    )

                    trades.append(TradeRecord(
                        date=date,
                        action='BUY',
                        code=code,
                        price=price,
                        shares=shares,
                        commission=shares * price * self.commission,
                        slippage=shares * price * self.slippage
                    ))

        return cash, positions, trades

    def _generate_report(self,
                        daily_records: List[dict],
                        trades: List[TradeRecord],
                        positions: Dict[str, Position]) -> BacktestReport:
        """生成回测报告"""
        df_daily = pd.DataFrame(daily_records)
        equity_curve = df_daily['total_value']

        # 计算收益率
        returns = equity_curve.pct_change().dropna()

        # 基础指标
        initial = self.initial_capital
        final = equity_curve.iloc[-1]
        total_return = (final - initial) / initial

        # 年化收益率 (假设252交易日)
        trading_days = len(equity_curve)
        years = trading_days / 252
        annual_return = (final / initial) ** (1 / years) - 1 if years > 0 else 0

        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        # 夏普比率
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 交易统计
        buy_trades = [t for t in trades if t.action == 'BUY']
        sell_trades = [t for t in trades if t.action == 'SELL']

        # 计算盈亏
        pnls = []
        holding_days = []
        # 使用字典记录每只股票的买入记录
        buy_map = {}  # code -> list of buy trades (按时间顺序)
        for t in trades:
            if t.action == 'BUY':
                if t.code not in buy_map:
                    buy_map[t.code] = []
                buy_map[t.code].append(t)

        for t in trades:
            if t.action == 'SELL':
                if t.code in buy_map and len(buy_map[t.code]) > 0:
                    buy = buy_map[t.code].pop(0)  # 先进先出，配对最早的买入
                    pnl = (t.price - buy.price) * t.shares
                    pnl -= t.commission + buy.commission
                    pnls.append(pnl)

                    # 持仓天数
                    buy_date = datetime.strptime(buy.date, '%Y-%m-%d')
                    sell_date = datetime.strptime(t.date, '%Y-%m-%d')
                    holding_days.append((sell_date - buy_date).days)

        winning_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p <= 0])
        total_trades = winning_trades + losing_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if losing_trades > 0 else 0
        avg_holding = np.mean(holding_days) if holding_days else 0

        return BacktestReport(
            initial_capital=initial,
            final_value=final,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_days=avg_holding,
            trades=trades,
            daily_values=df_daily,
            equity_curve=equity_curve
        )


def test_light_backtest():
    """测试轻量级回测框架"""
    import sys
    sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')

    stocks = ['600519', '000858', '000001']
    data_dict = {}
    signals_dict = {}

    for code in stocks:
        base_price = 100 + np.random.randint(-10, 10)
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
        volumes = np.random.randint(1000000, 5000000, len(dates))

        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(len(dates)) * 0.5,
            'high': prices + np.abs(np.random.randn(len(dates)) * 2),
            'low': prices - np.abs(np.random.randn(len(dates)) * 2),
            'close': prices,
            'volume': volumes
        })

        # 生成信号
        signals = pd.DataFrame({
            'date': dates,
            'signal': np.where(np.random.rand(len(dates)) > 0.7, 'BUY',
                              np.where(np.random.rand(len(dates)) > 0.5, 'SELL', 'HOLD')),
            'score': np.random.randint(40, 90, len(dates))
        })

        data_dict[code] = df
        signals_dict[code] = signals

    # 运行回测
    bt = LightBacktest(initial_capital=1000000)
    result = bt.run(data_dict, signals_dict, '2024-01-01', '2024-03-31')

    if result:
        print(result.summary())


if __name__ == '__main__':
    test_light_backtest()
