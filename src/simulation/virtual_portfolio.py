#!/usr/bin/env python3
"""
模拟盘系统
==========

每天运行策略生成交易信号，模拟真实资金账户

功能：
1. 每日信号生成
2. 模拟账户（持仓、现金）
3. 资金曲线记录
4. 定期推送到飞书
"""

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
PORTFOLIO_FILE = f'{DATA_DIR}/cache/virtual_portfolio.json'
POSITION_FILE = f'{DATA_DIR}/cache/positions.json'


@dataclass
class Position:
    """持仓"""
    code: str
    name: str
    shares: int
    entry_date: str
    entry_price: float
    current_price: float
    pnl_pct: float


@dataclass
class Portfolio:
    """模拟账户"""
    cash: float
    total_value: float
    positions: List[dict]
    last_update: str


def load_positions() -> List[Position]:
    """加载持仓"""
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            data = json.load(f)
            return [Position(**p) for p in data]
    return []


def save_positions(positions: List[Position]):
    """保存持仓"""
    os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
    with open(POSITION_FILE, 'w') as f:
        json.dump([asdict(p) for p in positions], f, ensure_ascii=False, indent=2)


def load_portfolio() -> Portfolio:
    """加载账户"""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            return Portfolio(**data)
    # 默认账户
    return Portfolio(
        cash=1000000,
        total_value=1000000,
        positions=[],
        last_update=datetime.now().strftime('%Y-%m-%d')
    )


def save_portfolio(portfolio: Portfolio):
    """保存账户"""
    portfolio.last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(asdict(portfolio), f, ensure_ascii=False, indent=2)


def get_latest_price(code: str) -> Optional[float]:
    """获取最新价格"""
    code_fmt = code.replace('.', '_')
    filepath = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if len(df) > 0:
            return float(df['close'].iloc[-1])
    return None


def calculate_portfolio_value(positions: List[Position]) -> float:
    """计算持仓价值"""
    total = 0
    for pos in positions:
        price = get_latest_price(pos.code)
        if price:
            pos.current_price = price
            pos.pnl_pct = (price - pos.entry_price) / pos.entry_price
            total += pos.shares * price
        else:
            total += pos.shares * pos.current_price
    return total


def generate_trading_signals() -> Dict[str, dict]:
    """生成交易信号"""
    # 加载股票池
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:50]  # 取50只
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

    signals = {}

    for code in codes:
        code_fmt = code.replace('.', '_')
        filepath = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'

        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)
        if len(df) < 60:
            continue

        # 计算指标
        close = df['close'].values
        ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        ma60 = pd.Series(close).rolling(60).mean().iloc[-1]

        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 50
        rsi = 100 - (100 / (1 + rs)) if not pd.isna(rs) else 50

        # 动量
        momentum = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

        # 信号
        signal = 'HOLD'
        score = 50

        if ma20 > ma60 and rsi < 65:  # 上升趋势
            signal = 'BUY'
            score = 70 + int(rsi < 40) * 10 + int(momentum > 0.03) * 10
        elif ma20 < ma60 or rsi > 75:  # 下降趋势
            signal = 'SELL'
            score = 70 + int(rsi > 80) * 10

        price = close[-1]
        signals[code] = {
            'signal': signal,
            'score': min(score, 100),
            'price': price,
            'rsi': rsi,
            'momentum': momentum,
            'ma20': ma20,
            'ma60': ma60
        }

    return signals


def run_daily_trade():
    """每日交易"""
    print("=" * 60)
    print("模拟盘每日交易")
    print("=" * 60)

    # 加载账户
    portfolio = load_portfolio()
    positions = load_positions()
    print(f"当前现金: {portfolio.cash:,.2f}")
    print(f"当前持仓: {len(positions)} 只")

    # 生成信号
    signals = generate_trading_signals()
    buy_signals = {k: v for k, v in signals.items() if v['signal'] == 'BUY'}
    sell_signals = {k: v for k, v in signals.items() if v['signal'] == 'SELL'}

    print(f"\n买入信号: {len(buy_signals)} 只")
    print(f"卖出信号: {len(sell_signals)} 只")

    # 执行卖出
    for pos in positions[:]:  # 复制列表
        if pos.code in sell_signals:
            sig = sell_signals[pos.code]
            price = sig['price']
            revenue = pos.shares * price * 0.999  # 扣除手续费
            portfolio.cash += revenue

            print(f"卖出 {pos.name} ({pos.code}): {pos.shares}股 @ {price:.2f} = {revenue:,.2f}")
            positions.remove(pos)

    # 执行买入
    if buy_signals and len(positions) < 10:
        # 按评分排序
        sorted_buys = sorted(buy_signals.items(), key=lambda x: x[1]['score'], reverse=True)

        for code, sig in sorted_buys[:10 - len(positions)]:
            if portfolio.cash < sig['price'] * 100:
                continue

            # 买入1手
            shares = 100
            cost = shares * sig['price'] * 1.0004  # 手续费
            if cost <= portfolio.cash:
                portfolio.cash -= cost
                name = get_stock_name(code)
                positions.append(Position(
                    code=code,
                    name=name,
                    shares=shares,
                    entry_date=datetime.now().strftime('%Y-%m-%d'),
                    entry_price=sig['price'],
                    current_price=sig['price'],
                    pnl_pct=0
                ))
                print(f"买入 {name} ({code}): {shares}股 @ {sig['price']:.2f}")

    # 更新持仓价值
    positions_value = calculate_portfolio_value(positions)
    portfolio.total_value = portfolio.cash + positions_value

    # 保存
    save_positions(positions)
    save_portfolio(portfolio)

    print(f"\n账户总价值: {portfolio.total_value:,.2f}")
    print(f"现金: {portfolio.cash:,.2f}")
    print(f"持仓: {positions_value:,.2f}")

    return portfolio, positions, signals


def get_stock_name(code: str) -> str:
    """获取股票名称"""
    names = {
        'sh.600519': '贵州茅台',
        'sh.600036': '招商银行',
        'sh.601318': '中国平安',
        'sh.000001': '上证指数',
        'sz.000002': '万科A',
        'sz.002594': '比亚迪',
        'sh.600276': '恒瑞医药',
        'sh.600887': '伊利股份',
        'sz.300750': '宁德时代',
        'sh.601888': '中国中免',
        'sh.688012': '中微公司',
        'sh.603986': '兆易创新',
        'sz.300502': '新易盛',
        'sz.300308': '中际旭创',
        'sh.600584': '长电科技',
        'sz.000021': '深科技',
        'sz.002409': '雅克科技',
        'sz.300408': '三环集团',
    }
    return names.get(code, code)


if __name__ == '__main__':
    portfolio, positions, signals = run_daily_trade()