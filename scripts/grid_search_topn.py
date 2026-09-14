#!/usr/bin/env python3
"""
top_n Grid Search - 持仓分散化参数优化
=======================================

测试不同的top_n值(1-10)，找出收益率和最大回撤的最佳平衡点

运行方式:
    python scripts/grid_search_topn.py
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.backtest.light_backtest import LightBacktest
from src.data.data_pipeline import DataPipeline
from src.factors.technical import FactorCalculator

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'

# 测试参数范围
TOP_N_RANGE = range(1, 11)  # 1-10
POSITION_SIZE = 0.10  # 固定10%单只仓位

# 固定其他参数
FIXED_PARAMS = {
    'buy_threshold': 55,
    'sell_threshold': 42,
    'stop_loss': 0.10,
    'take_profit': 0.20,
    'position_size': POSITION_SIZE,
    'rsi_oversold': 30,
    'rsi_overbought': 65,
    'vol_threshold': 2.0,
}


def run_backtest_for_topn(top_n: int, stock_codes: list) -> dict:
    """为指定top_n运行回测"""

    pipeline = DataPipeline()
    all_data = {}

    # 加载数据
    for code in stock_codes:
        df = pipeline.load(code)
        if df is not None and len(df) >= 60:
            calc = FactorCalculator()
            factors = calc.calculate_all_factors(df)
            if not factors.empty:
                all_data[code] = factors

    if len(all_data) < 10:
        return None

    # 简单回测逻辑
    params = FIXED_PARAMS.copy()
    params['top_n'] = top_n

    cash = 2000000.0
    positions = {}  # code -> {shares, entry_price, entry_date}
    trades = []
    equity_curve = []

    dates = sorted(set(df['date'].iloc[60:] for df in all_data.values()))
    start_date = dates[0] if dates else None
    end_date = dates[-1] if dates else None

    # 只测试最近252个交易日（约1年）
    if len(dates) > 252:
        dates = dates[-252:]
        start_date = dates[0]

    for date in dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d') if isinstance(date, str) else str(date)

        # 获取当日信号
        signals = []
        for code, df in all_data.items():
            if date not in df['date'].values:
                continue
            idx = df[df['date'] == date].index[0]
            if idx < 60:
                continue

            row = df.iloc[idx]
            score = row.get('composite_score', 50)

            signals.append({
                'code': code,
                'score': score,
                'price': row['close'],
                'date': date_str
            })

        # 按评分排序
        signals.sort(key=lambda x: -x['score'])

        # 买入信号
        buy_signals = [s for s in signals if s['score'] > params['buy_threshold']][:top_n]

        # 卖出信号（持仓中超卖的）
        for code in list(positions.keys()):
            if code not in [s['code'] for s in signals]:
                continue
            sig = next(s for s in signals if s['code'] == code)
            if sig['score'] < params['sell_threshold']:
                # 卖出
                shares = positions[code]['shares']
                entry = positions[code]['entry_price']
                current = sig['price']
                pnl = (current - entry) / entry
                trades.append({
                    'date': date_str,
                    'code': code,
                    'action': 'SELL',
                    'pnl': pnl,
                    'reason': 'signal_sell' if pnl >= 0 else 'stop_loss'
                })
                cash += shares * current
                del positions[code]

        # 止损检查
        for code in list(positions.keys()):
            shares = positions[code]['shares']
            entry = positions[code]['entry_price']
            current = next((s['price'] for s in signals if s['code'] == code), None)
            if current:
                pnl = (current - entry) / entry
                if pnl <= -params['stop_loss']:
                    trades.append({
                        'date': date_str,
                        'code': code,
                        'action': 'SELL',
                        'pnl': pnl,
                        'reason': 'stop_loss'
                    })
                    cash += shares * current
                    del positions[code]

        # 止盈检查
        for code in list(positions.keys()):
            shares = positions[code]['shares']
            entry = positions[code]['entry_price']
            current = next((s['price'] for s in signals if s['code'] == code), None)
            if current:
                pnl = (current - entry) / entry
                if pnl >= params['take_profit']:
                    trades.append({
                        'date': date_str,
                        'code': code,
                        'action': 'SELL',
                        'pnl': pnl,
                        'reason': 'take_profit'
                    })
                    cash += shares * current
                    del positions[code]

        # 买入新股
        while len(positions) < top_n and buy_signals:
            sig = buy_signals.pop(0)
            if sig['code'] in positions:
                continue

            # 计算买入数量
            target_value = cash * params['position_size']
            shares = max(100, int(target_value / sig['price'] / 100) * 100)

            if shares * sig['price'] <= cash:
                positions[sig['code']] = {
                    'shares': shares,
                    'entry_price': sig['price'],
                    'entry_date': date_str
                }
                cash -= shares * sig['price']
                trades.append({
                    'date': date_str,
                    'code': sig['code'],
                    'action': 'BUY',
                    'price': sig['price'],
                    'shares': shares,
                    'reason': 'signal_buy'
                })

        # 更新权益曲线
        positions_value = sum(
            p['shares'] * next((s['price'] for s in signals if s['code'] == code), p['entry_price'])
            for code, p in positions.items()
        )
        total_value = cash + positions_value
        equity_curve.append({
            'date': date_str,
            'total_value': total_value,
            'cash': cash,
            'positions': len(positions)
        })

    # 计算绩效指标
    if not equity_curve:
        return None

    equity_df = pd.DataFrame(equity_curve)
    initial_value = 2000000.0
    final_value = equity_df['total_value'].iloc[-1]

    total_return = (final_value - initial_value) / initial_value

    # 年化收益
    n_days = len(equity_df)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 最大回撤
    equity_df['peak'] = equity_df['total_value'].cummax()
    equity_df['drawdown'] = (equity_df['total_value'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()

    # 夏普比率（简化）
    returns = equity_df['total_value'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # 交易统计
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    win_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]

    return {
        'top_n': top_n,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_count': len(sell_trades),
        'win_rate': len(win_trades) / len(sell_trades) if sell_trades else 0,
        'avg_holding_days': n_days / len(sell_trades) if sell_trades else 0,
        'final_value': final_value,
        'positions': len(positions)
    }


def main():
    print("=" * 70)
    print("top_n Grid Search - 持仓分散化参数优化")
    print("=" * 70)
    print(f"测试范围: top_n = 1 ~ 10")
    print(f"固定参数: position_size = {POSITION_SIZE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载股票池
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    with open(pool_path, 'r') as f:
        pool = json.load(f)
        stock_codes = pool.get('stocks', [])[:50]  # 取前50只

    print(f"加载股票数: {len(stock_codes)}")

    results = []

    for top_n in TOP_N_RANGE:
        print(f"\n测试 top_n = {top_n}...")
        result = run_backtest_for_topn(top_n, stock_codes)

        if result:
            results.append(result)
            print(f"  收益率: {result['total_return']*100:.2f}%")
            print(f"  年化收益: {result['annual_return']*100:.2f}%")
            print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {result['max_drawdown']*100:.2f}%")
            print(f"  交易次数: {result['trade_count']}")
            print(f"  胜率: {result['win_rate']*100:.1f}%")

    # 排序找最优
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    # 按综合评分排序（收益/回撤比）
    for r in results:
        r['score'] = r['annual_return'] / abs(r['max_drawdown']) if r['max_drawdown'] != 0 else 0

    results.sort(key=lambda x: -x['score'])

    print(f"{'top_n':>5} | {'收益率':>8} | {'年化':>8} | {'夏普':>6} | {'最大回撤':>9} | {'交易数':>6} | {'胜率':>6}")
    print("-" * 70)

    for r in results:
        print(f"{r['top_n']:>5} | {r['total_return']*100:>7.2f}% | {r['annual_return']*100:>7.2f}% | {r['sharpe_ratio']:>6.2f} | {r['max_drawdown']*100:>8.2f}% | {r['trade_count']:>6} | {r['win_rate']*100:>5.1f}%")

    # 最优结果
    best = results[0]
    print("\n" + "=" * 70)
    print(f"最优top_n: {best['top_n']}")
    print(f"  收益率: {best['total_return']*100:.2f}%")
    print(f"  年化收益: {best['annual_return']*100:.2f}%")
    print(f"  夏普比率: {best['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {best['max_drawdown']*100:.2f}%")
    print("=" * 70)

    # 保存结果
    result_file = f'{CACHE_DIR}/topn_grid_search.json'
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_n_range': list(TOP_N_RANGE),
            'position_size': POSITION_SIZE,
            'results': results,
            'best_top_n': best['top_n']
        }, f, indent=2)
    print(f"\n结果已保存到: {result_file}")

    return best['top_n']


if __name__ == '__main__':
    best_top_n = main()
    print(f"\n建议将DEFAULT_PARAMS['top_n']设置为: {best_top_n}")
