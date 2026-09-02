#!/usr/bin/env python3
"""单变量测试 - 每次只改变一个参数"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import os
from datetime import datetime
from src.simulation.evolving_portfolio import (
    load_stock_data, EnhancedModel, MultiStrategySignal,
    Position, Portfolio, DEFAULT_PARAMS
)


def get_trading_dates(data_dict, start_date):
    """获取交易日期列表"""
    all_dates = set()
    for df in data_dict.values():
        dates = df['date'].tolist()
        all_dates.update(dates)
    dates = sorted([d for d in all_dates if d >= start_date])
    return dates

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'

START_DATE = '2026-05-01'
END_DATE = '2026-08-22'


def run_backtest(codes, params, start_date, end_date):
    """运行回测，返回收益率"""
    data_dict = load_stock_data(codes)
    if not data_dict:
        return None

    trading_dates = get_trading_dates(data_dict, start_date)
    if end_date:
        trading_dates = [d for d in trading_dates if d <= end_date]
    if not trading_dates:
        return None

    model = EnhancedModel()
    model.load()
    signal_gen = MultiStrategySignal(model)

    cash = 2000000
    positions = []
    equity_curve = []
    total_value = 2000000

    for date in trading_dates:
        positions_value = sum(p.shares * p.current_price for p in positions)
        total_value = cash + positions_value

        daily_data = {}
        for code, df in data_dict.items():
            df_date = df[df['date'] <= date]
            if len(df_date) >= 60:
                daily_data[code] = df_date

        if not daily_data:
            continue

        signals = signal_gen.calculate_signals(daily_data, params)

        buy_signals = {k: v for k, v in signals.items() if v['signal'] == 'BUY'}
        sell_signals = {k: v for k, v in signals.items() if v['signal'] == 'SELL'}

        # 止损止盈
        for pos in positions[:]:
            code = pos.code
            if code in daily_data:
                df = daily_data[code]
                current_price = float(df['close'].iloc[-1])
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price

                if pnl_pct < -params['stop_loss']:
                    cash += pos.shares * current_price * 0.999
                    positions.remove(pos)
                    continue

                if pnl_pct > params['take_profit']:
                    cash += pos.shares * current_price * 0.999
                    positions.remove(pos)
                    continue

                pos.current_price = current_price
                pos.pnl_pct = pnl_pct

        # 卖出信号
        for pos in positions[:]:
            if pos.code in sell_signals:
                sig = sell_signals[pos.code]
                if sig['price'] > 0:
                    cash += pos.shares * sig['price'] * 0.999
                    positions.remove(pos)

        # 买入
        if buy_signals and len(positions) < int(params['top_n']):
            candidates = []
            for code, sig in buy_signals.items():
                if sig['price'] <= 0:
                    continue
                score_weight = max(1, sig['score'] - 40)
                candidates.append({
                    'code': code,
                    'sig': sig,
                    'score_weight': score_weight
                })

            if candidates:
                total_weight = sum(c['score_weight'] for c in candidates)
                weights = [c['score_weight'] / total_weight for c in candidates]

                base_position_value = total_value * params['position_size']

                for c, w in zip(candidates, weights):
                    if len(positions) >= int(params['top_n']):
                        break
                    code = c['code']
                    sig = c['sig']
                    position_value = base_position_value * (1 + w * (len(candidates) - 1))
                    shares = max(100, int(position_value / sig['price'] / 100) * 100)
                    cost = shares * sig['price'] * 1.0004

                    if cost <= cash:
                        cash -= cost
                        positions.append(Position(
                            code=code,
                            name=sig.get('name', code),
                            shares=shares,
                            entry_date=date,
                            entry_price=sig['price'],
                            current_price=sig['price'],
                            pnl_pct=0
                        ))

        positions_value = sum(p.shares * p.current_price for p in positions)
        total_value = cash + positions_value
        equity_curve.append(total_value)

    final_return = (total_value / 2000000 - 1)
    return final_return


def get_stock_codes():
    """获取股票列表"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('stocks', [])[:100]
    return ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']


def main():
    codes = get_stock_codes()[:20]  # 限制20只
    print(f"股票数量: {len(codes)}")
    print(f"回测区间: {START_DATE} ~ {END_DATE}")
    print()

    # 基线参数
    baseline = {
        'buy_threshold': 55,
        'sell_threshold': 40,
        'stop_loss': 0.05,
        'take_profit': 0.12,
        'position_size': 0.08,
        'top_n': 5,
    }

    baseline_return = run_backtest(codes, baseline, START_DATE, END_DATE)
    print(f"基线收益率: {baseline_return:+.2%}")
    print()

    # 单变量测试 - 每次只测2个值
    tests = [
        # buy_threshold
        ('buy_threshold=50', {**baseline, 'buy_threshold': 50}),
        ('buy_threshold=60', {**baseline, 'buy_threshold': 60}),

        # sell_threshold
        ('sell_threshold=35', {**baseline, 'sell_threshold': 35}),
        ('sell_threshold=45', {**baseline, 'sell_threshold': 45}),

        # stop_loss
        ('stop_loss=0.03', {**baseline, 'stop_loss': 0.03}),
        ('stop_loss=0.07', {**baseline, 'stop_loss': 0.07}),

        # take_profit
        ('take_profit=0.08', {**baseline, 'take_profit': 0.08}),
        ('take_profit=0.15', {**baseline, 'take_profit': 0.15}),

        # top_n
        ('top_n=3', {**baseline, 'top_n': 3}),
        ('top_n=8', {**baseline, 'top_n': 8}),

        # position_size
        ('position_size=0.06', {**baseline, 'position_size': 0.06}),
        ('position_size=0.12', {**baseline, 'position_size': 0.12}),
    ]

    results = []
    for name, params in tests:
        ret = run_backtest(codes, params, START_DATE, END_DATE)
        if ret is not None:
            diff = ret - baseline_return
            results.append((name, ret, diff))
            marker = "✓" if diff > 0 else "✗"
            print(f"{marker} {name}: {ret:+.2%} (差异: {diff:+.2%})")

    print()
    print("=" * 60)
    print("改进的参数（优于基线）:")
    improved = [(n, r, d) for n, r, d in results if d > 0]
    improved.sort(key=lambda x: x[2], reverse=True)
    for name, ret, diff in improved:
        print(f"  {name}: {ret:+.2%} (改进 {diff:+.2%})")

    print()
    print("最差的参数:")
    worst = min(results, key=lambda x: x[2])
    print(f"  {worst[0]}: {worst[1]:+.2%}")


if __name__ == '__main__':
    main()
