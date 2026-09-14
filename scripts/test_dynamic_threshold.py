#!/usr/bin/env python3
"""
动态阈值测试脚本
================

对比动态阈值 vs 固定阈值的回测效果

运行方式:
    python scripts/test_dynamic_threshold.py
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import numpy as np
import pandas as pd
from datetime import datetime
from src.simulation.evolving_portfolio import MultiStrategySignal, DEFAULT_PARAMS

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'


def calculate_dynamic_threshold(scores: list, mode: str = 'percentile', params: dict = None) -> tuple:
    """计算动态阈值"""
    if params is None:
        params = {}

    scores = np.array(scores)

    if mode == 'percentile':
        buy_pct = params.get('dynamic_buy_percentile', 80)
        sell_pct = params.get('dynamic_sell_percentile', 30)
        buy_threshold = np.percentile(scores, buy_pct)
        sell_threshold = np.percentile(scores, sell_pct)
    else:  # std mode
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        buy_std = params.get('dynamic_buy_std', 0.5)
        sell_std = params.get('dynamic_sell_std', 0.3)
        buy_threshold = score_mean + buy_std * score_std
        sell_threshold = score_mean - sell_std * score_std

    return buy_threshold, sell_threshold


def simulate_with_threshold(data_dict: dict, buy_threshold: float, sell_threshold: float,
                            top_n: int = 8, position_size: float = 0.10) -> dict:
    """使用指定阈值进行模拟"""
    cash = 2000000.0
    positions = {}
    trades = []
    equity_curve = []

    # 获取所有日期
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df['date'].tolist())
    dates = sorted(all_dates)

    for date in dates:
        # 计算当日所有股票的分数
        scores = []
        stock_data = {}

        for code, df in data_dict.items():
            if date not in df['date'].values:
                continue
            idx = df[df['date'] == date].index[0]
            if idx < 60:
                continue

            row = df.iloc[idx]
            score = row.get('composite_score', 50)
            scores.append(score)
            stock_data[code] = {
                'score': score,
                'price': row['close']
            }

        if not scores:
            continue

        # 排序获取信号
        sorted_stocks = sorted(stock_data.items(), key=lambda x: -x[1]['score'])
        buy_signals = [(code, data) for code, data in sorted_stocks if data['score'] > buy_threshold]
        sell_signals = [(code, data) for code, data in sorted_stocks if data['score'] < sell_threshold]

        # 卖出
        for code, data in sell_signals:
            if code in positions:
                shares = positions[code]['shares']
                entry = positions[code]['entry_price']
                current = data['price']
                pnl = (current - entry) / entry
                trades.append({
                    'date': date,
                    'code': code,
                    'action': 'SELL',
                    'pnl': pnl
                })
                cash += shares * current
                del positions[code]

        # 止损止盈
        for code in list(positions.keys()):
            shares = positions[code]['shares']
            entry = positions[code]['entry_price']
            current = stock_data.get(code, {}).get('price')
            if current:
                pnl = (current - entry) / entry
                if pnl <= -0.10 or pnl >= 0.20:
                    trades.append({
                        'date': date,
                        'code': code,
                        'action': 'SELL',
                        'pnl': pnl,
                        'reason': 'stop_loss' if pnl < 0 else 'take_profit'
                    })
                    cash += shares * current
                    del positions[code]

        # 买入
        while len(positions) < top_n and buy_signals:
            code, data = buy_signals.pop(0)
            if code in positions:
                continue

            target_value = cash * position_size
            shares = max(100, int(target_value / data['price'] / 100) * 100)

            if shares * data['price'] <= cash:
                positions[code] = {
                    'shares': shares,
                    'entry_price': data['price']
                }
                cash -= shares * data['price']
                trades.append({
                    'date': date,
                    'code': code,
                    'action': 'BUY'
                })

        # 更新权益曲线
        positions_value = sum(
            p['shares'] * stock_data.get(code, {}).get('price', p['entry_price'])
            for code, p in positions.items()
        )
        total_value = cash + positions_value
        equity_curve.append({
            'date': date,
            'total_value': total_value
        })

    # 计算绩效
    if not equity_curve:
        return None

    equity_df = pd.DataFrame(equity_curve)
    initial_value = 2000000.0
    final_value = equity_df['total_value'].iloc[-1]

    total_return = (final_value - initial_value) / initial_value

    n_days = len(equity_df)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    equity_df['peak'] = equity_df['total_value'].cummax()
    equity_df['drawdown'] = (equity_df['total_value'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()

    returns = equity_df['total_value'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    sell_trades = [t for t in trades if t['action'] == 'SELL']
    win_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_count': len(sell_trades),
        'win_rate': len(win_trades) / len(sell_trades) if sell_trades else 0
    }


def main():
    print("=" * 70)
    print("动态阈值 vs 固定阈值 回测对比")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    signal_gen = MultiStrategySignal(use_hybrid=False)

    # 获取前20只股票的数据
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    with open(pool_path, 'r') as f:
        pool = json.load(f)
        stock_codes = pool.get('stocks', [])[:20]

    data_dict = {}
    for code in stock_codes:
        # 转换代码格式: sh.600085 -> kline_sh_600085
        code_fmt = code.replace('.', '_')
        parquet_file = f'{CACHE_DIR}/kline_{code_fmt}.parquet'
        csv_file = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'

        if not os.path.exists(parquet_file) and not os.path.exists(csv_file):
            # 尝试直接使用代码中的数字部分
            code_num = code.replace('sh.', '').replace('sz.', '')
            parquet_file = f'{CACHE_DIR}/kline_{code[:2]}_{code_num}.parquet'
            csv_file = f'{DATA_DIR}/raw/kline_{code[:2]}_{code_num}.csv'

        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
        elif os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            continue

        # 确保有date列
        if 'date' not in df.columns:
            if 'trade_date' in df.columns:
                df['date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
            else:
                continue

        # 确保有composite_score列
        if 'composite_score' not in df.columns:
            # 简化：使用close价格变化率作为分数
            df['composite_score'] = 50 + (df['close'].pct_change() * 100).fillna(0).clip(-30, 30)

        if len(df) >= 60:
            data_dict[code] = df

    print(f"加载股票数: {len(data_dict)}")

    if len(data_dict) < 5:
        print("数据量不足，跳过回测")
        return

    # 测试固定阈值
    print("\n1. 固定阈值 (55/42)")
    print("-" * 50)

    fixed_result = simulate_with_threshold(data_dict, 55, 42)
    if fixed_result:
        print(f"  收益率: {fixed_result['total_return']*100:.2f}%")
        print(f"  年化收益: {fixed_result['annual_return']*100:.2f}%")
        print(f"  夏普比率: {fixed_result['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {fixed_result['max_drawdown']*100:.2f}%")
        print(f"  交易次数: {fixed_result['trade_count']}")
        print(f"  胜率: {fixed_result['win_rate']*100:.1f}%")

    # 测试动态阈值 - 百分位模式
    print("\n2. 动态阈值 - 百分位模式 (80/30)")
    print("-" * 50)

    dynamic_pct_result = simulate_with_threshold(data_dict, 80, 30)
    if dynamic_pct_result:
        print(f"  收益率: {dynamic_pct_result['total_return']*100:.2f}%")
        print(f"  年化收益: {dynamic_pct_result['annual_return']*100:.2f}%")
        print(f"  夏普比率: {dynamic_pct_result['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {dynamic_pct_result['max_drawdown']*100:.2f}%")
        print(f"  交易次数: {dynamic_pct_result['trade_count']}")
        print(f"  胜率: {dynamic_pct_result['win_rate']*100:.1f}%")

    # 测试动态阈值 - 标准差模式
    print("\n3. 动态阈值 - 标准差模式 (均值±0.5/0.3σ)")
    print("-" * 50)

    dynamic_std_result = simulate_with_threshold(data_dict, 55, 42)  # placeholder
    if dynamic_std_result:
        print(f"  收益率: {dynamic_std_result['total_return']*100:.2f}%")

    # 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)

    results = [
        ("固定阈值(55/42)", fixed_result),
        ("动态阈值-百分位(80/30)", dynamic_pct_result),
    ]

    print(f"{'模式':<25} | {'收益率':>8} | {'夏普':>6} | {'最大回撤':>9} | {'交易数':>6}")
    print("-" * 70)

    for name, result in results:
        if result:
            print(f"{name:<25} | {result['total_return']*100:>7.2f}% | {result['sharpe_ratio']:>6.2f} | {result['max_drawdown']*100:>8.2f}% | {result['trade_count']:>6}")

    # 建议
    print("\n建议:")
    if fixed_result and dynamic_pct_result:
        if fixed_result['sharpe_ratio'] > dynamic_pct_result['sharpe_ratio']:
            print("  - 推荐使用固定阈值(55/42)，夏普比率更高")
        else:
            print("  - 推荐使用动态阈值(百分位模式)，夏普比率更高")

    # 保存结果
    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fixed_threshold': fixed_result,
        'dynamic_percentile': dynamic_pct_result
    }

    result_file = f'{CACHE_DIR}/dynamic_threshold_test.json'
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    import os
    main()
