#!/usr/bin/env python3
"""
从指定日期开始的历史模拟回测
"""
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict

CACHE_DIR = 'data/cache'

@dataclass
class Position:
    code: str
    name: str
    shares: int
    entry_date: str
    entry_price: float
    current_price: float
    pnl_pct: float

def load_stock_data(codes: list) -> Dict:
    """加载股票数据"""
    data_dict = {}
    for code in codes:
        cache_file = f'{CACHE_DIR}/kline_{code.replace(".", "_")}.parquet'
        if __import__('os').path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            if 'vol' in df.columns:
                df = df[df['vol'] > 0]
            elif 'volume' in df.columns:
                df = df[df['volume'] > 0]
            if len(df) >= 60 and 'date' in df.columns:
                data_dict[code] = df
    return data_dict

def calculate_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算增强特征"""
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    for period in [1, 3, 5, 10, 20, 60]:
        df[f'return_{period}d'] = df['close'].pct_change(period)

    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    for window in [5, 10, 20, 60, 120]:
        df[f'ma{window}'] = pd.Series(close).rolling(window).mean()

    for period in [5, 10, 20, 60]:
        df[f'momentum_{period}d'] = close / (np.roll(close, period) + 1e-8) - 1

    return df

def calc_rsi(prices, period=14):
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def run_simulation(start_date: str = '2026-05-01'):
    """运行模拟"""
    print("="*60)
    print(f"历史模拟回测: {start_date} 至今")
    print("="*60)

    # Load stock pool - use full data pool for better backtest
    with open('data/expanded_stock_pool_full.json', 'r') as f:
        pool = json.load(f)
    codes = pool.get('stocks', [])[:50]

    # Load data
    print(f"\n加载股票数据...")
    data_dict = load_stock_data(codes)
    print(f"加载了 {len(data_dict)} 只股票")

    # Find latest available date
    latest_dates = []
    for code, df in data_dict.items():
        if 'date' in df.columns:
            latest_dates.append(df['date'].max())
    latest_date = max(latest_dates) if latest_dates else start_date
    print(f"数据范围: {start_date} ~ {latest_date}")

    # Initialize
    cash = 2000000.0
    positions = []
    equity_curve = []

    # Load CFFEX data
    with open(f'{CACHE_DIR}/citic_positions_history.json', 'r') as f:
        cffex_raw = json.load(f)
    daily_net = {}
    for key, item in cffex_raw.items():
        d = item['date']
        if d not in daily_net:
            daily_net[d] = 0
        daily_net[d] += item['net']
    cffex_dates = sorted(daily_net.keys())

    # Parameters - balanced momentum (best performing)
    params = {
        'top_n': 6,
        'position_size': 0.10,
        'buy_threshold': 52,
        'sell_threshold': 48,
        'stop_loss': 0.05,
        'take_profit': 0.20,
        'trend_weight': 0.30,
        'momentum_weight': 0.70,
    }

    # Get all trading dates
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df['date'].tolist())
    all_dates = sorted([d for d in all_dates if d >= start_date])

    print(f"\n开始模拟 ({len(all_dates)} 个交易日)...")

    buy_count = 0
    sell_count = 0

    for i, current_date in enumerate(all_dates):
        if i % 20 == 0:
            print(f"  进度: {i}/{len(all_dates)} ({current_date})")

        # Get stock data up to current date
        signals = {}
        for code, df in data_dict.items():
            df_hist = df[df['date'] <= current_date].copy()
            if len(df_hist) < 60:  # Require 60 days for MA60
                continue

            close = df_hist['close'].values
            date_vals = df_hist['date'].values

            # Trend following strategy - avoid buying in downtrends
            ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
            ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
            ma60 = pd.Series(close).rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
            rsi = calc_rsi(close).iloc[-1]

            # Trend score: only bullish if price above all moving averages
            price = close[-1]
            trend_score = 50
            if price > ma5 > ma20 > ma60:  # Strong uptrend
                trend_score = 100
            elif price > ma5 and ma5 > ma20:  # Moderate uptrend
                trend_score = 75
            elif price > ma5:  # Weak uptrend
                trend_score = 60
            elif rsi < 35:  # Oversold - could bounce
                trend_score = 45
            else:
                trend_score = 30  # Downtrend

            # Momentum score: positive momentum preferred
            ret_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
            momentum_score = 50 + (40 if ret_20d > 0.10 else (20 if ret_20d > 0 else (-20 if ret_20d < -0.05 else 0)))

            combined = (
                trend_score * params['trend_weight'] +
                momentum_score * params['momentum_weight']
            )

            # Apply CFFEX boost (contrarian: high net = potential reversal)
            applicable_dates = [d for d in cffex_dates if d <= current_date]
            cffex_signal = 0
            if applicable_dates:
                prev_date = applicable_dates[-1]
                if prev_date in daily_net and applicable_dates[-1] in cffex_dates:
                    prev_idx = cffex_dates.index(applicable_dates[-1])
                    if prev_idx > 0:
                        curr_net = daily_net[cffex_dates[prev_idx]]
                        prev_net = daily_net[cffex_dates[prev_idx-1]]
                        chg = curr_net - prev_net
                        # High net shorts with decreasing net = potential bounce
                        if curr_net > 50000 and chg < -2000:
                            cffex_signal = 5  # Buy signal
                        elif curr_net > 70000 and chg < -5000:
                            cffex_signal = 8  # Strong buy
                        elif curr_net < 30000:
                            cffex_signal = -3  # Reduce exposure when net is low
                    combined += cffex_signal

            current_price = close[-1]

            signals[code] = {
                'score': combined,
                'price': current_price,
                'signal': 'BUY' if combined > params['buy_threshold'] else ('SELL' if combined < params['sell_threshold'] else 'HOLD')
            }

        # Process sells
        for pos in positions[:]:
            if pos.code in signals and signals[pos.code]['signal'] == 'SELL':
                sig = signals[pos.code]
                pnl = (sig['price'] - pos.entry_price) / pos.entry_price
                if pnl < -params['stop_loss'] or pnl > params['take_profit'] or sig['signal'] == 'SELL':
                    cash += pos.shares * sig['price'] * 0.999
                    sell_count += 1
                    positions.remove(pos)

        # Process buys
        if len(positions) < params['top_n']:
            buy_signals = [(code, sig) for code, sig in signals.items()
                          if sig['signal'] == 'BUY' and code not in [p.code for p in positions]]
            buy_signals.sort(key=lambda x: x[1]['score'], reverse=True)

            for code, sig in buy_signals[:params['top_n'] - len(positions)]:
                target_value = (cash + sum(p.shares * p.current_price for p in positions)) * params['position_size']
                shares = max(100, int(target_value / sig['price'] / 100) * 100)
                cost = shares * sig['price'] * 1.0004

                if cost <= cash:
                    cash -= cost
                    positions.append(Position(
                        code=code,
                        name=pool.get('names', {}).get(code, code),
                        shares=shares,
                        entry_date=current_date,
                        entry_price=sig['price'],
                        current_price=sig['price'],
                        pnl_pct=0
                    ))
                    buy_count += 1

        # Update positions
        positions_value = 0
        for pos in positions:
            if pos.code in signals:
                pos.current_price = signals[pos.code]['price']
                pos.pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
            positions_value += pos.shares * pos.current_price

        total_value = cash + positions_value
        equity_curve.append({
            'date': current_date,
            'total_value': total_value,
            'cash': cash,
            'positions_value': positions_value,
            'positions': len(positions)
        })

    # Save results
    result = {
        'start_date': start_date,
        'end_date': latest_date,
        'initial_capital': 2000000,
        'final_value': equity_curve[-1]['total_value'] if equity_curve else 2000000,
        'returns': (equity_curve[-1]['total_value'] / 2000000 - 1) * 100 if equity_curve else 0,
        'total_trades': buy_count + sell_count,
        'buy_trades': buy_count,
        'sell_trades': sell_count,
        'equity_curve': equity_curve
    }

    # Save
    with open(f'{CACHE_DIR}/backtest_result.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("模拟完成!")
    print("="*60)
    print(f"期间: {start_date} ~ {latest_date}")
    print(f"初始资金: 2,000,000")
    print(f"最终价值: {result['final_value']:,.2f}")
    print(f"收益率: {result['returns']:+.2f}%")
    print(f"买入交易: {buy_count}")
    print(f"卖出交易: {sell_count}")
    print(f"最终持仓: {len(positions)} 只")

    return result

if __name__ == '__main__':
    run_simulation('2026-05-01')
