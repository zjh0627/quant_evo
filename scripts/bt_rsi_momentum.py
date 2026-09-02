#!/usr/bin/env python3
"""
均线金叉死叉策略回测
"""

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
from src.backtest.light_backtest import LightBacktest


def generate_ma_cross_signals(df: pd.DataFrame) -> pd.DataFrame:
    """均线金叉死叉信号"""
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)
    df['trend_change'] = df['trend'].diff()

    signals = []
    scores = []
    for idx, row in df.iterrows():
        if pd.isna(row['trend_change']):
            signals.append('HOLD')
            scores.append(50)
        elif row['trend_change'] > 0:
            signals.append('BUY')
            scores.append(80)  # 金叉给高评分
        elif row['trend_change'] < 0:
            signals.append('SELL')
            scores.append(80)
        else:
            # 趋势延续，给中等评分
            if row['trend'] > 0:
                signals.append('HOLD')
                scores.append(60)
            else:
                signals.append('HOLD')
                scores.append(40)

    df['signal'] = signals
    df['score'] = scores

    return df[['date', 'signal', 'score']]


def load_stock_data(codes: list, data_dir: str = 'data/raw') -> dict:
    data_dict = {}
    for code in codes:
        code_fmt = code.replace('.', '_')
        filepath = os.path.join(data_dir, f'kline_{code_fmt}.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[df['volume'] > 0]
            if len(df) >= 60:
                data_dict[code] = df
    return data_dict


def main():
    import json

    print("=" * 60)
    print("均线金叉死叉策略回测")
    print("=" * 60)

    pool_path = 'data/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:30]
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

    print(f"加载股票: {len(codes)} 只")

    data_dict = load_stock_data(codes)
    print(f"有效数据: {len(data_dict)} 只")

    signals_dict = {}
    for code, df in data_dict.items():
        try:
            signals_dict[code] = generate_ma_cross_signals(df)
        except Exception as e:
            print(f"信号生成失败 {code}: {e}")

    print(f"信号生成完成: {len(signals_dict)} 只")

    if not signals_dict:
        return

    # 统计信号分布
    all_signals = []
    for sig_df in signals_dict.values():
        all_signals.extend(sig_df['signal'].tolist())
    from collections import Counter
    sig_counts = Counter(all_signals)
    print(f"信号分布: {dict(sig_counts)}")

    bt = LightBacktest(
        initial_capital=1000000,
        commission=0.0003,
        slippage=0.0001,
        position_size=0.1,
        max_positions=10,
        stop_loss=-0.15,
        take_profit=0.30
    )

    print("\n运行回测...")
    result = bt.run(
        data_dict,
        signals_dict,
        start_date='2024-07-01',
        end_date='2026-07-20'
    )

    if result:
        print("\n" + result.summary())

        bh_returns = []
        for code, df in data_dict.items():
            if len(df) > 0:
                start_df = df[df['date'] >= '2024-07-01']
                end_df = df[df['date'] <= '2026-07-20']
                if len(start_df) > 0 and len(end_df) > 0:
                    start_price = start_df['close'].iloc[0]
                    end_price = end_df['close'].iloc[-1]
                    bh_returns.append((end_price - start_price) / start_price)

        if bh_returns:
            avg_bh_return = np.mean(bh_returns)
            print(f"\n平均买入持有收益: {avg_bh_return:.2%}")
            print(f"策略总收益: {result.total_return:.2%}")
            print(f"超额收益: {result.total_return - avg_bh_return:.2%}")

        output = {
            'strategy': {
                'total_return': float(result.total_return),
                'annual_return': float(result.annual_return),
                'max_drawdown': float(result.max_drawdown),
                'sharpe': float(result.sharpe_ratio),
                'win_rate': float(result.win_rate),
                'total_trades': int(result.total_trades)
            },
            'buy_hold': {
                'avg_return': float(np.mean(bh_returns)) if bh_returns else 0
            }
        }

        os.makedirs('data/cache', exist_ok=True)
        with open('data/cache/backtest_result.json', 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        print("回测失败")


if __name__ == '__main__':
    main()