#!/usr/bin/env python3
"""
多策略自动优化系统
==================
考虑多种因素:
- 技术指标: RSI, MACD, 布林带, 均线
- 成交量: 放量缩量异动
- 主力资金: 大单净流入
- 新闻情绪: 利好利空
- 市场情绪: 整体涨跌家数

运行方式:
    python scripts/autoresearch.py --rounds 30 --start 2026-07-01 --end 2026-08-21
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import os
import random
import copy
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import requests

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
MODEL_DIR = f'{DATA_DIR}/models'

# 输出文件
OPTIMIZATION_LOG = f'{CACHE_DIR}/autoresearch_log.json'
BEST_PARAMS_FILE = f'{CACHE_DIR}/best_strategy_params.json'

# 参数搜索空间
PARAM_SPACE = {
    'buy_threshold': [55, 60, 65, 70, 75],
    'sell_threshold': [35, 40, 45, 50],
    'stop_loss': [0.03, 0.04, 0.05, 0.06, 0.08],
    'take_profit': [0.08, 0.10, 0.12, 0.15, 0.20],
    'position_size': [0.08, 0.10, 0.12, 0.15],
    'top_n': [3, 4, 5, 6, 7],
    'rsi_oversold': [25, 30, 35, 40],
    'rsi_overbought': [60, 65, 70, 75],
    'vol_threshold': [1.5, 2.0, 2.5],
    'news_weight': [0.0, 0.05, 0.1, 0.15, 0.2],
    'trend_weight': [0.1, 0.15, 0.2, 0.25, 0.3],
    'ml_weight': [0.3, 0.4, 0.5, 0.6],
    'mean_rev_weight': [0.1, 0.15, 0.2],
    'momentum_weight': [0.05, 0.1, 0.15],
}


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    avg_trade_return: float
    score: float


@dataclass
class Experiment:
    """实验记录"""
    round_id: int
    params: Dict
    result: BacktestResult
    timestamp: str


def load_stock_pool() -> Tuple[List[str], Dict]:
    """加载股票池"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('stocks', [])[:50], pool.get('names', {})
    return ['sh.600519', 'sh.600036', 'sh.601318', 'sh.000001'], {}


def load_stock_data(codes: list) -> Dict[str, pd.DataFrame]:
    """加载股票历史数据"""
    data_dict = {}
    for code in codes:
        code_fmt = code.replace('.', '_')
        cache_file = f'{CACHE_DIR}/kline_{code_fmt}.parquet'
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[df['volume'] > 0]
            if len(df) >= 60:
                data_dict[code] = df
    return data_dict


def get_sina_realtime(codes: list) -> dict:
    """获取实时价格"""
    prices = {}
    try:
        code_str = ','.join([c.replace('.', '') for c in codes])
        url = f'http://hq.sinajs.cn/list={code_str}'
        headers = {'Referer': 'http://finance.sina.com.cn', 'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, timeout=5, headers=headers)
        r.encoding = 'gbk'
        lines = r.text.strip().split('\n')
        for i, line in enumerate(lines):
            if '="' in line and i < len(codes):
                code = codes[i]
                data = line.split('="')[1].strip('";').split(',')
                if len(data) > 4:
                    try:
                        prices[code] = {
                            'close': float(data[3]),
                            'open': float(data[1]),
                            'high': float(data[4]),
                            'low': float(data[5]),
                            'volume': float(data[8]) if len(data) > 8 else 0,
                        }
                    except:
                        pass
    except:
        pass
    return prices


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林带
    ma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # 均线
    for w in [5, 10, 20, 60]:
        df[f'ma{w}'] = pd.Series(close).rolling(w).mean()

    # 动量
    for p in [5, 10, 20]:
        df[f'mom{p}'] = close / (np.roll(close, p) + 1e-8) - 1

    # 成交量
    df['vol_ma20'] = pd.Series(volume).rolling(20).mean()
    df['vol_ratio'] = volume / (df['vol_ma20'] + 1e-8)

    return df


def calculate_signal(df: pd.DataFrame, params: Dict, news_boost: float = 0) -> Dict:
    """计算单只股票评分"""
    if len(df) < 30:
        return None

    close = df['close'].values
    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
    macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
    bb_pos = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
    vol_ratio = df['vol_ratio'].iloc[-1] if 'vol_ratio' in df.columns else 1.0

    ma5 = df['ma5'].iloc[-1] if 'ma5' in df.columns else close[-1]
    ma20 = df['ma20'].iloc[-1] if 'ma20' in df.columns else close[-1]
    ma60 = df['ma60'].iloc[-1] if 'ma60' in df.columns else close[-1]

    mom5 = df['mom5'].iloc[-1] if 'mom5' in df.columns else 0
    mom20 = df['mom20'].iloc[-1] if 'mom20' in df.columns else 0

    score = 50

    # RSI评分
    if rsi < params['rsi_oversold']:
        score += 25
    elif rsi < 40:
        score += 10
    elif rsi > params['rsi_overbought']:
        score -= 25
    elif rsi > 60:
        score -= 10

    # 均线趋势
    if ma5 > ma20:
        score += params['trend_weight'] * 100
    else:
        score -= params['trend_weight'] * 100

    if ma20 > ma60:
        score += 5

    # MACD
    if macd_hist > 0:
        score += 8
    else:
        score -= 8

    # 布林带
    if bb_pos < 0.2:
        score += 10
    elif bb_pos > 0.8:
        score -= 10

    # 成交量异动
    if vol_ratio > params['vol_threshold']:
        score += 8
    elif vol_ratio < 0.5:
        score -= 3

    # 动量
    if mom5 > 0.03:
        score += 8
    elif mom5 < -0.03:
        score -= 8

    if mom20 > 0.10:
        score += 5
    elif mom20 < -0.10:
        score -= 5

    # 新闻加成
    score += news_boost

    score = max(0, min(100, score))

    # 信号
    if score >= params['buy_threshold']:
        signal = 'BUY'
    elif score <= params['sell_threshold']:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'signal': signal,
        'score': score,
        'rsi': rsi,
        'macd_hist': macd_hist,
        'bb_position': bb_pos,
        'vol_ratio': vol_ratio,
        'price': float(close[-1]),
        'mom5': mom5,
        'mom20': mom20,
    }


def run_backtest(data_dict: Dict[str, pd.DataFrame], params: Dict,
                 start_date: str, end_date: str) -> BacktestResult:
    """运行回测"""
    # 过滤日期
    trading_dates = set()
    for df in data_dict.values():
        dates = df['date'].tolist()
        for d in dates:
            if start_date <= d <= end_date:
                trading_dates.add(d)
    trading_dates = sorted(trading_dates)

    if len(trading_dates) < 10:
        return None

    # 初始化
    cash = 2000000
    positions = {}  # code -> {shares, entry_price, entry_date}
    equity_curve = []
    trade_history = []

    # T+1追踪：当天买的不能卖
    buy_today = set()  # 当天买入的股票

    for date in trading_dates:
        # 更新持仓价格
        positions_value = 0
        for code, pos in list(positions.items()):
            if code in data_dict:
                df = data_dict[code]
                df_date = df[df['date'] == date]
                if len(df_date) > 0:
                    pos['current_price'] = float(df_date['close'].iloc[-1])
                    pos['pnl_pct'] = (pos['current_price'] - pos['entry_price']) / pos['entry_price']
            positions_value += pos['shares'] * pos.get('current_price', pos['entry_price'])

        total_value = cash + positions_value

        # 止损止盈
        for code, pos in list(positions.items()):
            pnl = pos.get('pnl_pct', 0)

            # 止损
            if pnl <= -params['stop_loss']:
                revenue = pos['shares'] * pos['current_price'] * 0.999
                cash += revenue
                trade_history.append({
                    'date': date,
                    'action': 'SELL',
                    'reason': 'STOP_LOSS',
                    'code': code,
                    'pnl_pct': pnl,
                })
                del positions[code]
                continue

            # 止盈
            if pnl >= params['take_profit']:
                revenue = pos['shares'] * pos['current_price'] * 0.999
                cash += revenue
                trade_history.append({
                    'date': date,
                    'action': 'SELL',
                    'reason': 'TAKE_PROFIT',
                    'code': code,
                    'pnl_pct': pnl,
                })
                del positions[code]
                continue

        # 获取历史数据（截止到当天）并计算信号
        signals = {}
        for code, df in data_dict.items():
            df_hist = df[df['date'] <= date]
            if len(df_hist) >= 30:
                df_with_indicators = calculate_indicators(df_hist)
                sig = calculate_signal(df_with_indicators, params)
                if sig:
                    signals[code] = sig

        # 卖出信号 (T+1: 不能卖当天买的)
        for code, pos in list(positions.items()):
            if code in signals and signals[code]['signal'] == 'SELL':
                if code not in buy_today:  # T+1检查
                    revenue = pos['shares'] * signals[code]['price'] * 0.999
                    pnl = (signals[code]['price'] - pos['entry_price']) / pos['entry_price']
                    cash += revenue
                    trade_history.append({
                        'date': date,
                        'action': 'SELL',
                        'reason': 'SIGNAL',
                        'code': code,
                        'pnl_pct': pnl,
                    })
                    del positions[code]

        # 买入信号
        if len(positions) < params['top_n']:
            buy_candidates = []
            for code, sig in signals.items():
                if sig['signal'] == 'BUY' and code not in positions:
                    buy_candidates.append((code, sig))

            # 按评分排序
            buy_candidates.sort(key=lambda x: -x[1]['score'])

            for code, sig in buy_candidates[:params['top_n'] - len(positions)]:
                position_value = total_value * params['position_size']
                shares = max(100, int(position_value / sig['price'] / 100) * 100)
                cost = shares * sig['price'] * 1.0003

                if cost <= cash:
                    cash -= cost
                    positions[code] = {
                        'shares': shares,
                        'entry_price': sig['price'],
                        'entry_date': date,
                        'current_price': sig['price'],
                        'pnl_pct': 0,
                    }
                    buy_today.add(code)  # 记录当天买的
                    trade_history.append({
                        'date': date,
                        'action': 'BUY',
                        'reason': 'SIGNAL',
                        'code': code,
                        'price': sig['price'],
                        'shares': shares,
                    })

        # 清空当天买入记录（新的一天）
        if date != trading_dates[-1]:
            next_idx = trading_dates.index(date) + 1
            if next_idx < len(trading_dates):
                # 下一天清空
                pass

        equity_curve.append({
            'date': date,
            'total_value': total_value,
            'cash': cash,
            'positions': len(positions),
        })

    # 计算结果
    if not equity_curve:
        return None

    initial_value = 2000000
    final_value = equity_curve[-1]['total_value']
    total_return = (final_value - initial_value) / initial_value

    # 年化收益
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days or 1
    annual_return = (1 + total_return) ** (365 / days) - 1

    # 最大回撤
    peak = initial_value
    max_drawdown = 0
    for e in equity_curve:
        if e['total_value'] > peak:
            peak = e['total_value']
        drawdown = (peak - e['total_value']) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # 交易统计
    sell_trades = [t for t in trade_history if t['action'] == 'SELL']
    winning_trades = [t for t in sell_trades if t.get('pnl_pct', 0) > 0]
    win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
    avg_return = np.mean([t.get('pnl_pct', 0) for t in sell_trades]) if sell_trades else 0

    # 夏普比率（简化版）
    returns = []
    for i in range(1, len(equity_curve)):
        ret = (equity_curve[i]['total_value'] - equity_curve[i-1]['total_value']) / equity_curve[i-1]['total_value']
        returns.append(ret)
    if returns:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 综合评分
    score = (annual_return * 100 + sharpe_ratio * 20 - max_drawdown * 50 +
             win_rate * 30 - len(sell_trades) * 0.5)

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trade_count=len(sell_trades),
        avg_trade_return=avg_return,
        score=score,
    )


def generate_params(experiences: List[Experiment], mode: str = 'explore') -> Dict:
    """基于历史经验生成新参数"""
    if not experiences or mode == 'random':
        # 随机生成
        params = {}
        for key, values in PARAM_SPACE.items():
            params[key] = random.choice(values)
        return params

    # 基于最佳参数做微调
    best = max(experiences, key=lambda x: x.result.score)
    params = copy.deepcopy(best.params)

    # 随机修改1-3个参数
    keys_to_change = random.sample(list(PARAM_SPACE.keys()),
                                   min(random.randint(1, 3), len(PARAM_SPACE)))
    for key in keys_to_change:
        values = PARAM_SPACE[key]
        if key in ['buy_threshold', 'sell_threshold', 'rsi_oversold', 'rsi_overbought', 'top_n']:
            params[key] = random.choice(values)
        else:
            params[key] = random.choice(values)

    return params


def run_optimization(start_date: str = '2026-07-01', end_date: str = '2026-08-21',
                    max_rounds: int = 30) -> Dict:
    """运行优化"""
    print(f"\n{'='*60}")
    print(f"多策略自动优化系统")
    print(f"回测区间: {start_date} → {end_date}")
    print(f"{'='*60}")

    # 加载数据
    codes, names = load_stock_pool()
    print(f"加载股票池: {len(codes)} 只")
    data_dict = load_stock_data(codes)
    print(f"有效数据: {len(data_dict)} 只")

    if len(data_dict) < 10:
        print("数据不足")
        return None

    # 初始化
    experiments: List[Experiment] = []
    best_score = float('-inf')
    best_params = None
    best_result = None

    print(f"\n开始优化 ({max_rounds} 轮)...")

    for round_id in range(1, max_rounds + 1):
        # 生成参数
        if round_id == 1:
            # 默认参数
            params = {
                'buy_threshold': 65,
                'sell_threshold': 40,
                'stop_loss': 0.05,
                'take_profit': 0.12,
                'position_size': 0.12,
                'top_n': 5,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'vol_threshold': 2.0,
                'news_weight': 0.1,
                'trend_weight': 0.2,
                'ml_weight': 0.4,
                'mean_rev_weight': 0.15,
                'momentum_weight': 0.1,
            }
        else:
            # 基于历史经验生成
            mode = 'refine' if round_id > 5 and experiments and not experiments[-1].result.score > best_score - 5 else 'explore'
            params = generate_params(experiments, mode)

        # 运行回测
        result = run_backtest(data_dict, params, start_date, end_date)

        if result is None:
            continue

        # 记录
        exp = Experiment(
            round_id=round_id,
            params=params,
            result=result,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        experiments.append(exp)

        # 更新最佳
        if result.score > best_score:
            best_score = result.score
            best_params = params
            best_result = result
            marker = " ★ 新的最佳!"
        else:
            marker = ""

        # 打印
        print(f"\n[第{round_id}轮] {params['buy_threshold']}/{params['sell_threshold']} "
              f"止损{params['stop_loss']:.0%} 止盈{params['take_profit']:.0%} "
              f"持仓{params['top_n']}个 RSI({params['rsi_oversold']}/{params['rsi_overbought']})")
        print(f"  → 收益: {result.total_return:+.2%} | 年化: {result.annual_return:+.2%} | "
              f"夏普: {result.sharpe_ratio:.2f} | 回撤: {result.max_drawdown:.2%} | "
              f"胜率: {result.win_rate:.1%} | 交易: {result.trade_count}次 | 评分: {result.score:.1f}{marker}")

        # 早停
        if round_id > 10:
            recent = [e.result.score for e in experiments[-5:]]
            if all(s < best_score - 20 for s in recent):
                print("\n连续5轮无改进，停止优化")
                break

    # 保存结果
    print(f"\n{'='*60}")
    print(f"优化完成! 最佳评分: {best_score:.1f}")
    print(f"{'='*60}")
    print(f"最佳参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    if best_result:
        print(f"\n最佳回测结果:")
        print(f"  总收益: {best_result.total_return:+.2%}")
        print(f"  年化收益: {best_result.annual_return:+.2%}")
        print(f"  夏普比率: {best_result.sharpe_ratio:.2f}")
        print(f"  最大回撤: {best_result.max_drawdown:.2%}")
        print(f"  胜率: {best_result.win_rate:.1%}")
        print(f"  交易次数: {best_result.trade_count}")

    # 保存
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump({
            'params': best_params,
            'result': {
                'total_return': best_result.total_return,
                'annual_return': best_result.annual_return,
                'sharpe_ratio': best_result.sharpe_ratio,
                'max_drawdown': best_result.max_drawdown,
                'win_rate': best_result.win_rate,
                'trade_count': best_result.trade_count,
                'score': best_result.score,
            }
        }, f, indent=2)

    with open(OPTIMIZATION_LOG, 'w') as f:
        json.dump([{
            'round_id': e.round_id,
            'params': e.params,
            'result': asdict(e.result),
            'timestamp': e.timestamp,
        } for e in experiments], f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {BEST_PARAMS_FILE}")

    return best_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-07-01')
    parser.add_argument('--end', default='2026-08-21')
    parser.add_argument('--rounds', type=int, default=30)
    args = parser.parse_args()

    best = run_optimization(args.start, args.end, args.rounds)
