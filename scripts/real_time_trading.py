#!/usr/bin/env python3
"""
实时交易监控系统
================
交易日 9:30-15:00 每15分钟运行一次

功能：
1. 获取实时价格
2. 生成交易信号
3. 执行买入/卖出
4. 止损止盈检查
5. 记录交易日志
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import os
from datetime import datetime, time
import time as time_module
from typing import Dict, List, Tuple
import requests
import pandas as pd

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
POSITIONS_FILE = f'{CACHE_DIR}/positions.json'
TRADE_LOG_FILE = f'{CACHE_DIR}/real_time_trades.json'
PYTHON_BIN = '/Users/keira/project/claude/quant_evo/.venv/bin/python'


def is_market_open() -> bool:
    """检查当前是否在交易时间"""
    now = datetime.now()
    # 周一到周五，交易时间 9:30-15:00
    if now.weekday() >= 5:  # 周末
        return False
    current_time = now.time()
    market_start = time(9, 30)
    market_end = time(15, 0)
    return market_start <= current_time <= market_end


def log(msg: str):
    """打印日志"""
    now = datetime.now().strftime('%H:%M:%S')
    print(f'[{now}] {msg}')


def get_sina_realtime_prices(codes: list) -> dict:
    """从新浪财经获取实时价格"""
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
                        prev_close = float(data[2])
                        current = float(data[3])
                        open_price = float(data[1])
                        high = float(data[4])
                        low = float(data[5])
                        prices[code] = {
                            'close': current,
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'prev_close': prev_close,
                            'change': (current - prev_close) / prev_close * 100 if prev_close > 0 else 0,
                            'volume': float(data[8]) if len(data) > 8 else 0
                        }
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        log(f'获取实时价格失败: {e}')
    return prices


def load_stock_pool() -> List[str]:
    """加载股票池"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('stocks', [])[:50]
    return ['sh.600519', 'sh.600036', 'sh.601318', 'sh.000001']


def load_stock_names() -> Dict[str, str]:
    """加载股票名称映射"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('names', {})
    return {}


def load_local_kline(code: str, days: int = 60) -> pd.DataFrame:
    """从Parquet加载本地K线数据"""
    code_fmt = code.replace('.', '_')
    filepath = f'{CACHE_DIR}/kline_{code_fmt}.parquet'
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        df['date'] = pd.to_datetime(df['date'])
        # 合并实时价格
        return df.tail(days)
    return None


def calculate_realtime_signal(df: pd.DataFrame, current_price: float, prev_close: float) -> dict:
    """计算实时交易信号"""
    if df is None or len(df) < 20:
        return None

    close = df['close'].values.copy()
    volume = df['volume'].values.copy()

    # 用实时价格更新最后一天
    if current_price > 0:
        close[-1] = current_price

    # RSI (14日)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss != 0 else 50
    rsi = 100 - (100 / (1 + rs))
    if pd.isna(rsi): rsi = 50

    # MACD (12, 26, 9)
    ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
    macd = (ema12 - ema26) * 2
    signal_line = pd.Series(close).ewm(span=9).mean().iloc[-1]
    macd_hist = macd - signal_line

    # 布林带
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    std20 = pd.Series(close).rolling(20).std().iloc[-1]
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20
    boll_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band > lower_band else 0.5

    # 均线
    ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    ma60 = pd.Series(close).rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20

    # 动量
    mom_5d = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
    mom_20d = (close[-1] - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0

    # 成交量异动
    vol_ma5 = pd.Series(volume).rolling(5).mean().iloc[-1]
    vol_ratio = volume[-1] / vol_ma5 if vol_ma5 > 0 else 1

    # 综合评分 (0-100)
    score = 50

    # RSI 评分
    if rsi < 25:
        score += 20  # 严重超卖
    elif rsi < 30:
        score += 15  # 超卖
    elif rsi < 40:
        score += 8
    elif rsi > 75:
        score -= 20  # 严重超买
    elif rsi > 70:
        score -= 15  # 超买
    elif rsi > 60:
        score -= 8

    # 均线趋势
    if ma5 > ma20:
        score += 10  # 上升趋势
    else:
        score -= 10  # 下降趋势

    # 动量
    if mom_5d > 3:
        score += 8
    elif mom_5d < -3:
        score -= 8

    if mom_20d > 10:
        score += 5
    elif mom_20d < -10:
        score -= 5

    # 布林带位置
    if boll_position < 0.2:
        score += 10  # 接近下轨，超卖
    elif boll_position > 0.8:
        score -= 10  # 接近上轨，超买

    # 成交量异动
    if vol_ratio > 2:
        score += 5  # 放量

    # MACD
    if macd_hist > 0:
        score += 5
    else:
        score -= 5

    # 限制在0-100
    score = max(0, min(100, score))

    # 交易信号 - 更激进的阈值
    if score >= 58 and rsi < 60:
        signal = 'BUY'
    elif score <= 42 or rsi > 72:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'rsi': float(rsi),
        'macd': float(macd),
        'macd_hist': float(macd_hist),
        'boll_position': float(boll_position),
        'ma5': float(ma5),
        'ma20': float(ma20),
        'ma60': float(ma60),
        'momentum_5d': float(mom_5d),
        'momentum_20d': float(mom_20d),
        'vol_ratio': float(vol_ratio),
        'score': float(score),
        'signal': signal,
        'close': float(current_price),
        'prev_close': float(prev_close)
    }


def load_positions() -> List[Dict]:
    """加载持仓"""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_positions(positions: List[Dict]):
    """保存持仓"""
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def load_trade_log() -> List[Dict]:
    """加载交易日志"""
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'r') as f:
            return json.load(f)
    return []


def save_trade_log(logs: List[Dict]):
    """保存交易日志"""
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)  # 只保留最近1000条


def execute_buy(positions: List[Dict], signal: Dict, prices: Dict, cash: float) -> Tuple[List[Dict], float, Dict]:
    """执行买入"""
    code = signal['code']
    name = signal['name']
    current = signal['close']

    # 获取股票中文名
    names = load_stock_names()
    display_name = names.get(code, name)

    # 读取默认参数
    from src.simulation.evolving_portfolio import DEFAULT_PARAMS
    top_n = DEFAULT_PARAMS['top_n']
    position_size = DEFAULT_PARAMS.get('position_size', 0.12)

    # 检查是否已持仓
    if any(p['code'] == code for p in positions):
        return positions, cash, None

    # 检查持仓数量限制
    if len(positions) >= top_n:
        return positions, cash, None

    # 计算买入数量 (100股整倍数)
    max_shares = int(cash * position_size / current / 100) * 100
    if max_shares < 100:
        return positions, cash, None

    # 使用评分加权确定仓位
    score_weight = signal['score'] / 100
    shares = max(100, int(max_shares * score_weight / 100) * 100)

    cost = shares * current * 1.0003  # 手续费
    if cost > cash:
        shares = int(cash / current / 100) * 100
        cost = shares * current * 1.0003

    if shares < 100:
        return positions, cash, None

    # 创建持仓
    # 计算买入时趋势
    df = load_local_kline(code)
    if df is not None and len(df) >= 20:
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        trend_intact = ma5 > ma20
    else:
        trend_intact = True

    position = {
        'code': code,
        'name': display_name,
        'shares': shares,
        'entry_price': current,
        'entry_date': datetime.now().strftime('%Y-%m-%d'),
        'current_price': current,
        'pnl_pct': 0,
        'stop_loss': DEFAULT_PARAMS.get('stop_loss', 0.05),
        'take_profit': DEFAULT_PARAMS.get('take_profit', 0.06),
        'reason': signal['signal'],
        'trend_intact': bool(trend_intact)  # 买入时趋势状态
    }

    positions.append(position)
    cash -= cost

    trade_record = {
        'time': datetime.now().isoformat(),
        'action': 'BUY',
        'code': code,
        'name': display_name,
        'price': current,
        'shares': shares,
        'cash': cash,
        'signal_score': signal['score'],
        'rsi': signal['rsi'],
        'reason': '实时信号'
    }

    log(f'买入 {display_name} @ {current:.2f} x {shares}股 (评分:{signal["score"]:.0f}, RSI:{signal["rsi"]:.1f})')

    return positions, cash, trade_record


def execute_sell(positions: List[Dict], code: str, signal: Dict, prices: Dict, reason: str) -> Tuple[List[Dict], float, Dict]:
    """执行卖出"""
    pos = next((p for p in positions if p['code'] == code), None)
    if not pos:
        return positions, 0, None

    current_price_dict = prices.get(code, {})
    current_price = current_price_dict.get('close', pos.get('current_price', pos['entry_price']))
    pnl = (current_price - pos['entry_price']) / pos['entry_price']

    revenue = pos['shares'] * current_price * 0.9997  # 手续费+印花税
    positions.remove(pos)

    trade_record = {
        'time': datetime.now().isoformat(),
        'action': 'SELL',
        'code': code,
        'name': pos['name'],
        'price': current_price,
        'shares': pos['shares'],
        'pnl_pct': pnl,
        'reason': reason,
        'rsi': signal.get('rsi') if signal else None
    }

    log(f'卖出 {pos["name"]} @ {current_price:.2f} ({pnl*100:+.2f}%) - {reason}')

    return positions, revenue, trade_record


def check_stop_loss_take_profit(positions: List[Dict], prices: Dict, df: pd.DataFrame = None) -> Tuple[List[Dict], List[Dict]]:
    """检查止损止盈 - 趋势跟踪+移动止损模式
    - 止损：3%固定止损
    - 移动止损：盈利>10%后，启动5%移动止损
    - 趋势破坏(MA5<MA20)且盈利>2%时止盈
    """
    actions = []
    new_positions = []

    for pos in positions:
        code = pos['code']
        current = prices.get(code, {})
        current_price = current.get('close', pos.get('current_price', pos['entry_price']))
        pnl = (current_price - pos['entry_price']) / pos['entry_price']

        pos['current_price'] = current_price
        pos['pnl_pct'] = pnl

        # 移动止损：盈利>10%后启动
        stop_loss = 0.03  # 基础止损3%
        if pnl > 0.10:  # 盈利超过10%
            stop_loss = max(0.03, pnl - 0.05)  # 移动止损：保持在盈利-5%位置

        # 止损 - 始终生效
        if pnl < -stop_loss:
            reason = '移动止损' if pnl > 0.05 else '止损'
            actions.append({
                'action': reason,
                'code': code,
                'name': pos['name'],
                'price': current,
                'pnl': pnl
            })
            new_positions.append(('sell', code, reason))
            continue

        # 止盈：趋势破坏(MA5<MA20)且盈利>2%
        trend_intact = pos.get('trend_intact', True)
        if not trend_intact and pnl > 0.02:
            actions.append({
                'action': '趋势破坏',
                'code': code,
                'name': pos['name'],
                'price': current,
                'pnl': pnl
            })
            new_positions.append(('sell', code, '趋势破坏'))
            continue

        new_positions.append(('hold', code, None))

    return new_positions, actions


def run_real_time_trading():
    """运行实时交易"""
    if not is_market_open():
        log('不在交易时间，跳过')
        return

    now = datetime.now()
    log('=' * 50)
    log(f'实时交易扫描 {now.strftime("%Y-%m-%d %H:%M:%S")}')
    log('=' * 50)

    # 加载持仓和现金
    from src.simulation.evolving_portfolio import DEFAULT_PARAMS
    positions = load_positions()
    cash = 2000000 - sum(p['shares'] * p['entry_price'] for p in positions)

    # 加载股票池
    codes = load_stock_pool()
    names = load_stock_names()

    # 获取实时价格
    prices = get_sina_realtime_prices(codes)
    log(f'获取实时价格: {len(prices)} 只')

    if len(prices) == 0:
        log('获取价格失败')
        return

    # 计算所有信号
    signals = []
    for code in codes:
        df = load_local_kline(code)
        if df is None or len(df) < 20:
            continue

        price_info = prices.get(code, {})
        current = price_info.get('close', 0)
        prev_close = price_info.get('prev_close', df['close'].iloc[-1])

        if current <= 0:
            continue

        signal = calculate_realtime_signal(df, current, prev_close)
        if signal:
            signal['code'] = code
            signal['name'] = names.get(code, code)
            signals.append(signal)

    # 按评分排序
    signals.sort(key=lambda x: -x['score'])

    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']

    log(f'信号: 买入={len(buy_signals)}, 卖出={len(sell_signals)}, 持仓={len(positions)}')

    # 计算持仓趋势 (MA5 > MA20 为上升趋势)
    for pos in positions:
        df = load_local_kline(pos['code'])
        if df is not None and len(df) >= 20:
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            pos['trend_intact'] = bool(ma5 > ma20)  # True = 上升趋势
        else:
            pos['trend_intact'] = True  # 默认上升

    # 检查持仓的止损止盈
    updates, actions = check_stop_loss_take_profit(positions, prices)

    trade_records = []

    # 执行止损止盈
    for action_type, code, reason in updates:
        if action_type == 'sell':
            signal = next((s for s in signals if s['code'] == code), None)
            positions, revenue, record = execute_sell(positions, code, signal, prices, reason)
            if record:
                trade_records.append(record)
                cash += revenue

    # 执行买入信号
    for signal in buy_signals[:3]:  # 最多买3只
        if len(positions) >= DEFAULT_PARAMS['top_n']:
            break
        positions, cash, record = execute_buy(positions, signal, prices, cash)
        if record:
            trade_records.append(record)

    # 执行卖出信号
    positions_codes = [p['code'] for p in positions]
    for signal in sell_signals:
        if signal['code'] in positions_codes:
            positions, revenue, record = execute_sell(positions, signal['code'], signal, prices, '卖出信号')
            if record:
                trade_records.append(record)
                cash += revenue

    # 保存结果
    save_positions(positions)

    # 记录交易
    if trade_records:
        logs = load_trade_log()
        logs.extend(trade_records)
        save_trade_log(logs)

    # 打印持仓状态
    log(f'持仓: {len(positions)}/{DEFAULT_PARAMS["top_n"]}, 现金: ¥{cash:,.0f}')
    for pos in positions:
        pnl = pos['pnl_pct'] * 100
        log(f'  {pos["name"]}: {pos["current_price"]:.2f} ({pnl:+.1f}%)')

    log(f'本次交易: {len(trade_records)} 笔')


def main():
    """主函数"""
    if not is_market_open():
        now = datetime.now()
        if now.weekday() >= 5:
            log(f'周末 ({now.strftime("%A")})，不在交易时间')
        else:
            log(f'非交易时间 (9:30-15:00)，跳过')
        return

    run_real_time_trading()


if __name__ == '__main__':
    main()
