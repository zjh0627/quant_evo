#!/usr/bin/env python3
"""
尾盘信号推送系统
===============
每天 15:50 推送当日信号

流程：
1. 获取当日实时数据，模拟日K
2. 计算技术指标信号
3. 生成买入/卖出信号
4. 保存信号到文件，供web端展示
5. 16:00执行交易
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import os
from datetime import datetime, time
import requests
import pandas as pd

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
SIGNAL_FILE = f'{CACHE_DIR}/daily_signals.json'
POSITIONS_FILE = f'{CACHE_DIR}/positions.json'


def log(msg: str):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')


def is_close_time() -> bool:
    """检查是否接近收盘 (15:00-16:00)"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return time(15, 0) <= t <= time(16, 0)


def get_sina_realtime(codes: list) -> dict:
    """获取新浪实时价格"""
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
                            'open': float(data[1]),
                            'high': float(data[4]),
                            'low': float(data[5]),
                            'close': float(data[3]),
                            'prev_close': float(data[2]),
                            'volume': float(data[8]) if len(data) > 8 else 0,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        }
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        log(f'获取实时价格失败: {e}')
    return prices


def load_stock_pool() -> tuple:
    """加载股票池"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('stocks', [])[:50], pool.get('names', {})
    return ['sh.600519', 'sh.600036', 'sh.601318', 'sh.000001'], {}


def load_local_kline(code: str, days: int = 60) -> pd.DataFrame:
    """从Parquet加载历史K线"""
    code_fmt = code.replace('.', '_')
    filepath = f'{CACHE_DIR}/kline_{code_fmt}.parquet'
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df.tail(days)
    return None


def load_positions() -> list:
    """加载持仓"""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return []


def calculate_daily_signal(df: pd.DataFrame, today_price: dict) -> dict:
    """计算日线信号
    使用历史日K + 今日模拟K线
    """
    if df is None or len(df) < 30:
        return None

    # 合并历史和今日数据
    today = pd.DataFrame([{
        'date': pd.to_datetime(today_price.get('date', datetime.now().strftime('%Y-%m-%d'))),
        'open': today_price.get('open', today_price.get('close')),
        'high': today_price.get('high', today_price.get('close')),
        'low': today_price.get('low', today_price.get('close')),
        'close': today_price.get('close'),
        'volume': today_price.get('volume', 0)
    }])

    df_combined = pd.concat([df, today], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset='date', keep='last')

    close = df_combined['close'].values
    volume = df_combined['volume'].values
    high = df_combined['high'].values
    low = df_combined['low'].values

    if len(close) < 30:
        return None

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
    boll_position = (close[-1] - lower_band) / (upper_band - lower_band) if upper_band > lower_band else 0.5

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

    # 综合评分
    score = 50

    # RSI
    if rsi < 25:
        score += 20  # 严重超卖
    elif rsi < 30:
        score += 15
    elif rsi < 40:
        score += 8
    elif rsi > 75:
        score -= 20
    elif rsi > 70:
        score -= 15
    elif rsi > 60:
        score -= 8

    # 均线趋势
    if ma5 > ma20:
        score += 10
    else:
        score -= 10

    if ma20 > ma60:
        score += 5  # 长期上升

    # 动量
    if mom_5d > 3:
        score += 8
    elif mom_5d < -3:
        score -= 8

    if mom_20d > 10:
        score += 5
    elif mom_20d < -10:
        score -= 5

    # 布林带
    if boll_position < 0.2:
        score += 10
    elif boll_position > 0.8:
        score -= 10

    # 成交量
    if vol_ratio > 2:
        score += 5

    # MACD
    if macd_hist > 0:
        score += 5

    score = max(0, min(100, score))

    # 趋势判断
    trend_intact = ma5 > ma20

    # 信号 - 更严格筛选
    if score >= 65 and rsi < 60:
        signal = 'BUY'
    elif score <= 40 or rsi > 70:
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
        'trend_intact': bool(trend_intact),
        'close': float(close[-1]),
        'today_change': float((close[-1] - close[-2]) / close[-2] * 100 if len(close) >= 2 else 0)
    }


def generate_daily_signals():
    """生成当日信号"""
    log('=' * 50)
    log(f'尾盘信号生成 {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    log('=' * 50)

    codes, names = load_stock_pool()
    positions = load_positions()
    position_codes = [p['code'] for p in positions]

    log(f'获取实时价格...')
    realtime_prices = get_sina_realtime(codes)
    log(f'获取到 {len(realtime_prices)} 只股票价格')

    if len(realtime_prices) == 0:
        log('获取价格失败')
        return None

    signals = []

    for code in codes:
        # 加载历史K线
        df = load_local_kline(code)
        price_info = realtime_prices.get(code)

        if price_info is None:
            continue

        signal_data = calculate_daily_signal(df, price_info)
        if signal_data is None:
            continue

        signal_data['code'] = code
        signal_data['name'] = names.get(code, code)
        signal_data['in_portfolio'] = code in position_codes

        signals.append(signal_data)

        # 打印信号
        if signal_data['signal'] != 'HOLD':
            emoji = '🟢' if signal_data['signal'] == 'BUY' else '🔴'
            log(f'{emoji} {signal_data["name"]}: {signal_data["signal"]} '
                f'(评分:{signal_data["score"]:.0f}, RSI:{signal_data["rsi"]:.1f}, '
                f'MA5>{signal_data["ma5"]:.2f}, 涨跌:{signal_data["today_change"]:+.2f}%)')

    # 排序
    signals.sort(key=lambda x: (0 if x['signal'] == 'BUY' else 1 if x['signal'] == 'SELL' else 2, -x['score']))

    # 保存信号
    signal_result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'signals': signals,
        'summary': {
            'buy_count': sum(1 for s in signals if s['signal'] == 'BUY'),
            'sell_count': sum(1 for s in signals if s['signal'] == 'SELL'),
            'hold_count': sum(1 for s in signals if s['signal'] == 'HOLD'),
            'portfolio_count': len(positions)
        }
    }

    with open(SIGNAL_FILE, 'w') as f:
        json.dump(signal_result, f, ensure_ascii=False, indent=2)

    log(f'信号已保存: 买入={signal_result["summary"]["buy_count"]}, '
        f'卖出={signal_result["summary"]["sell_count"]}, '
        f'持有={signal_result["summary"]["hold_count"]}')

    return signal_result


def execute_trades():
    """执行交易 (收盘价)"""
    log('=' * 50)
    log('执行交易 (尾盘)')
    log('=' * 50)

    codes, names = load_stock_pool()
    positions = load_positions()
    position_codes = [p['code'] for p in positions]

    # 获取收盘价
    realtime_prices = get_sina_realtime(codes)
    if not realtime_prices:
        log('获取价格失败')
        return

    # 加载信号
    if not os.path.exists(SIGNAL_FILE):
        log('无信号文件')
        return

    with open(SIGNAL_FILE, 'r') as f:
        signal_data = json.load(f)

    signals = signal_data.get('signals', [])
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']

    trades = []

    # 执行卖出 (T+1: 昨天买的今天可以卖)
    for sig in sell_signals[:3]:
        if sig['code'] in position_codes:
            pos = next((p for p in positions if p['code'] == sig['code']), None)
            if pos:
                price = realtime_prices.get(sig['code'], {}).get('close', sig['close'])
                pnl = (price - pos['entry_price']) / pos['entry_price']
                revenue = pos['shares'] * price * 0.9997
                trades.append({
                    'action': 'SELL',
                    'code': sig['code'],
                    'name': sig['name'],
                    'price': price,
                    'shares': pos['shares'],
                    'pnl': pnl
                })
                log(f'卖出 {sig["name"]} @ {price:.2f} ({pnl*100:+.2f}%)')

    # 执行买入
    for sig in buy_signals[:3]:
        if len(positions) >= 7:
            break
        if sig['code'] in position_codes:
            continue

        price = sig['close']
        position_size = 0.12
        max_shares = int(2000000 * position_size / price / 100) * 100
        if max_shares < 100:
            continue

        shares = max(100, int(max_shares * (sig['score'] / 100) / 100) * 100)
        cost = shares * price * 1.0003

        if shares < 100:
            continue

        positions.append({
            'code': sig['code'],
            'name': sig['name'],
            'shares': shares,
            'entry_price': price,
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'current_price': price
        })

        trades.append({
            'action': 'BUY',
            'code': sig['code'],
            'name': sig['name'],
            'price': price,
            'shares': shares
        })
        log(f'买入 {sig["name"]} @ {price:.2f} x {shares}股')

    # 保存持仓
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)

    log(f'交易完成: {len(trades)}笔')

    return trades


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate':
            # 只生成信号
            generate_daily_signals()
        elif sys.argv[1] == 'execute':
            # 执行交易
            execute_trades()
        elif sys.argv[1] == 'push':
            # 生成信号并推送
            result = generate_daily_signals()
            if result:
                print('\n📱 信号推送内容:')
                print(f'日期: {result["date"]}')
                print(f'时间: {result["timestamp"]}')
                print(f'买入信号: {result["summary"]["buy_count"]}只')
                print(f'卖出信号: {result["summary"]["sell_count"]}只')
                print('\n详细信号:')
                for sig in result['signals'][:10]:
                    if sig['signal'] != 'HOLD':
                        emoji = '🟢' if sig['signal'] == 'BUY' else '🔴'
                        print(f'{emoji} {sig["name"]}: {sig["signal"]} '
                              f'(评分:{sig["score"]:.0f}, RSI:{sig["rsi"]:.1f})')
    else:
        # 完整流程
        if is_close_time():
            result = generate_daily_signals()
            if result:
                execute_trades()
        else:
            generate_daily_signals()


if __name__ == '__main__':
    main()
