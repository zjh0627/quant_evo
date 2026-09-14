#!/usr/bin/env python3
"""
quant_evo Web UI
Web Dashboard for monitoring the quant system
"""

from flask import Flask, render_template, jsonify, request
import json
import os
import numpy as np
from datetime import datetime

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
STATE_FILE = f'{PROJECT_ROOT}/data/cache/system_state.json'
RESEARCH_LOG = f'{PROJECT_ROOT}/data/cache/research_log.json'
BEST_PARAMS = f'{PROJECT_ROOT}/data/cache/best_params.json'
METADATA_FILE = f'{PROJECT_ROOT}/data/cache/metadata.json'

app = Flask(__name__, template_folder='templates', static_folder='static')

def load_json(filepath, default=None):
    """加载JSON文件"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    return default

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/todo')
def todo():
    """Sprint 看板"""
    return render_template('todo.html')

@app.route('/portfolio')
def portfolio():
    """模拟盘"""
    return render_template('portfolio.html')

@app.route('/api/status')
def api_status():
    """系统状态"""
    state = load_json(STATE_FILE, {})
    metadata = load_json(METADATA_FILE, {})

    return jsonify({
        'status': state.get('status', 'unknown'),
        'current_round': state.get('current_round', 0),
        'best_score': state.get('best_score', 0),
        'last_update': state.get('last_update', 'N/A'),
        'stock_count': metadata.get('count', 0),
        'start_date': metadata.get('start_date', ''),
        'end_date': metadata.get('end_date', '')
    })

@app.route('/api/research')
def api_research():
    """研究结果"""
    research_log = load_json(RESEARCH_LOG, {})
    best_params = load_json(BEST_PARAMS, {})

    experiments = research_log.get('experiments', [])

    return jsonify({
        'experiments': experiments[-20:],  # 最近20轮
        'total_rounds': len(experiments),
        'best_score': research_log.get('best_score', 0),
        'best_params': best_params,
        'best_metrics': research_log.get('best_metrics', {})
    })

@app.route('/api/stocks')
def api_stocks():
    """股票池状态"""
    metadata = load_json(METADATA_FILE, {})
    codes = metadata.get('codes', [])

    # 获取每只股票的最新评分
    from src.factors.technical import FactorCalculator
    from src.data.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    stocks = []

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 0:
            calc = FactorCalculator()
            factors = calc.calculate_all_factors(df)
            if not factors.empty:
                score = calc.get_composite_score(factors)
                signal = calc.generate_signal(factors)
                latest = factors.iloc[-1]

                stocks.append({
                    'code': code,
                    'name': get_stock_name(code),
                    'close': float(latest.get('close', 0)),
                    'score': score,
                    'signal': signal,
                    'momentum_5d': float(latest.get('momentum_5d', 0) * 100),
                    'rsi': float(latest.get('rsi', 50)),
                    'date': str(df['date'].iloc[-1]) if 'date' in df.columns else str(latest.get('date', ''))
                })

    # 按评分排序
    stocks.sort(key=lambda x: x['score'], reverse=True)

    return jsonify({'stocks': stocks})


@app.route('/api/signals')
def api_signals():
    """实时交易信号"""
    import pandas as pd
    from src.simulation.evolving_portfolio import load_stock_data, calculate_enhanced_features, get_stock_name, STOCK_NAMES

    # 从本地CSV加载数据
    try:
        pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
        if os.path.exists(pool_path):
            with open(pool_path, 'r') as f:
                pool = json.load(f)
                codes = pool.get('stocks', [])[:50]
        else:
            codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

        data_dict = load_stock_data(codes)

        signals = []
        for code, df in data_dict.items():
            if len(df) < 60:
                continue

            try:
                close = df['close'].values

                # 计算RSI
                delta = pd.Series(close).diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
                rs = gain / loss if loss != 0 else 50
                rsi = 100 - (100 / (1 + rs))
                if pd.isna(rsi): rsi = 50

                # 计算动量
                mom_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
                mom_5d = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0

                # 计算趋势
                ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
                ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
                uptrend = ma20 > ma60

                # 综合评分
                score = 50
                if rsi < 30:
                    score += 20
                    signal = 'BUY'
                    reason = f'RSI超卖({rsi:.1f})'
                elif rsi > 70:
                    score -= 20
                    signal = 'SELL'
                    reason = f'RSI超买({rsi:.1f})'
                elif uptrend and mom_20d > 0.05:
                    score += 15
                    signal = 'BUY'
                    reason = f'趋势向上({mom_20d:.1%})'
                elif not uptrend and mom_20d < -0.05:
                    score -= 15
                    signal = 'SELL'
                    reason = f'趋势向下({mom_20d:.1%})'
                else:
                    signal = 'HOLD'
                    reason = f'RSI中性({rsi:.1f})'

                if mom_20d > 0.1:
                    score += 10
                elif mom_20d < -0.1:
                    score -= 10

                score = max(0, min(100, score))

                signals.append({
                    'code': code,
                    'name': get_stock_name(code),
                    'close': float(close[-1]),
                    'signal': signal,
                    'score': float(score),
                    'rsi': float(rsi),
                    'momentum': float(mom_20d * 100),
                    'reason': reason,
                    'date': df['date'].iloc[-1]
                })
            except Exception as e:
                continue

        # 按信号和评分排序
        signal_order = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
        signals.sort(key=lambda x: (signal_order[x['signal']], -x['score']))

        return jsonify({
            'signals': signals,
            'count': len(signals),
            'buy_count': sum(1 for s in signals if s['signal'] == 'BUY'),
            'sell_count': sum(1 for s in signals if s['signal'] == 'SELL'),
            'hold_count': sum(1 for s in signals if s['signal'] == 'HOLD'),
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e), 'signals': [], 'count': 0})


@app.route('/api/signals/detailed')
def api_signals_detailed():
    """详细信号 - 包含RSI/MACD/布林带/KDJ等多指标"""
    import pandas as pd
    from src.simulation.evolving_portfolio import load_stock_data, get_stock_name

    try:
        pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
        if os.path.exists(pool_path):
            with open(pool_path, 'r') as f:
                pool = json.load(f)
                codes = pool.get('stocks', [])[:50]
        else:
            codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

        data_dict = load_stock_data(codes)

        signals = []
        for code, df in data_dict.items():
            if len(df) < 60:
                continue

            try:
                close = df['close'].values
                high = df['high'].values
                low = df['low'].values
                vol = df['volume'].values if 'volume' in df.columns else None

                # RSI
                delta = pd.Series(close).diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
                rs = gain / loss if loss != 0 else 50
                rsi = 100 - (100 / (1 + rs))
                if pd.isna(rsi): rsi = 50

                # MACD
                ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
                ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
                macd = ema12 - ema26
                signal_line = pd.Series(close).ewm(span=9, adjust=False).mean().iloc[-1]
                macd_hist = macd - signal_line

                # 布林带
                ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
                std20 = pd.Series(close).rolling(20).std().iloc[-1]
                upper_band = ma20 + 2 * std20
                lower_band = ma20 - 2 * std20
                boll_position = (close[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5

                # KDJ
                low_9 = pd.Series(low).rolling(9).min().iloc[-1]
                high_9 = pd.Series(high).rolling(9).max().iloc[-1]
                k = 50
                d = 50
                if high_9 != low_9:
                    rsv = (close[-1] - low_9) / (high_9 - low_9) * 100
                    k = 2/3 * k + 1/3 * rsv
                    d = 2/3 * d + 1/3 * k
                j = 3 * k - 2 * d

                # 动量
                mom_5d = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
                mom_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

                # 趋势
                ma60 = pd.Series(close).rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
                uptrend = bool(ma20 > ma60)

                # ATR (Average True Range)
                high_low = pd.Series(high) - pd.Series(low)
                high_close = np.abs(pd.Series(high) - pd.Series(close).shift())
                low_close = np.abs(pd.Series(low) - pd.Series(close).shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                atr_ratio = atr / close[-1] if close[-1] != 0 else 0

                # ADX (Average Directional Index)
                high_diff = pd.Series(high).diff()
                low_diff = -pd.Series(low).diff()
                plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0).rolling(14).mean()
                minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0).rolling(14).mean()
                plus_di = (plus_dm / atr * 100).iloc[-1] if atr > 0 else 0
                minus_di = (minus_dm / atr * 100).iloc[-1] if atr > 0 else 0
                dx_series = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else pd.Series([0])
                adx = dx_series if isinstance(dx_series, (int, float)) else dx_series.iloc[-1]
                adx_signal = 'long' if plus_di > minus_di else 'short'
                adx_strength = 'strong' if adx > 25 else 'weak'

                # 资金流因子
                if vol is not None and len(vol) > 20:
                    tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
                    money_flow = tp * pd.Series(vol)
                    money_inflow = money_flow.where(pd.Series(close) > pd.Series(close).shift(), 0).rolling(20).sum().iloc[-1]
                    money_outflow = money_flow.where(pd.Series(close) < pd.Series(close).shift(), 0).rolling(20).sum().iloc[-1]
                    money_net = money_inflow - money_outflow
                    money_strength = money_net / money_flow.rolling(20).sum().iloc[-1] if money_flow.rolling(20).sum().iloc[-1] != 0 else 0
                    money_state = 'inflow' if money_net > 0 else 'outflow'
                else:
                    money_net = 0
                    money_strength = 0
                    money_state = 'neutral'

                # 信号判断
                score = 50
                reasons = []

                # RSI分析
                rsi_state = 'neutral'
                if rsi < 30:
                    score += 25
                    rsi_state = 'oversold'
                    reasons.append(f'RSI超卖({rsi:.0f})')
                elif rsi > 70:
                    score -= 25
                    rsi_state = 'overbought'
                    reasons.append(f'RSI超买({rsi:.0f})')
                elif rsi < 40:
                    reasons.append(f'RSI偏弱({rsi:.0f})')
                elif rsi > 60:
                    reasons.append(f'RSI偏强({rsi:.0f})')

                # MACD分析
                macd_state = 'neutral'
                if macd_hist > 0 and macd > signal_line:
                    score += 15
                    macd_state = 'golden_cross'
                    reasons.append('MACD金叉')
                elif macd_hist < 0 and macd < signal_line:
                    score -= 15
                    macd_state = 'death_cross'
                    reasons.append('MACD死叉')

                # 布林带分析
                boll_state = 'middle'
                if close[-1] <= lower_band:
                    score += 15
                    boll_state = 'lower_band'
                    reasons.append('布林下轨支撑')
                elif close[-1] >= upper_band:
                    score -= 15
                    boll_state = 'upper_band'
                    reasons.append('布林上轨压力')
                elif boll_position < 0.3:
                    reasons.append('布林下轨附近')
                elif boll_position > 0.7:
                    reasons.append('布林上轨附近')

                # KDJ分析
                kdj_state = 'neutral'
                if k < 20:
                    score += 10
                    kdj_state = 'oversold'
                    reasons.append(f'KDJ超卖(K={k:.0f})')
                elif k > 80:
                    score -= 10
                    kdj_state = 'overbought'
                    reasons.append(f'KDJ超买(K={k:.0f})')

                # 趋势分析
                if uptrend:
                    score += 10
                    reasons.append('上升趋势')
                else:
                    score -= 10
                    reasons.append('下降趋势')

                # 动量分析
                if mom_20d > 0.1:
                    score += 10
                    reasons.append(f'强势动量({mom_20d:.1%})')
                elif mom_20d < -0.1:
                    score -= 10
                    reasons.append(f'弱势动量({mom_20d:.1%})')

                score = max(0, min(100, score))

                # 综合信号
                if score >= 70:
                    signal = 'BUY'
                elif score <= 30:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'

                signals.append({
                    'code': code,
                    'name': get_stock_name(code),
                    'close': float(close[-1]),
                    'signal': signal,
                    'score': float(score),
                    'rsi': float(rsi),
                    'rsi_state': rsi_state,
                    'macd': float(macd),
                    'macd_signal': float(signal_line),
                    'macd_hist': float(macd_hist),
                    'macd_state': macd_state,
                    'boll_upper': float(upper_band),
                    'boll_middle': float(ma20),
                    'boll_lower': float(lower_band),
                    'boll_position': float(boll_position),
                    'boll_state': boll_state,
                    'kdj_k': float(k),
                    'kdj_d': float(d),
                    'kdj_j': float(j),
                    'kdj_state': kdj_state,
                    'momentum_5d': float(mom_5d * 100),
                    'momentum_20d': float(mom_20d * 100),
                    'uptrend': uptrend,
                    'atr': float(atr),
                    'atr_ratio': float(atr_ratio),
                    'adx': float(adx),
                    'adx_signal': adx_signal,
                    'adx_strength': adx_strength,
                    'money_net': float(money_net),
                    'money_state': money_state,
                    'reason': ' + '.join(reasons[:3]) if reasons else '暂无明显信号',
                    'date': df['date'].iloc[-1]
                })
            except Exception as e:
                continue

        # 按信号和评分排序
        signal_order = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
        signals.sort(key=lambda x: (signal_order[x['signal']], -x['score']))

        return jsonify({
            'signals': signals,
            'count': len(signals),
            'buy_count': sum(1 for s in signals if s['signal'] == 'BUY'),
            'sell_count': sum(1 for s in signals if s['signal'] == 'SELL'),
            'hold_count': sum(1 for s in signals if s['signal'] == 'HOLD'),
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e), 'signals': [], 'count': 0})


@app.route('/api/review/trades')
def api_review_trades():
    """交易复盘 - 按月统计"""
    trade_file = f'{PROJECT_ROOT}/data/cache/trade_history.json'
    trades = []
    if os.path.exists(trade_file):
        with open(trade_file, 'r') as f:
            trades = json.load(f)

    if not trades:
        return jsonify({'periods': [], 'summary': {}})

    # 按月分组
    periods = {}
    for t in trades:
        month = t['date'][:7]  # YYYY-MM
        if month not in periods:
            periods[month] = []
        periods[month].append(t)

    # 统计每月数据
    monthly_stats = []
    for month in sorted(periods.keys(), reverse=True):
        month_trades = periods[month]
        buy_trades = [t for t in month_trades if t['action'] == 'BUY']
        sell_trades = [t for t in month_trades if t['action'] == 'SELL']
        win_trades = [t for t in sell_trades if t.get('pnl_pct', 0) > 0]

        # 计算盈亏金额
        total_pnl = sum(t.get('pnl_amount', 0) for t in sell_trades)
        win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0

        # 止损止盈次数
        stop_loss_count = len([t for t in sell_trades if t.get('reason') == 'STOP_LOSS'])
        take_profit_count = len([t for t in sell_trades if t.get('reason') == 'TAKE_PROFIT'])

        # 最佳/最差交易
        if sell_trades:
            best = max(sell_trades, key=lambda x: x.get('pnl_pct', 0))
            worst = min(sell_trades, key=lambda x: x.get('pnl_pct', 0))
        else:
            best = worst = None

        # 平均持仓天数
        holding_days = []
        for t in sell_trades:
            if t.get('entry_date') and t.get('date'):
                try:
                    entry = datetime.strptime(t['entry_date'], '%Y-%m-%d')
                    exit = datetime.strptime(t['date'], '%Y-%m-%d')
                    holding_days.append((exit - entry).days)
                except:
                    pass
        avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0

        monthly_stats.append({
            'month': month,
            'total_trades': len(month_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'stop_loss_count': stop_loss_count,
            'take_profit_count': take_profit_count,
            'best_trade': best,
            'worst_trade': worst,
            'avg_holding_days': avg_holding
        })

    # 总体统计
    all_sell = [t for t in trades if t['action'] == 'SELL']
    all_win = [t for t in all_sell if t.get('pnl_pct', 0) > 0]
    summary = {
        'total_trades': len(trades),
        'total_buy': len([t for t in trades if t['action'] == 'BUY']),
        'total_sell': len(all_sell),
        'overall_win_rate': len(all_win) / len(all_sell) if all_sell else 0,
        'total_pnl': sum(t.get('pnl_amount', 0) for t in all_sell),
        'best_trade': max(all_sell, key=lambda x: x.get('pnl_pct', 0)) if all_sell else None,
        'worst_trade': min(all_sell, key=lambda x: x.get('pnl_pct', 0)) if all_sell else None,
    }

    return jsonify({
        'periods': monthly_stats,
        'summary': summary
    })


@app.route('/api/params', methods=['GET', 'POST'])
def api_params():
    """参数管理"""
    if request.method == 'POST':
        params = request.json
        with open(BEST_PARAMS, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return jsonify({'success': True})

    best_params = load_json(BEST_PARAMS, {})
    return jsonify({'params': best_params})

@app.route('/api/start-research', methods=['POST'])
def api_start_research():
    """启动研究"""
    # 这里会触发后台的研究循环
    return jsonify({'success': True, 'message': '研究已启动'})

@app.route('/api/notify/config', methods=['GET', 'POST'])
def api_notify_config():
    """配置通知（微信/飞书）"""
    from src.notification.wechat_notify import load_config, save_config

    if request.method == 'POST':
        data = request.json
        wechat_webhook = data.get('wechat_webhook', '')
        lark_webhook = data.get('lark_webhook', '')
        notify_type = data.get('type', 'lark')  # wechat 或 lark
        enabled = data.get('enabled', True)
        push_interval = data.get('interval', 60)

        config = {
            'notify_type': notify_type,
            'wechat_webhook': wechat_webhook,
            'lark_webhook': lark_webhook,
            'enabled': enabled,
            'push_interval': push_interval
        }
        save_config(config)
        return jsonify({'success': True, 'message': '配置已保存'})

    config = load_config()
    return jsonify({
        'type': config.get('notify_type', 'lark'),
        'wechat_webhook': config.get('wechat_webhook', ''),
        'lark_webhook': config.get('lark_webhook', ''),
        'enabled': config.get('enabled', False),
        'interval': config.get('push_interval', 60)
    })

@app.route('/api/notify/test', methods=['POST'])
def api_notify_test():
    """测试通知"""
    from src.notification.lark_notify import LarkNotifier
    from src.notification.wechat_notify import WeChatNotifier

    data = request.json
    notify_type = data.get('type', 'lark')
    webhook = data.get('webhook', '')

    if not webhook:
        return jsonify({'success': False, 'message': '请先配置 Webhook'})

    if notify_type == 'lark':
        notifier = LarkNotifier(webhook_url=webhook)
        success = notifier.send_card("🧪 量化系统飞书通知测试\n连接正常！")
    else:
        notifier = WeChatNotifier(webhook_url=webhook)
        success = notifier.send_text("🧪 量化系统微信通知测试\n连接正常！")

    return jsonify({'success': success, 'message': '发送成功' if success else '发送失败'})

@app.route('/api/notify/push', methods=['POST'])
def api_notify_push():
    """手动推送信号"""
    from src.notification.lark_notify import LarkNotifier
    from src.notification.wechat_notify import WeChatNotifier

    data = request.json or {}
    notify_type = data.get('type', 'lark')
    webhook = data.get('webhook', '')

    if notify_type == 'lark':
        notifier = LarkNotifier(webhook_url=webhook)
    else:
        notifier = WeChatNotifier(webhook_url=webhook)

    if not notifier.webhook_url:
        return jsonify({'success': False, 'message': '未配置 Webhook'})

    # 获取最新信号
    signals_data = api_signals()
    signals = signals_data.get_json()

    if 'error' in signals:
        return jsonify({'success': False, 'message': signals.get('error')})

    buy_signals = [s for s in signals.get('signals', []) if s['signal'] == 'BUY']
    sell_signals = [s for s in signals.get('signals', []) if s['signal'] == 'SELL']

    # 发送报告
    success = notifier.send_trade_signal_report(
        buy_count=signals.get('buy_count', 0),
        sell_count=signals.get('sell_count', 0),
        hold_count=signals.get('hold_count', 0),
        top_buys=buy_signals,
        top_sells=sell_signals
    )

    return jsonify({'success': success, 'message': '推送成功' if success else '推送失败'})

@app.route('/api/portfolio')
def api_portfolio():
    """模拟账户状态"""
    import requests
    from src.simulation.evolving_portfolio import load_portfolio, load_positions

    portfolio = load_portfolio()
    positions = load_positions()

    # 从新浪财经获取实时价格
    latest_prices = {}
    prev_prices = {}  # 昨日收盘价
    codes = list(set([p.code for p in positions]))
    if codes:
        try:
            # 批量查询
            code_str = ','.join([c.replace('sh.', 'sh').replace('sz.', 'sz') for c in codes])
            url = f'http://hq.sinajs.cn/list={code_str}'
            headers = {'Referer': 'http://finance.sina.com.cn'}
            r = requests.get(url, timeout=5, headers=headers)
            text = r.text
            # 解析每个股票的价格
            for code in codes:
                c = code.replace('sh.', 'sh').replace('sz.', 'sz')
                search = f'hq_str_{c}='
                if search in text:
                    start = text.find(search) + len(search) + 1
                    end = text.find('"', start)
                    data = text[start:end].split(',')
                    if len(data) > 3:
                        try:
                            latest_prices[code] = float(data[3])  # 当前价
                            prev_prices[code] = float(data[2])   # 昨收价
                        except:
                            pass
        except Exception as e:
            pass

    # 更新持仓价格
    for p in positions:
        if p.code in latest_prices:
            p.current_price = latest_prices[p.code]
            p.pnl_pct = (p.current_price - p.entry_price) / p.entry_price

    initial = 2000000
    positions_value = sum(p.shares * p.current_price for p in positions)
    total_value = portfolio.cash + positions_value
    ret_pct = (total_value - initial) / initial

    return jsonify({
        'total_value': total_value,
        'cash': portfolio.cash,
        'positions_value': positions_value,
        'return_pct': ret_pct,
        'position_count': len(positions),
        'positions': [
            {
                'code': p.code,
                'name': p.name,
                'shares': p.shares,
                'entry_date': p.entry_date,
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'pnl_pct': p.pnl_pct,
                'value': p.shares * p.current_price,
                'value_pct': (p.shares * p.current_price) / positions_value if positions_value > 0 else 0,
                'today_change': (latest_prices.get(p.code, p.current_price) - prev_prices.get(p.code, p.current_price)) / prev_prices.get(p.code, p.current_price) if prev_prices.get(p.code) else 0
            }
            for p in positions
        ],
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': 'sina',
        'equity_curve': portfolio.equity_curve[-30:] if portfolio.equity_curve else []
    })


@app.route('/api/stock_pool')
def api_stock_pool():
    """股票池列表 + 自选股"""
    import requests

    # 读取股票池
    pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
    stocks = []
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            stocks = pool.get('stocks', [])
            names = pool.get('names', {})

    # 自选股列表
    custom_file = f'{PROJECT_ROOT}/data/cache/custom_stocks.json'
    custom_stocks = []
    if os.path.exists(custom_file):
        with open(custom_file, 'r') as f:
            custom_stocks = json.load(f)

    # 获取实时价格
    if stocks:
        try:
            code_str = ','.join([c.replace('.', '') for c in stocks])
            url = f'http://hq.sinajs.cn/list={code_str}'
            headers = {'Referer': 'http://finance.sina.com.cn'}
            r = requests.get(url, timeout=5, headers=headers)
            r.encoding = 'gbk'
            text = r.text

            # 解析价格
            stock_prices = {}
            lines = text.strip().split('\n')
            for i, line in enumerate(lines):
                if '="' in line and i < len(stocks):
                    code = stocks[i]
                    data = line.split('="')[1].strip('";').split(',')
                    if len(data) > 3:
                        try:
                            prev_close = float(data[2])
                            current = float(data[3])
                            change = (current - prev_close) / prev_close * 100 if prev_close > 0 else 0
                            stock_prices[code] = {
                                'current': current,
                                'prev_close': prev_close,
                                'change': change,
                                'name': names.get(code, data[0])
                            }
                        except:
                            pass
        except:
            pass

    # 构建返回
    pool_list = []
    for code in stocks:
        info = stock_prices.get(code, {})
        pool_list.append({
            'code': code,
            'name': info.get('name', names.get(code, code)),
            'current': info.get('current', 0),
            'prev_close': info.get('prev_close', 0),
            'change': info.get('change', 0),
            'is_custom': code in custom_stocks
        })

    return jsonify({
        'stocks': pool_list,
        'custom_stocks': custom_stocks,
        'total': len(stocks)
    })


@app.route('/api/stocks/add', methods=['POST'])
def api_stocks_add():
    """添加自选股"""
    data = request.get_json()
    code = data.get('code', '').strip()

    if not code:
        return jsonify({'success': False, 'error': '股票代码不能为空'})

    # 验证格式
    if not (code.startswith('sh.') or code.startswith('sz.')):
        code = 'sh.' + code if code.startswith('6') else 'sz.' + code

    custom_file = f'{PROJECT_ROOT}/data/cache/custom_stocks.json'
    custom_stocks = []
    if os.path.exists(custom_file):
        with open(custom_file, 'r') as f:
            custom_stocks = json.load(f)

    if code not in custom_stocks:
        custom_stocks.append(code)
        with open(custom_file, 'w') as f:
            json.dump(custom_stocks, f)

    return jsonify({'success': True, 'custom_stocks': custom_stocks})


@app.route('/api/stocks/remove', methods=['POST'])
def api_stocks_remove():
    """删除自选股"""
    data = request.get_json()
    code = data.get('code', '').strip()

    custom_file = f'{PROJECT_ROOT}/data/cache/custom_stocks.json'
    if os.path.exists(custom_file):
        with open(custom_file, 'r') as f:
            custom_stocks = json.load(f)
        if code in custom_stocks:
            custom_stocks.remove(code)
            with open(custom_file, 'w') as f:
                json.dump(custom_stocks, f)

    return jsonify({'success': True, 'custom_stocks': custom_stocks})


@app.route('/api/custom_stocks/monitor')
def api_custom_stocks_monitor():
    """自选股监控 - 涨跌幅/信号提醒"""
    import requests
    import pandas as pd

    custom_file = f'{PROJECT_ROOT}/data/cache/custom_stocks.json'
    custom_stocks = []
    if os.path.exists(custom_file):
        with open(custom_file, 'r') as f:
            custom_stocks = json.load(f)

    if not custom_stocks:
        return jsonify({
            'custom_stocks': [],
            'alerts': [],
            'count': 0,
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    # 获取实时价格
    try:
        code_str = ','.join([c.replace('.', '') for c in custom_stocks])
        url = f'http://hq.sinajs.cn/list={code_str}'
        headers = {'Referer': 'http://finance.sina.com.cn'}
        r = requests.get(url, timeout=5, headers=headers)
        r.encoding = 'gbk'
        text = r.text

        stock_prices = {}
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            if i < len(custom_stocks):
                code = custom_stocks[i]
                if '="' in line:
                    data = line.split('="')[1].strip('";').split(',')
                    if len(data) > 3:
                        try:
                            prev_close = float(data[2])
                            current = float(data[3])
                            change = (current - prev_close) / prev_close * 100 if prev_close > 0 else 0
                            stock_prices[code] = {
                                'name': data[0],
                                'current': current,
                                'prev_close': prev_close,
                                'change': change
                            }
                        except:
                            pass
    except Exception as e:
        return jsonify({'error': str(e), 'custom_stocks': [], 'alerts': []})

    # 获取每只股票的技术信号
    from src.factors.technical import FactorCalculator
    from src.simulation.evolving_portfolio import load_stock_data

    data_dict = load_stock_data(custom_stocks)
    alerts = []
    stock_details = []

    for code in custom_stocks:
        info = stock_prices.get(code, {})
        current = info.get('current', 0)
        change = info.get('change', 0)

        stock_item = {
            'code': code,
            'name': info.get('name', code),
            'current': current,
            'prev_close': info.get('prev_close', 0),
            'change': change,
            'change_cls': 'positive' if change >= 0 else 'negative',
            'alerts': []
        }

        if code in data_dict:
            df = data_dict[code]
            if len(df) >= 20:
                calc = FactorCalculator()
                try:
                    factors = calc.calculate_all_factors(df)
                    if not factors.empty:
                        latest = factors.iloc[-1]
                        rsi = float(latest.get('rsi', 50))
                        score = calc.get_composite_score(factors)

                        stock_item['rsi'] = rsi
                        stock_item['score'] = score

                        # RSI超卖提醒
                        if rsi < 30:
                            alert = {'type': 'RSI超卖', 'level': 'warning', 'msg': f'RSI={rsi:.0f}，建议关注'}
                            alerts.append({**alert, 'code': code, 'name': stock_item['name']})
                            stock_item['alerts'].append(alert)
                        elif rsi > 70:
                            alert = {'type': 'RSI超买', 'level': 'danger', 'msg': f'RSI={rsi:.0f}，注意风险'}
                            alerts.append({**alert, 'code': code, 'name': stock_item['name']})
                            stock_item['alerts'].append(alert)

                        # 跌幅超过5%提醒
                        if change < -5:
                            alert = {'type': '大幅下跌', 'level': 'warning', 'msg': f'今日下跌{change:.1f}%，关注是否超跌'}
                            alerts.append({**alert, 'code': code, 'name': stock_item['name']})
                            stock_item['alerts'].append(alert)
                        elif change > 5:
                            alert = {'type': '大幅上涨', 'level': 'info', 'msg': f'今日上涨{change:.1f}%，注意获利'}
                            alerts.append({**alert, 'code': code, 'name': stock_item['name']})
                            stock_item['alerts'].append(alert)
                except:
                    pass

        stock_details.append(stock_item)

    # 按涨跌幅排序
    stock_details.sort(key=lambda x: x['change'], reverse=True)

    return jsonify({
        'custom_stocks': stock_details,
        'alerts': alerts,
        'count': len(custom_stocks),
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/portfolio/analysis')
def api_portfolio_analysis():
    """持仓分析 - 盈亏统计/风险指标"""
    from src.simulation.evolving_portfolio import load_portfolio, load_positions

    portfolio = load_portfolio()
    positions = load_positions()

    # 加载交易历史
    trade_file = f'{PROJECT_ROOT}/data/cache/trade_history.json'
    trades = []
    if os.path.exists(trade_file):
        with open(trade_file, 'r') as f:
            trades = json.load(f)

    # 基本统计
    total_value = portfolio.total_value
    initial = 2000000
    total_return = (total_value - initial) / initial

    # 持仓分布
    positions_value = total_value - portfolio.cash
    position_distribution = []
    for p in positions:
        value = p.shares * p.current_price
        pct = value / positions_value if positions_value > 0 else 0
        position_distribution.append({
            'name': p.name,
            'value': value,
            'pct': pct
        })

    # 交易统计
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    win_trades = [t for t in sell_trades if t.get('pnl_pct', 0) > 0]
    win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0

    # 盈亏统计
    if sell_trades:
        pnl_list = [t.get('pnl_pct', 0) * 100 for t in sell_trades]
        best_trade = max(sell_trades, key=lambda x: x.get('pnl_pct', 0))
        worst_trade = min(sell_trades, key=lambda x: x.get('pnl_pct', 0))
    else:
        pnl_list = []
        best_trade = None
        worst_trade = None

    # 最大回撤计算
    max_drawdown = 0
    if portfolio.equity_curve and len(portfolio.equity_curve) > 0:
        values = [e['total_value'] for e in portfolio.equity_curve]
        peak = values[0]
        for v in values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    # 夏普比率 (简化版: 使用日收益标准差)
    sharpe_ratio = 0
    if portfolio.equity_curve and len(portfolio.equity_curve) > 1:
        returns = []
        values = [e['total_value'] for e in portfolio.equity_curve]
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        if returns:
            import numpy as np
            mean_ret = np.mean(returns) * 252  # 年化
            std_ret = np.std(returns) * np.sqrt(252)
            if std_ret > 0:
                sharpe_ratio = mean_ret / std_ret

    return jsonify({
        'total_value': total_value,
        'cash': portfolio.cash,
        'positions_value': positions_value,
        'total_return': total_return,
        'return_pct': total_return * 100,
        'position_count': len(positions),
        'position_distribution': position_distribution,
        'trade_stats': {
            'total_trades': len(trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        },
        'risk_stats': {
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio
        },
        'equity_curve': portfolio.equity_curve[-30:] if portfolio.equity_curve else []
    })


@app.route('/api/portfolio/run', methods=['POST'])
def api_portfolio_run():
    """运行模拟盘"""
    from src.simulation.evolving_portfolio import EvolvingPortfolio

    portfolio = EvolvingPortfolio()
    portfolio.run_daily()

    return jsonify({'success': True, 'message': '模拟盘已运行'})

@app.route('/api/portfolio/evolve', methods=['POST'])
def api_portfolio_evolve():
    """优化策略参数"""
    from src.simulation.evolving_portfolio import EvolvingPortfolio
    from src.simulation.evolving_portfolio import load_stock_data
    import json

    portfolio = EvolvingPortfolio()

    # 加载数据
    pool_path = 'data/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:30]
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

    data_dict = load_stock_data(codes)
    params = portfolio.optimizer.evolve(data_dict, rounds=10)

    return jsonify({'success': True, 'params': params, 'score': portfolio.optimizer.best_score})

# 全局股票名称映射
STOCK_NAMES = {}

def load_stock_names():
    """从股票池加载名称映射"""
    global STOCK_NAMES
    PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
    pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            STOCK_NAMES = pool.get('names', {})
    return STOCK_NAMES

def get_stock_name(code: str) -> str:
    if not STOCK_NAMES:
        load_stock_names()
    return STOCK_NAMES.get(code, code)


@app.route('/api/trade_history')
def api_trade_history():
    """交易历史"""
    trade_file = f'{PROJECT_ROOT}/data/cache/trade_history.json'
    if os.path.exists(trade_file):
        with open(trade_file, 'r') as f:
            history = json.load(f)
        return jsonify({'trades': history, 'count': len(history)})
    return jsonify({'trades': [], 'count': 0})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置量化账户 - 清空所有数据"""
    from datetime import datetime

    CACHE_DIR = f'{PROJECT_ROOT}/data/cache'
    INITIAL_CASH = 2000000.0

    try:
        # 1. 重置 virtual_portfolio.json
        portfolio_data = {
            "cash": INITIAL_CASH,
            "total_value": INITIAL_CASH,
            "positions": [],
            "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "equity_curve": []
        }
        with open(f'{CACHE_DIR}/virtual_portfolio.json', 'w', encoding='utf-8') as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=2)

        # 2. 重置 positions.json
        with open(f'{CACHE_DIR}/positions.json', 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

        # 3. 重置 real_time_trades.json
        with open(f'{CACHE_DIR}/real_time_trades.json', 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

        return jsonify({
            'success': True,
            'message': '账户重置成功',
            'cash': INITIAL_CASH,
            'total_value': INITIAL_CASH
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'重置失败: {str(e)}'
        }), 500


@app.route('/api/live_signals')
def api_live_signals():
    """实时信号"""
    import subprocess
    try:
        # 运行实时扫描脚本
        result = subprocess.run(
            ['.venv39/bin/python', f'{PROJECT_ROOT}/scripts/live_simulation.py'],
            capture_output=True, text=True, timeout=300
        )
    except:
        pass

    # 读取结果
    signals_file = f'{PROJECT_ROOT}/data/cache/live_signals.json'
    if os.path.exists(signals_file):
        with open(signals_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({'signals': [], 'count': 0, 'timestamp': ''})

@app.route('/api/kline/<code>')
def api_kline(code):
    """获取股票K线数据"""
    import pandas as pd
    code_fmt = code.replace('.', '_')
    csv_file = f'{PROJECT_ROOT}/data/raw/kline_{code_fmt}.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return jsonify({
            'code': code,
            'name': get_stock_name(code),
            'data': df.to_dict('records')
        })
    return jsonify({'code': code, 'data': []})


@app.route('/api/market/indices')
def api_market_indices():
    """大盘指数 - 上证/深证/创业板/科创50 (使用新浪财经)"""
    import requests

    indices = [
        {'code': 'sh000001', 'name': '上证指数', 'sina_code': 'sh000001'},
        {'code': 'sz399001', 'name': '深证成指', 'sina_code': 'sz399001'},
        {'code': 'sz399006', 'name': '创业板指', 'sina_code': 'sz399006'},
        {'code': 'sh000688', 'name': '科创50', 'sina_code': 'sh000688'},
    ]

    results = []

    try:
        code_str = ','.join([idx['sina_code'] for idx in indices])
        url = f'http://hq.sinajs.cn/list={code_str}'
        headers = {'Referer': 'http://finance.sina.com.cn'}
        r = requests.get(url, timeout=5, headers=headers)
        r.encoding = 'gbk'
        text = r.text

        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            if i < len(indices):
                idx = indices[i]
                if '="' in line:
                    data = line.split('="')[1].strip('";').split(',')
                    if len(data) > 3:
                        try:
                            prev_close = float(data[2])
                            current = float(data[3])
                            pct_chg = (current - prev_close) / prev_close * 100 if prev_close > 0 else 0
                            results.append({
                                'code': idx['code'],
                                'name': idx['name'],
                                'close': current,
                                'pct_chg': pct_chg,
                                'change': current - prev_close,
                                'high': float(data[4]) if len(data) > 4 else current,
                                'low': float(data[5]) if len(data) > 5 else current,
                                'date': data[30] if len(data) > 30 else ''
                            })
                        except:
                            pass
    except Exception as e:
        print(f"获取大盘指数失败: {e}")

    return jsonify({
        'indices': results,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/market/sectors')
def api_market_sectors():
    """板块涨跌 - 从本地股票池按行业分组计算"""
    from src.simulation.evolving_portfolio import load_stock_data

    pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:50]
            names = pool.get('names', {})
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']
        names = {}

    data_dict = load_stock_data(codes)

    # 按涨跌幅排序
    sector_stats = []
    for code, df in data_dict.items():
        if len(df) >= 2:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            close = float(latest.get('close', 0))
            prev_close = float(prev.get('close', 0))
            if prev_close > 0:
                pct_chg = (close - prev_close) / prev_close * 100
                sector_stats.append({
                    'code': code,
                    'name': names.get(code, code),
                    'pct_chg': pct_chg,
                    'close': close
                })

    # 按涨跌幅排序
    sector_stats.sort(key=lambda x: x['pct_chg'], reverse=True)

    return jsonify({
        'gainers': sector_stats[:5],   # 涨幅前5
        'losers': sector_stats[-5:],   # 跌幅前5
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/news/signals')
def api_news_signals():
    """新闻驱动信号 (使用缓存,异步更新)"""
    signals_file = f'{PROJECT_ROOT}/data/cache/news_signals.json'

    # 优先返回缓存
    if os.path.exists(signals_file):
        file_time = os.path.getmtime(signals_file)
        cache_age = datetime.now().timestamp() - file_time

        with open(signals_file, 'r') as f:
            signals = json.load(f)

        # 缓存小于5分钟直接返回
        if cache_age < 300:
            return jsonify({
                'signals': signals,
                'count': len(signals),
                'updated': datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S'),
                'cached': True
            })

    # 缓存太旧,尝试返回已存在的缓存(不让前端一直等待)
    if os.path.exists(signals_file):
        with open(signals_file, 'r') as f:
            signals = json.load(f)
        return jsonify({
            'signals': signals,
            'count': len(signals),
            'updated': datetime.fromtimestamp(os.path.getmtime(signals_file)).strftime('%Y-%m-%d %H:%M:%S'),
            'cached': True
        })

    return jsonify({'signals': [], 'count': 0, 'updated': '', 'cached': False})


@app.route('/api/news/important')
def api_news_important():
    """重要新闻列表"""
    cache_file = f'{PROJECT_ROOT}/data/cache/news_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)

        items = cache.get('items', [])
        important = []
        for item in items:
            score = item.get('sentiment_score', 50)
            impact = item.get('impact_level', 'low')
            impact_mult = {'high': 1.4, 'medium': 1.1, 'low': 1.0}
            final_score = score * impact_mult.get(impact, 1.0)
            if final_score >= 60 or impact == 'high':
                important.append(item)

        important.sort(key=lambda x: (x.get('sentiment_score', 50) * (1.3 if x.get('impact_level') == 'high' else 1.0), x.get('published_at', '')), reverse=True)

        return jsonify({
            'news': important[:30],
            'count': len(important),
            'last_update': cache.get('last_update', ''),
            'total_in_cache': len(items)
        })

    return jsonify({'news': [], 'count': 0, 'last_update': '', 'total_in_cache': 0})


@app.route('/api/news/refresh', methods=['POST'])
def api_news_refresh():
    """手动刷新新闻"""
    from src.news.news_agent import NewsAgent

    agent = NewsAgent(push_enabled=True)
    result = agent.run()

    return jsonify({
        'success': True,
        'new_news': result['new_news'],
        'important_news': result['important_news'],
        'signals': result['signals'],
        'message': f"获取{result['new_news']}条新闻, 生成{result['signals']}条信号"
    })


@app.route('/api/news/config', methods=['GET', 'POST'])
def api_news_config():
    """新闻配置"""
    from src.news.news_agent import NEWS_CONFIG

    config_file = f'{PROJECT_ROOT}/data/cache/news_config.json'
    default = {
        'enabled': True,
        'min_impact_score': 60,
        'lark_webhook': ''
    }

    if request.method == 'POST':
        data = request.json
        config = {**default, **data}
        with open(config_file, 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True})

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = default

    return jsonify(config)


if __name__ == '__main__':
    print("启动 quant_evo Web UI...")
    print("访问 http://localhost:5124")
    app.run(host='0.0.0.0', port=5124, debug=True)
