#!/usr/bin/env python3
"""
回测运行器 - 从指定日期开始运行模拟盘回测
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

from src.simulation.evolving_portfolio import (
    load_stock_data, calculate_enhanced_features, get_stock_name,
    EnhancedModel, MultiStrategySignal, Position, Portfolio,
    save_positions, save_portfolio, DEFAULT_PARAMS
)
from src.ml.portfolio_optimizer import MVOPortfolio

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'

PORTFOLIO_FILE = f'{CACHE_DIR}/virtual_portfolio.json'
POSITION_FILE = f'{CACHE_DIR}/positions.json'
TRADE_HISTORY_FILE = f'{CACHE_DIR}/trade_history.json'

# 交易历史
trade_history = []


def get_trade_time():
    """获取交易时间戳"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def reset_portfolio():
    """重置模拟盘"""
    portfolio = Portfolio(
        cash=2000000,
        total_value=2000000,
        positions=[],
        last_update='',
        equity_curve=[]
    )
    save_portfolio(portfolio)
    save_positions([])
    print("模拟盘已重置")


def get_trading_dates(data_dict: Dict[str, pd.DataFrame], start_date: str) -> List[str]:
    """获取交易日期列表"""
    all_dates = set()
    for df in data_dict.values():
        dates = df['date'].tolist()
        all_dates.update(dates)

    dates = sorted([d for d in all_dates if d >= start_date])
    return dates


def run_backtest(start_date: str = '2026-07-01', end_date: str = None):
    """运行回测"""
    print(f"\n{'='*60}")
    print(f"回测: {start_date} -> {end_date or '至今'}")
    print(f"{'='*60}")

    # 重置
    reset_portfolio()

    # 加载股票池
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:100]
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

    print(f"加载股票: {len(codes)} 只")
    data_dict = load_stock_data(codes)
    print(f"有效数据: {len(data_dict)} 只")

    # 加载模型
    model = EnhancedModel()
    model.load()
    signal_gen = MultiStrategySignal(model)

    # 参数
    params = DEFAULT_PARAMS.copy()

    # 交易日
    trading_dates = get_trading_dates(data_dict, start_date)
    if end_date:
        trading_dates = [d for d in trading_dates if d <= end_date]

    print(f"交易天数: {len(trading_dates)}")
    print(f"起始日期: {trading_dates[0] if trading_dates else 'N/A'}")
    print(f"结束日期: {trading_dates[-1] if trading_dates else 'N/A'}")

    # 初始化组合
    cash = 2000000
    positions = []
    equity_curve = []
    total_value = 2000000

    # 防回转机制初始化
    from datetime import timedelta
    cooldown = {}  # code -> cooldown_end_date (after sell/stop_loss)
    min_hold_days = params.get('min_hold_days', 3)
    sell_cooldown_days = params.get('sell_cooldown_days', 5)
    stop_loss_cooldown_days = params.get('stop_loss_cooldown_days', 10)

    def add_to_cooldown(code, days, date):
        """添加股票到冷却期"""
        entry = pd.to_datetime(date)
        cooldown_end = (entry + timedelta(days=days)).strftime('%Y-%m-%d')
        cooldown[code] = cooldown_end

    def is_in_cooldown(code, date):
        """检查股票是否在冷却期"""
        if code not in cooldown:
            return False
        return date < cooldown[code]

    def get_hold_days(entry_date, current_date):
        """计算持仓天数"""
        entry = pd.to_datetime(entry_date)
        current = pd.to_datetime(current_date)
        return (current - entry).days

    for date in trading_dates:
        # 先更新持仓价格和组合总值
        positions_value = sum(p.shares * p.current_price for p in positions)
        total_value = cash + positions_value

        # 获取当日数据
        daily_data = {}
        for code, df in data_dict.items():
            df_date = df[df['date'] <= date]
            if len(df_date) >= 60:
                daily_data[code] = df_date

        if not daily_data:
            continue

        # 计算信号
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

                # 更新持仓最高价(用于移动止损)
                if not hasattr(pos, 'max_price') or pos.max_price is None:
                    pos.max_price = pos.entry_price
                if current_price > pos.max_price:
                    pos.max_price = current_price

                use_trailing = params.get('use_trailing_stop', True)
                trailing_pct = params.get('trailing_stop_pct', 0.08)

                # 计算从最高价的回撤
                drawdown_from_high = (pos.max_price - current_price) / pos.max_price if pos.max_price > 0 else 0

                # 止损判断(固定止损或移动止损)
                stop_triggered = False
                stop_reason = ''
                if pnl_pct < -params['stop_loss']:
                    stop_triggered = True
                    stop_reason = 'FIXED_STOP'
                elif use_trailing and drawdown_from_high > trailing_pct and pos.max_price > pos.entry_price * 1.05:
                    # 移动止损:从最高价回落超过trailing_pct且有浮盈
                    stop_triggered = True
                    stop_reason = 'TRAILING_STOP'

                if stop_triggered:
                    revenue = pos.shares * current_price * 0.999
                    cash += revenue
                    trade_history.append({
                        'date': date,
                        'time': get_trade_time(),
                        'action': 'SELL',
                        'reason': stop_reason,
                        'code': pos.code,
                        'name': pos.name,
                        'price': current_price,
                        'shares': pos.shares,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': revenue - pos.shares * pos.entry_price * 1.0004
                    })
                    # 防回转：止损后加入冷却期
                    add_to_cooldown(pos.code, stop_loss_cooldown_days, date)
                    positions.remove(pos)
                    print(f"[{date}] {stop_reason} {pos.name} @ {current_price:.2f} ({pnl_pct:.1%})")
                    continue

                # 止盈
                if pnl_pct > params['take_profit']:
                    revenue = pos.shares * current_price * 0.999
                    cash += revenue
                    trade_history.append({
                        'date': date,
                        'time': get_trade_time(),
                        'action': 'SELL',
                        'reason': 'TAKE_PROFIT',
                        'code': pos.code,
                        'name': pos.name,
                        'price': current_price,
                        'shares': pos.shares,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': revenue - pos.shares * pos.entry_price * 1.0004
                    })
                    # 止盈后也需要冷却（防止马上又买回）
                    add_to_cooldown(pos.code, sell_cooldown_days, date)
                    positions.remove(pos)
                    print(f"[{date}] 止盈 {pos.name} @ {current_price:.2f} ({pnl_pct:.1%})")
                    continue

                pos.current_price = current_price
                pos.pnl_pct = pnl_pct

        # 卖出信号（检查最小持仓期）
        for pos in positions[:]:
            if pos.code in sell_signals:
                hold_days = get_hold_days(pos.entry_date, date)
                # 防回转：检查最小持仓天数
                if hold_days < min_hold_days:
                    continue  # 持有期不足，跳过卖出信号
                sig = sell_signals[pos.code]
                if sig['price'] > 0:
                    revenue = pos.shares * sig['price'] * 0.999
                    pnl = (sig['price'] - pos.entry_price) / pos.entry_price
                    trade_history.append({
                        'date': date,
                        'time': get_trade_time(),
                        'action': 'SELL',
                        'reason': 'SIGNAL',
                        'code': pos.code,
                        'name': pos.name,
                        'price': sig['price'],
                        'shares': pos.shares,
                        'pnl_pct': pnl,
                        'pnl_amount': revenue - pos.shares * pos.entry_price * 1.0004
                    })
                    # 防回转：卖出后加入冷却期
                    add_to_cooldown(pos.code, sell_cooldown_days, date)
                    cash += revenue
                    positions.remove(pos)
                    print(f"[{date}] 卖出 {pos.name} @ {sig['price']:.2f}")

        # 买入 - 评分加权仓位管理（过滤冷却期股票）
        if buy_signals and len(positions) < int(params['top_n']):
            # 构建候选列表（过滤冷却期）
            candidates = []
            for code, sig in buy_signals.items():
                if sig['price'] <= 0:
                    continue
                # 防回转：跳过冷却期股票
                if is_in_cooldown(code, date):
                    continue
                # 评分加权权重：分数越高权重越大
                score_weight = max(1, sig['score'] - 40)  # score范围约40-70
                candidates.append({
                    'code': code,
                    'sig': sig,
                    'score': sig['score'],
                    'score_weight': score_weight
                })

            if candidates:
                # 计算权重
                n = len(candidates)
                total_weight = sum(c['score_weight'] for c in candidates)
                weights = [c['score_weight'] / total_weight for c in candidates]

                # 基础金额
                base_position_value = total_value * params['position_size']

                # MVO仓位分配
                use_mvo = params.get('use_mvo_allocation', False)
                if use_mvo and len(candidates) >= 2:
                    # 使用Mean-Variance优化
                    try:
                        mvo = MVOPortfolio(
                            risk_aversion=1.0,
                            max_weight=0.20,
                            sector_max_weight=0.35,
                            lookback_days=60
                        )
                        scores_dict = {c['code']: c['score'] for c in candidates}
                        mvo_weights = mvo.calculate_weights(daily_data, scores_dict)

                        if mvo_weights:
                            print(f"  [MVO] 优化权重: { {k[:8]: f'{v:.1%}' for k, v in list(mvo_weights.items())[:5]} }")

                        # 按MVO权重分配
                        for c in candidates:
                            if len(positions) >= int(params['top_n']):
                                break
                            code = c['code']
                            sig = c['sig']
                            weight = mvo_weights.get(code, 0)

                            position_value = total_value * weight * params['position_size'] * 2
                            shares = max(100, int(position_value / sig['price'] / 100) * 100)
                            cost = shares * sig['price'] * 1.0004

                            if cost <= cash:
                                cash -= cost
                                positions.append(Position(
                                    code=code,
                                    name=get_stock_name(code),
                                    shares=shares,
                                    entry_date=date,
                                    entry_price=sig['price'],
                                    current_price=sig['price'],
                                    pnl_pct=0
                                ))
                                trade_history.append({
                                    'date': date,
                                    'time': get_trade_time(),
                                    'action': 'BUY',
                                    'reason': 'MVO',
                                    'code': code,
                                    'name': get_stock_name(code),
                                    'price': sig['price'],
                                    'shares': shares,
                                    'pnl_pct': 0,
                                    'pnl_amount': 0
                                })
                                print(f"[{date}] 买入(MVO) {get_stock_name(code)} @ {sig['price']:.2f} (权重:{weight:.1%})")
                        continue  # 跳过下面的分数加权分配
                    except Exception as e:
                        print(f"  [MVO] 优化失败: {e}, 使用分数加权")

                # 按分数加权分配仓位
                for c, w in zip(candidates, weights):
                    if len(positions) >= int(params['top_n']):
                        break

                    code = c['code']
                    sig = c['sig']
                    # 权重大的分更多金额
                    position_value = base_position_value * (1 + w * (n - 1))
                    shares = max(100, int(position_value / sig['price'] / 100) * 100)
                    cost = shares * sig['price'] * 1.0004

                    if cost <= cash:
                        cash -= cost
                        positions.append(Position(
                            code=code,
                            name=get_stock_name(code),
                            shares=shares,
                            entry_date=date,
                            entry_price=sig['price'],
                            current_price=sig['price'],
                            pnl_pct=0
                        ))
                        trade_history.append({
                            'date': date,
                            'time': get_trade_time(),
                            'action': 'BUY',
                            'reason': 'SIGNAL',
                            'code': code,
                            'name': get_stock_name(code),
                            'price': sig['price'],
                            'shares': shares,
                            'pnl_pct': 0,
                            'pnl_amount': 0
                        })
                        print(f"[{date}] 买入 {get_stock_name(code)} @ {sig['price']:.2f} (评分:{sig['score']:.0f})")

        # 更新组合
        positions_value = sum(p.shares * p.current_price for p in positions)
        total_value = cash + positions_value

        equity_curve.append({
            'date': date,
            'total_value': total_value,
            'cash': cash,
            'positions_value': positions_value
        })

        # 每5天打印状态
        if len(equity_curve) % 5 == 0 or date == trading_dates[-1]:
            ret_pct = (total_value - 2000000) / 2000000
            print(f"[{date}] 总值: {total_value:,.0f} ({ret_pct:+.1%}) | 持仓: {len(positions)} | 现金: {cash:,.0f}")

    # 合并同股票持仓（计算加权平均成本）
    merged_positions = {}
    for p in positions:
        if p.code in merged_positions:
            mp = merged_positions[p.code]
            # 加权平均成本
            total_cost = mp['shares'] * mp['entry_price'] + p.shares * p.entry_price
            mp['shares'] += p.shares
            mp['entry_price'] = total_cost / mp['shares']
            # 使用最早的买入日期
            if p.entry_date < mp['entry_date']:
                mp['entry_date'] = p.entry_date
            mp['current_price'] = p.current_price
        else:
            merged_positions[p.code] = {
                'code': p.code,
                'name': p.name,
                'shares': p.shares,
                'entry_date': p.entry_date,
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'pnl_pct': p.pnl_pct
            }

    portfolio = Portfolio(
        cash=cash,
        total_value=total_value,
        positions=list(merged_positions.values()),
        last_update=trading_dates[-1] if trading_dates else '',
        equity_curve=equity_curve
    )

    save_portfolio(portfolio)
    # 保存合并后的持仓（转换为Position对象）
    merged_list = []
    for mp in merged_positions.values():
        merged_list.append(Position(
            code=mp['code'],
            name=mp['name'],
            shares=mp['shares'],
            entry_date=mp['entry_date'],
            entry_price=mp['entry_price'],
            current_price=mp['current_price'],
            pnl_pct=mp['pnl_pct']
        ))
    save_positions(merged_list)

    # 保存交易历史
    with open(TRADE_HISTORY_FILE, 'w') as f:
        json.dump(trade_history, f, ensure_ascii=False, indent=2)

    # 最终结果
    print(f"\n{'='*60}")
    print("回测结果")
    print(f"{'='*60}")
    print(f"起始日期: {trading_dates[0] if trading_dates else 'N/A'}")
    print(f"结束日期: {trading_dates[-1] if trading_dates else 'N/A'}")
    print(f"交易天数: {len(trading_dates)}")
    print(f"初始资金: 2,000,000")
    print(f"最终价值: {total_value:,.0f}")
    print(f"收益率: {(total_value/2000000-1):+.2%}")
    print(f"持仓数: {len(positions)}")

    # 计算年化收益
    if len(trading_dates) > 1:
        days = (pd.to_datetime(trading_dates[-1]) - pd.to_datetime(trading_dates[0])).days
        if days > 0:
            annual_return = (total_value / 2000000) ** (365 / days) - 1
            print(f"年化收益: {annual_return:+.2%}")
            print(f"交易天数: {days} 天")

    return equity_curve, positions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-07-01', help='开始日期')
    parser.add_argument('--end', default=None, help='结束日期')
    args = parser.parse_args()

    run_backtest(start_date=args.start, end_date=args.end)