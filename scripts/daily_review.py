#!/usr/bin/env python3
"""
每日反思脚本
============
按照autoresearch规则，每天16:00（收盘后）自动运行

反思步骤：
1. 数据更新 - 获取最新K线数据
2. 信号扫描 - 获取今日实时信号
3. 持仓检查 - 检查止损止盈
4. 回测运行 - 运行当日回测
5. 参数评估 - 评估当前策略表现
6. 优化决策 - 是否需要调整参数
7. 结果记录 - 保存反思日志
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
LOG_FILE = f'{CACHE_DIR}/daily_review_log.json'
PYTHON_BIN = '/Users/keira/project/claude/quant_evo/.venv/bin/python'
RETRO_DIR = '/Users/keira/keira-data/量化系统/retro'


def log(msg: str):
    """打印日志"""
    now = datetime.now().strftime('%H:%M:%S')
    print(f'[{now}] {msg}')


def load_stock_pool() -> List[str]:
    """加载股票池"""
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            return pool.get('stocks', [])[:50]
    return ['sh.600519', 'sh.600036', 'sh.601318', 'sh.000001']


def step1_update_data() -> bool:
    """步骤1: 更新数据"""
    log('='*50)
    log('步骤1: 更新K线数据')

    try:
        from src.data.data_pipeline import DataPipeline

        pipeline = DataPipeline()
        codes = load_stock_pool()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        data = pipeline.fetch_and_save(codes, start_date, end_date, use_cache=False)
        log(f'  成功更新 {len(data)} 只股票的数据')
        return True
    except Exception as e:
        log(f'  数据更新失败: {e}')
        return False


def step2_scan_signals() -> Dict:
    """步骤2: 扫描实时信号"""
    log('='*50)
    log('步骤2: 扫描实时信号')

    try:
        import subprocess
        result = subprocess.run(
            [PYTHON_BIN, 'scripts/live_simulation.py'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        # 读取信号文件
        signals_file = f'{CACHE_DIR}/live_signals.json'
        with open(signals_file, 'r') as f:
            signals_data = json.load(f)

        signals = signals_data.get('signals', [])
        buy_count = sum(1 for s in signals if s.get('signal') == 'BUY')
        sell_count = sum(1 for s in signals if s.get('signal') == 'SELL')

        log(f'  买入信号: {buy_count} 只')
        log(f'  卖出信号: {sell_count} 只')

        return signals_data
    except Exception as e:
        log(f'  信号扫描失败: {e}')
        return {}


def step3_check_positions(signals: Dict) -> Tuple[List, float]:
    """步骤3: 检查持仓，执行止损止盈"""
    log('='*50)
    log('步骤3: 检查持仓')

    from src.simulation.evolving_portfolio import Position, save_positions
    import requests

    # 加载持仓
    positions_file = f'{CACHE_DIR}/positions.json'
    with open(positions_file, 'r') as f:
        positions = [Position(**p) for p in json.load(f)]

    if not positions:
        log('  无持仓，跳过')
        return [], 0

    # 获取实时价格
    codes = [p.code for p in positions]
    code_str = ','.join([c.replace('.', '') for c in codes])

    try:
        r = requests.get(f'http://hq.sinajs.cn/list={code_str}', timeout=5,
                        headers={'Referer': 'http://finance.sina.com.cn'})
        r.encoding = 'gbk'
        lines = r.text.strip().split('\n')

        prices = {}
        for i, line in enumerate(lines):
            if '="' in line and i < len(codes):
                data = line.split('="')[1].strip('";').split(',')
                if len(data) > 3:
                    prices[codes[i]] = float(data[3])
    except Exception as e:
        log(f'  获取实时价格失败: {e}')
        prices = {p.code: p.current_price for p in positions}

    # 加载参数
    from src.simulation.evolving_portfolio import DEFAULT_PARAMS

    cash = 0
    actions = []
    new_positions = positions.copy()

    for pos in positions:
        code = pos.code
        current = prices.get(code, pos.current_price)
        pnl = (current - pos.entry_price) / pos.entry_price

        # 止损检查
        if pnl < -DEFAULT_PARAMS['stop_loss']:
            revenue = pos.shares * current * 0.999
            cash += revenue
            new_positions.remove(pos)
            actions.append({
                'action': '止损',
                'code': code,
                'name': pos.name,
                'price': current,
                'pnl': pnl
            })
            log(f'  止损 {pos.name} @ {current:.2f} ({pnl:.1%})')
            continue

        # 止盈检查
        if pnl > DEFAULT_PARAMS['take_profit']:
            revenue = pos.shares * current * 0.999
            cash += revenue
            new_positions.remove(pos)
            actions.append({
                'action': '止盈',
                'code': code,
                'name': pos.name,
                'price': current,
                'pnl': pnl
            })
            log(f'  止盈 {pos.name} @ {current:.2f} ({pnl:.1%})')
            continue

        # 卖出信号检查
        sig = next((s for s in signals.get('signals', []) if s.get('code') == code), None)
        if sig and sig.get('signal') == 'SELL':
            revenue = pos.shares * current * 0.999
            cash += revenue
            new_positions.remove(pos)
            actions.append({
                'action': '卖出信号',
                'code': code,
                'name': pos.name,
                'price': current,
                'pnl': pnl
            })
            log(f'  卖出 {pos.name} @ {current:.2f} ({pnl:.1%})')
            continue

        # 更新价格
        pos.current_price = current
        pos.pnl_pct = pnl

    # 保存新持仓
    save_positions(new_positions)
    log(f'  持仓数: {len(new_positions)}/{DEFAULT_PARAMS["top_n"]}')

    return actions, cash


def step4_run_backtest() -> Dict:
    """步骤4: 运行回测"""
    log('='*50)
    log('步骤4: 运行回测')

    try:
        import subprocess
        result = subprocess.run(
            [PYTHON_BIN, 'scripts/run_backtest.py', '--start',
             (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
             '--end', datetime.now().strftime('%Y-%m-%d')],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=PROJECT_ROOT
        )

        # 读取回测结果
        portfolio_file = f'{CACHE_DIR}/virtual_portfolio.json'
        with open(portfolio_file, 'r') as f:
            portfolio = json.load(f)

        return {
            'total_value': portfolio.get('total_value', 0),
            'cash': portfolio.get('cash', 0),
            'equity_curve': portfolio.get('equity_curve', [])
        }
    except Exception as e:
        log(f'  回测失败: {e}')
        return {}


def step5_evaluate_params(backtest_result: Dict) -> Tuple[bool, str]:
    """步骤5: 评估当前参数"""
    log('='*50)
    log('步骤5: 评估策略参数')

    equity = backtest_result.get('equity_curve', [])
    if len(equity) < 5:
        log('  数据不足，跳过评估')
        return False, '数据不足'

    # 计算近期表现
    recent = equity[-10:] if len(equity) >= 10 else equity
    start_val = recent[0]['total_value']
    end_val = recent[-1]['total_value']
    recent_return = (end_val - start_val) / start_val

    # 计算最大回撤
    peak = recent[0]['total_value']
    max_drawdown = 0
    for e in recent:
        if e['total_value'] > peak:
            peak = e['total_value']
        dd = (peak - e['total_value']) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    log(f'  近期收益: {recent_return:+.2%}')
    log(f'  近期最大回撤: {max_drawdown:.2%}')

    # 评估标准
    needs_adjustment = False
    reason = ''

    if recent_return < -0.05:  # 亏损超过5%
        needs_adjustment = True
        reason = '亏损超过5%'
    elif recent_return > 0.10:  # 盈利超过10%
        needs_adjustment = True
        reason = '盈利良好，可考虑加仓'
    elif max_drawdown > 0.10:  # 回撤超过10%
        needs_adjustment = True
        reason = '回撤过大，需调整风控'

    if needs_adjustment:
        log(f'  建议: {reason}')
    else:
        log(f'  状态: 正常')

    return needs_adjustment, reason


def step6_optimize_params(need_adjust: bool, reason: str) -> bool:
    """步骤6: 优化参数（如果需要）"""
    if not need_adjust:
        return False

    log('='*50)
    log('步骤6: 优化策略参数')

    try:
        from src.research.research_loop import AutoResearchLoop

        research = AutoResearchLoop(use_ai=False)

        # 加载数据
        stock_data = research._load_stock_data()
        if len(stock_data) < 10:
            log('  数据不足，跳过优化')
            return False

        # 回测区间
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # 参数候选
        param_candidates = [
            # 当前参数
            {'top_n': 7, 'buy_threshold': 50, 'sell_threshold': 48,
             'stop_loss': 0.05, 'take_profit': 0.06, 'position_size': 0.12},
            # 更保守
            {'top_n': 6, 'buy_threshold': 52, 'sell_threshold': 46,
             'stop_loss': 0.04, 'take_profit': 0.05, 'position_size': 0.10},
            # 更激进
            {'top_n': 8, 'buy_threshold': 48, 'sell_threshold': 48,
             'stop_loss': 0.06, 'take_profit': 0.08, 'position_size': 0.10},
        ]

        best_score = 0
        best_params = None

        for params in param_candidates:
            full_params = {
                'ml_weight': 0.5, 'trend_weight': 0.25,
                'mean_rev_weight': 0.15, 'momentum_weight': 0.1,
                **params
            }
            result = research.run_round(1, full_params, stock_data, start_date, end_date)

            if result.score > best_score:
                best_score = result.score
                best_params = full_params

        if best_params and best_score > 0:
            # 更新参数
            from src.simulation.evolving_portfolio import DEFAULT_PARAMS
            DEFAULT_PARAMS.update(best_params)

            # 保存状态
            state = {
                'iteration': 9999,
                'best_score': best_score,
                'best_params': best_params,
                'timestamp': datetime.now().isoformat(),
                'reason': reason
            }
            with open(f'{CACHE_DIR}/auto_research_state.json', 'w') as f:
                json.dump(state, f, indent=2)

            log(f'  参数已优化，评分: {best_score:.1f}')
            return True

    except Exception as e:
        log(f'  参数优化失败: {e}')

    return False


def step7_record_log(actions: List, backtest: Dict, optimized: bool):
    """步骤7: 记录反思日志"""
    from src.simulation.evolving_portfolio import DEFAULT_PARAMS

    log('='*50)
    log('步骤7: 记录反思日志')

    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')

    log_entry = {
        'date': date_str,
        'timestamp': today.isoformat(),
        'actions': actions,
        'backtest': {
            'total_value': backtest.get('total_value', 0),
            'cash': backtest.get('cash', 0)
        },
        'optimized': optimized,
        'params': {k: v for k, v in DEFAULT_PARAMS.items()}
    }

    # 读取历史
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)

    logs.append(log_entry)

    # 只保留最近30天
    logs = logs[-30:]

    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    # 保存markdown格式总结
    try:
        os.makedirs(RETRO_DIR, exist_ok=True)
        md_file = f'{RETRO_DIR}/{date_str}.md'

        # 加载持仓
        positions_file = f'{CACHE_DIR}/positions.json'
        positions = []
        if os.path.exists(positions_file):
            with open(positions_file, 'r') as f:
                positions = json.load(f)

        buy_count = sum(1 for a in actions if a.get('action') in ['买入'])
        sell_count = sum(1 for a in actions if a.get('action') in ['止损', '止盈', '卖出信号'])

        md_content = f"""# 每日反思 - {date_str}

## 收盘后检查 ({today.strftime('%H:%M')})

### 1. 数据更新
- 成功更新 50 只股票数据
- 数据来源：baostock

### 2. 信号扫描
- 买入信号：待查询
- 卖出信号：待查询

### 3. 持仓操作
"""

        for a in actions:
            md_content += f"- **{a.get('action', '操作')}**：{a.get('name', '')} @ {a.get('price', 0):.2f} ({a.get('pnl', 0):+.1%})\n"

        md_content += f"""
- 当前持仓：{len(positions)}/{DEFAULT_PARAMS['top_n']} 只

### 4. 回测评估
- 近期收益：{((backtest.get('total_value', 2000000) - 2000000) / 2000000):+.2%}
- 状态：{'**需优化**' if optimized else '**正常**'}

### 5. 参数状态
- 迭代轮次：{log_entry.get('iteration', 'N/A')}
- 评分：{log_entry.get('best_score', 'N/A')}

---

## 策略参数
```python
{json.dumps(dict(DEFAULT_PARAMS), indent=2, ensure_ascii=False)}
```

## 下一步
{'- 执行参数优化' if optimized else '- 继续监控市场信号'}
"""

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        log(f'  已保存markdown: {md_file}')
    except Exception as e:
        log(f'  保存markdown失败: {e}')

    log(f'  已记录，今日操作: {len(actions)} 次')


def run_daily_review():
    """运行每日反思"""
    log('='*60)
    log(f'每日反思开始 - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    log('='*60)

    # 步骤1: 更新数据
    if not step1_update_data():
        log('数据更新失败，反思中断')
        return False

    # 步骤2: 扫描信号
    signals = step2_scan_signals()

    # 步骤3: 检查持仓
    actions, cash = step3_check_positions(signals)

    # 步骤4: 运行回测
    backtest = step4_run_backtest()

    # 步骤5: 评估参数
    need_adjust, reason = step5_evaluate_params(backtest)

    # 步骤6: 优化参数
    optimized = step6_optimize_params(need_adjust, reason)

    # 步骤7: 记录日志
    step7_record_log(actions, backtest, optimized)

    log('='*60)
    log('每日反思完成')
    log('='*60)

    return True


if __name__ == '__main__':
    run_daily_review()
