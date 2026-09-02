#!/usr/bin/env python3
"""
系统状态报告
============

生成quant_evo系统状态报告

使用方法:
    python scripts/system_status.py
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dolphindb_reader import DolphinDBReader


def check_dolphindb():
    """检查DolphinDB数据"""
    print("\n" + "="*60)
    print("DolphinDB 数据层状态")
    print("="*60)

    reader = DolphinDBReader()

    # 1. 日线数据统计
    print("\n【日线行情表 daily_k】")
    try:
        cnt = reader.execute('select count(*) as cnt from loadTable("dfs://quant_evo_market", "daily_k")')
        print(f"  总记录: {cnt.iloc[0][0]:,} 条")

        min_d = reader.execute('select min(trade_date) as d from loadTable("dfs://quant_evo_market", "daily_k")')
        max_d = reader.execute('select max(trade_date) as d from loadTable("dfs://quant_evo_market", "daily_k")')
        print(f"  日期范围: {min_d.iloc[0][0]} ~ {max_d.iloc[0][0]}")

        stocks = reader.execute('select count(security_id) as cnt from (select distinct security_id from loadTable("dfs://quant_evo_market", "daily_k"))')
        print(f"  股票数量: {stocks.iloc[0][0]} 只")
    except Exception as e:
        print(f"  查询失败: {e}")

    # 2. 因子数据统计
    print("\n【因子表 factor_daily】")
    try:
        cnt = reader.execute('select count(*) as cnt from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"  总记录: {cnt.iloc[0][0]:,} 条")

        min_d = reader.execute('select min(trade_date) as d from loadTable("dfs://quant_evo_factors", "factor_daily")')
        max_d = reader.execute('select max(trade_date) as d from loadTable("dfs://quant_evo_factors", "factor_daily")')
        print(f"  日期范围: {min_d.iloc[0][0]} ~ {max_d.iloc[0][0]}")

        # 信号分布
        signals = reader.execute('''
            select signal, count(*) as cnt
            from loadTable("dfs://quant_evo_factors", "factor_daily")
            group by signal
        ''')
        if signals is not None and len(signals) > 0:
            print(f"  信号分布:")
            for _, row in signals.iterrows():
                print(f"    {row['signal']}: {row['cnt']:,} ({row['cnt']/cnt.iloc[0][0]*100:.1f}%)")

        # 评分统计
        scores = reader.execute('''
            select min(composite_score) as min_s,
                   max(composite_score) as max_s,
                   avg(composite_score) as avg_s
            from loadTable("dfs://quant_evo_factors", "factor_daily")
        ''')
        if scores is not None and len(scores) > 0:
            print(f"  评分: 最低{min_scores.iloc[0][0]:.1f}, 最高{max_scores.iloc[0][0]:.1f}, 平均{avg_scores.iloc[0][0]:.1f}")
    except Exception as e:
        print(f"  查询失败: {e}")

    reader.disconnect()


def check_modules():
    """检查Python模块"""
    print("\n" + "="*60)
    print("Python 模块状态")
    print("="*60)

    modules = [
        ('dolphindb', 'DolphinDB SDK'),
        ('xgboost', 'XGBoost'),
        ('tensorflow', 'TensorFlow'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]

    for mod, name in modules:
        try:
            __import__(mod)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (未安装)")


def check_scripts():
    """检查脚本文件"""
    print("\n" + "="*60)
    print("脚本文件状态")
    print("="*60)

    scripts_dir = '/Users/keira/project/claude/quant_evo/scripts'
    scripts = [
        'batch_backtest.py',
        'batch_calc_factors.py',
        'batch_calc_factors_to_dolphindb.py',
        'batch_calc_all_factors.py',
        'dolphindb_backtest.py',
        'train_and_evolve.py',
        'optimize_strategy.py',
        'write_to_dolphindb.py',
        'system_status.py',
    ]

    for s in scripts:
        path = os.path.join(scripts_dir, s)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {s} ({size:,} bytes)")
        else:
            print(f"  ✗ {s} (不存在)")


def check_data_files():
    """检查数据文件"""
    print("\n" + "="*60)
    print("数据文件状态")
    print("="*60)

    data_dir = '/Users/keira/project/claude/quant_evo/data'

    # Parquet文件
    cache_dir = os.path.join(data_dir, 'cache')
    if os.path.exists(cache_dir):
        parquet_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
        print(f"  Parquet文件: {len(parquet_files)} 个")

    # 因子结果
    factors_dir = os.path.join(data_dir, 'factors')
    if os.path.exists(factors_dir):
        factor_files = [f for f in os.listdir(factors_dir) if f.endswith('.json')]
        print(f"  因子结果文件: {len(factor_files)} 个")

    # 回测结果
    results_dir = os.path.join(data_dir, 'backtest_results')
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        print(f"  回测结果文件: {len(result_files)} 个")


def check_services():
    """检查服务状态"""
    print("\n" + "="*60)
    print("服务状态")
    print("="*60)

    import socket

    # DolphinDB
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', 8848))
    if result == 0:
        print("  ✓ DolphinDB (localhost:8848)")
    else:
        print("  ✗ DolphinDB (未运行)")
    sock.close()

    # Web UI (Flask)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', 5124))
    if result == 0:
        print("  ✓ Web UI (localhost:5124)")
    else:
        print("  ✗ Web UI (未运行)")
    sock.close()

    # News WebSocket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', 5125))
    if result == 0:
        print("  ✓ News WebSocket (localhost:5125)")
    else:
        print("  ✗ News WebSocket (未运行)")
    sock.close()


def main():
    print("\n" + "#"*60)
    print("# quant_evo 系统状态报告")
    print("# 生成时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("#"*60)

    check_dolphindb()
    check_modules()
    check_scripts()
    check_data_files()
    check_services()

    print("\n" + "="*60)
    print("报告生成完成")
    print("="*60)


if __name__ == '__main__':
    main()