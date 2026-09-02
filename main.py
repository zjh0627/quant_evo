#!/usr/bin/env python3
"""
quant_evo - 自主进化量化系统

用法:
    python main.py init           # 初始化数据管道
    python main.py backtest      # 运行回测
    python main.py analyze <code> # 分析单只股票
    python main.py research      # 自主研究优化参数
    python main.py web           # 启动 Web UI (端口 5124)
"""

import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_pipeline import DataPipeline
from src.backtest.engine import BacktestEngine
from src.factors.technical import FactorCalculator, calculate_stock_factors


def cmd_init():
    """初始化数据管道"""
    from src.data.data_pipeline import init_data_pipeline
    init_data_pipeline()


def cmd_backtest():
    """运行回测"""
    pipeline = DataPipeline()

    # 加载已有数据
    stock_data = {}
    codes = pipeline.get_stock_pool()

    print("加载数据...")
    for code in codes:
        df = pipeline.load(code)  # 使用 load 方法，自动选择 Parquet 或 CSV
        if df is not None:
            stock_data[code] = df
    print(f"已加载 {len(stock_data)} 只股票\n")

    if not stock_data:
        print("没有数据，请先运行 python main.py init")
        return

    # 运行回测
    engine = BacktestEngine(initial_capital=1000000)
    result = engine.run(
        stock_data,
        start_date='2025-01-01',
        end_date='2025-06-30',
        rebalance_days=5,
        min_score=55
    )

    engine.print_result(result)


def cmd_analyze(code: str):
    """分析单只股票"""
    pipeline = DataPipeline()

    df = pipeline.load_from_csv(code)
    if df is None:
        print(f"未找到 {code} 的数据")
        return

    calc = FactorCalculator()
    factors = calc.calculate_all_factors(df)

    if factors.empty:
        print(f"无法计算 {code} 的因子")
        return

    signal = calc.generate_signal(factors)
    score = calc.get_composite_score(factors)

    latest = factors.iloc[-1]

    print(f"\n{'='*50}")
    print(f"股票分析: {code}")
    print(f"{'='*50}")
    print(f"最新价格:    {latest.get('close'):.2f}")
    print(f"综合评分:    {score:.1f}")
    print(f"交易信号:    {signal}")
    print(f"\n因子详情:")
    print(f"  5日动量:   {latest.get('momentum_5d', 0)*100:.2f}%")
    print(f"  20日动量:  {latest.get('momentum_20d', 0)*100:.2f}%")
    print(f"  均线多头:   {'是' if latest.get('ma_bull') == 1 else '否'}")
    print(f"  RSI:       {latest.get('rsi', 50):.1f}")
    print(f"  MACD:      {'正值' if latest.get('macd_hist', 0) > 0 else '负值'}")
    print(f"  布林带:    {latest.get('bb_position', 0.5)*100:.1f}%")


def cmd_research():
    """运行自主研究"""
    from src.research.research_loop import AutoResearchLoop

    research = AutoResearchLoop()
    best_params, best_metrics = research.run(
        start_date='2025-01-01',
        end_date='2025-06-30',
        max_rounds=20
    )
    research.print_experiment_history()
    print("\n最终最佳参数:")
    print(json.dumps(best_params, indent=2))


def cmd_web():
    """启动 Web UI"""
    from web_ui import app
    print("启动 quant_evo Web UI...")
    print("访问 http://localhost:5124")
    app.run(host='0.0.0.0', port=5124, debug=False)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'init':
        cmd_init()
    elif cmd == 'backtest':
        cmd_backtest()
    elif cmd == 'analyze':
        if len(sys.argv) < 3:
            print("请指定股票代码，如: python main.py analyze sh.600519")
        else:
            cmd_analyze(sys.argv[2])
    elif cmd == 'research':
        cmd_research()
    elif cmd == 'web':
        cmd_web()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
