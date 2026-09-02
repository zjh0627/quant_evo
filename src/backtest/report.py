#!/usr/bin/env python3
"""
回测报告生成脚本
================

生成回测报告并保存

使用方法:
    python -m src.backtest.report --weekly
    python -m src.backtest.report --monthly
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_weekly_report():
    """生成周报"""
    logger.info("生成周报...")

    from src.backtest.light_backtest import LightBacktest
    from src.data.data_pipeline import DataPipeline
    import pandas as pd
    import numpy as np

    # 获取数据
    pipeline = DataPipeline()
    stocks = ['sh.600519', 'sz.000858', 'sh.600036']

    data_dict = {}
    signals_dict = {}

    for code in stocks:
        df = pipeline.load_from_csv(code.replace('.', '_'))
        if df is not None and len(df) > 0:
            # 生成信号
            df['signal'] = 'HOLD'
            df.loc[df['rsi_6'] < 30, 'signal'] = 'BUY'
            df.loc[df['rsi_6'] > 70, 'signal'] = 'SELL'
            df['score'] = 50 + (50 - df['rsi_6']) if 'rsi_6' in df.columns else 60

            data_dict[code] = df
            signals_dict[code] = df[['date', 'signal', 'score']]

    # 运行回测
    bt = LightBacktest(initial_capital=10000000)
    result = bt.run(data_dict, signals_dict, '2024-01-01', '2024-03-31')

    # 生成报告
    report = {
        'report_date': datetime.now().strftime('%Y-%m-%d'),
        'period': 'weekly',
        'total_return': f"{result.total_return:.2%}" if result else "N/A",
        'annual_return': f"{result.annual_return:.2%}" if result else "N/A",
        'sharpe_ratio': f"{result.sharpe_ratio:.2f}" if result else "N/A",
        'max_drawdown': f"{result.max_drawdown:.2%}" if result else "N/A",
        'win_rate': f"{result.win_rate:.2%}" if result else "N/A",
        'total_trades': result.total_trades if result else 0,
    }

    # 保存报告
    report_dir = f"{sys.path[0]}/../../reports"
    os.makedirs(report_dir, exist_ok=True)
    report_file = f"{report_dir}/backtest_{datetime.now().strftime('%Y%m%d')}.json"

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"报告已保存: {report_file}")

    return report


def generate_monthly_report():
    """生成月报"""
    logger.info("生成月报...")
    # 类似周报，但覆盖更长周期
    return generate_weekly_report()


def main():
    parser = argparse.ArgumentParser(description='回测报告')
    parser.add_argument('--weekly', action='store_true', help='生成周报')
    parser.add_argument('--monthly', action='store_true', help='生成月报')

    args = parser.parse_args()

    if args.weekly:
        report = generate_weekly_report()
    elif args.monthly:
        report = generate_monthly_report()
    else:
        report = generate_weekly_report()

    print(f"\n{'='*50}")
    print("回测报告")
    print(f"{'='*50}")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
