#!/usr/bin/env python3
"""
批量计算因子
============

批量计算股票池所有股票的因子

使用方法:
    python scripts/batch_calc_factors.py --all
    python scripts/batch_calc_factors.py --codes sh.600519 sz.000858
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import DataPipeline
from src.factors.technical import FactorCalculator
from src.data.data_cleaner import DataCleaner
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTOR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'factors')


def calculate_factors_for_stock(code: str, df: pd.DataFrame) -> dict:
    """计算单只股票的因子"""
    calc = FactorCalculator()

    # 清洗数据
    cleaner = DataCleaner()
    result = cleaner.clean(df)
    df_clean = result.df

    # 计算因子
    factors = calc.calculate_all_factors(df_clean)
    signal = calc.generate_signal(factors)
    score = calc.get_composite_score(factors)

    return {
        'code': code,
        'signal': signal,
        'score': score,
        'factors': factors,
        'dataframe': df_clean
    }


def batch_calculate(codes: list, save: bool = True) -> list:
    """批量计算因子"""
    pipeline = DataPipeline()

    os.makedirs(FACTOR_OUTPUT_DIR, exist_ok=True)

    results = []
    total = len(codes)

    for i, code in enumerate(codes):
        print(f"[{i+1}/{total}] {code}...", end=' ')

        try:
            # 加载数据
            df = pipeline.load(code)
            if df is None or len(df) < 30:
                print(f"数据不足")
                continue

            # 计算因子
            result = calculate_factors_for_stock(code, df)
            results.append(result)

            # 保存
            if save:
                output_file = os.path.join(FACTOR_OUTPUT_DIR, f"{code.replace('.', '_')}_factors.json")
                save_data = {
                    'code': code,
                    'signal': result['signal'],
                    'score': result['score'],
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(output_file, 'w') as f:
                    json.dump(save_data, f, indent=2)

            print(f"signal={result['signal']}, score={result['score']:.1f}")

        except Exception as e:
            print(f"失败: {e}")

    return results


def show_summary(results: list):
    """显示汇总"""
    if not results:
        print("无结果")
        return

    signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    for r in results:
        signals[r['signal']] = signals.get(r['signal'], 0) + 1

    scores = [r['score'] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"\n{'='*60}")
    print("因子计算汇总")
    print(f"{'='*60}")
    print(f"计算股票数: {len(results)}")
    print(f"信号分布: 买入 {signals['BUY']}, 卖出 {signals['SELL']}, 持有 {signals['HOLD']}")
    print(f"平均评分: {avg_score:.1f}")

    # 显示评分最高的 10 只
    top10 = sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    print(f"\n评分最高 10 只:")
    for r in top10:
        print(f"  {r['code']}: {r['score']:.1f} ({r['signal']})")


def main():
    parser = argparse.ArgumentParser(description='批量计算因子')
    parser.add_argument('--all', action='store_true', help='计算全部股票')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--save', action='store_true', default=True, help='保存结果')

    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.codes:
        codes = args.codes
    elif args.all:
        codes = pipeline.get_stock_pool()
    else:
        codes = ['sh.600519', 'sh.600036', 'sz.000858']

    print(f"\n{'='*60}")
    print("批量因子计算")
    print(f"{'='*60}")
    print(f"股票数量: {len(codes)}")

    results = batch_calculate(codes, save=args.save)

    show_summary(results)

    # 保存汇总
    if args.save and results:
        summary_file = os.path.join(FACTOR_OUTPUT_DIR, 'factor_summary.json')
        summary = [{
            'code': r['code'],
            'signal': r['signal'],
            'score': r['score']
        } for r in results]
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n汇总已保存: {summary_file}")


if __name__ == '__main__':
    main()
