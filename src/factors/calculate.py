#!/usr/bin/env python3
"""
因子计算脚本
============

批量计算技术因子

使用方法:
    # 计算所有股票
    python -m src.factors.calculate --all

    # 计算指定股票
    python -m src.factors.calculate --codes 600519 000858
"""

import argparse
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_factors_for_stock(code: str, df) -> dict:
    """计算单只股票的因子"""
    from src.factors.technical import FactorCalculator

    calc = FactorCalculator()

    factors = calc.calculate_all_factors(df)
    signal = calc.generate_signal(factors)
    score = calc.get_composite_score(factors)

    return {
        'code': code,
        'signal': signal,
        'score': score,
        'factors': factors
    }


def calculate_all(stock_codes: list = None):
    """计算所有股票的因子"""
    if stock_codes is None:
        stock_codes = [
            'sh.600519', 'sh.600036', 'sh.600000', 'sh.600028',
            'sz.000858', 'sz.000001', 'sz.000002', 'sz.000333'
        ]

    from src.data.data_pipeline import DataPipeline
    from src.data.data_cleaner import DataCleaner

    pipeline = DataPipeline()
    cleaner = DataCleaner()
    results = []

    for code in stock_codes:
        try:
            # 加载数据
            df = pipeline.load_from_csv(code.replace('.', '_'))
            if df is None or len(df) < 30:
                logger.warning(f"{code} 数据不足，跳过")
                continue

            # 清洗数据
            result = cleaner.clean(df)
            df_clean = result.df

            # 计算因子
            factors = calculate_factors_for_stock(code, df_clean)
            results.append(factors)

            logger.info(f"{code}: signal={factors['signal']}, score={factors['score']:.1f}")

        except Exception as e:
            logger.error(f"{code} 计算失败: {e}")

    return results


def save_factors_to_dolphindb(results: list):
    """保存因子到 DolphinDB"""
    try:
        from src.data.dolphindb_writer import DolphinDBWriter
        writer = DolphinDBWriter()

        for r in results:
            code = r['code'].replace('.', '_')
            df = r.get('dataframe')
            if df is not None:
                writer.write_factor(code, df)
                logger.info(f"保存 {code} 因子到 DolphinDB")

    except Exception as e:
        logger.error(f"保存因子失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='因子计算')
    parser.add_argument('--all', action='store_true', help='计算所有股票')
    parser.add_argument('--codes', nargs='+', help='指定股票代码')
    parser.add_argument('--save', action='store_true', help='保存到 DolphinDB')

    args = parser.parse_args()

    if args.codes:
        codes = args.codes
    elif args.all:
        codes = None
    else:
        codes = ['sh.600519']

    logger.info(f"开始计算因子: {codes or '全部'}")

    results = calculate_all(codes)

    logger.info(f"计算完成: {len(results)} 只股票")

    if args.save and results:
        save_factors_to_dolphindb(results)

    # 打印结果
    print(f"\n{'='*60}")
    print("因子计算结果")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['code']}: signal={r['signal']}, score={r['score']:.1f}")


if __name__ == '__main__':
    main()
