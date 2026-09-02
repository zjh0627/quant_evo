#!/usr/bin/env python3
"""
ML策略训练与进化脚本
===================

完整的ML驱动策略训练和进化流程

使用方法:
    python scripts/train_and_evolve.py --train-only    # 仅训练
    python scripts/train_and_evolve.py --evolve-only   # 仅进化（需要已有模型）
    python scripts/train_and_evolve.py --rounds 20    # 完整流程，20轮进化
"""

import argparse
import sys
import os

sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.data.data_pipeline import DataPipeline
from src.ml.model_training import MLModelTrainer
from src.ml.prediction_service import PredictionService
from src.research.ml_evolution import MLEvolutionLoop


def load_stock_data(n_stocks=None):
    """加载股票数据"""
    pipeline = DataPipeline()
    stock_data = {}
    codes = pipeline.get_stock_pool()

    if n_stocks:
        codes = codes[:n_stocks]

    print(f"加载股票数据 ({len(codes)} 只)...")

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 100:
            stock_data[code] = df

    print(f"实际加载: {len(stock_data)} 只股票")
    return stock_data


def train_models(stock_data):
    """训练模型"""
    print("\n" + "="*60)
    print("阶段1: 训练ML模型")
    print("="*60)

    trainer = MLModelTrainer()
    results = trainer.train_ensemble(stock_data)

    print("\n训练完成!")
    if results.get('xgb_ic'):
        print(f"XGBoost 验证IC: {results['xgb_ic']:.4f}")
    if results.get('lstm_ic'):
        print(f"LSTM 验证IC: {results['lstm_ic']:.4f}")

    return results


def run_evolution(stock_data, max_rounds):
    """运行进化"""
    print("\n" + "="*60)
    print(f"阶段2: 策略进化 ({max_rounds} 轮)")
    print("="*60)

    evolution = MLEvolutionLoop()
    best_params = evolution.evolve(stock_data, max_rounds=max_rounds)

    return best_params


def test_ml_strategy():
    """测试ML策略（使用已有模型）"""
    print("\n" + "="*60)
    print("阶段3: 测试ML策略")
    print("="*60)

    pipeline = DataPipeline()
    stock_data = load_stock_data(n_stocks=20)

    # 加载模型
    pred_service = PredictionService()
    if not pred_service.load_models():
        print("模型不存在，请先训练")
        return

    # 生成预测
    predictions = pred_service.predict(stock_data)

    # 显示预测结果
    print("\n预测结果 (按集成预测排序):")
    top_preds = pred_service.get_top_predictions(predictions, n=10)
    for code, pred in top_preds:
        print(f"  {code}: {pred:.2%}")

    from src.strategies.ml_strategy import MLStrategy, MLStrategyGenerator
    from src.backtest.light_backtest import LightBacktest

    # 生成信号
    strategy = MLStrategyGenerator.create_balanced_strategy(pred_service)
    signals = strategy.generate_signals(stock_data, predictions)

    # 回测
    bt = LightBacktest(initial_capital=1000000)
    result = bt.run(stock_data, signals, '2024-07-01', '2026-07-01')

    if result:
        print("\n回测结果:")
        print(result.summary())


def main():
    parser = argparse.ArgumentParser(description='ML策略训练与进化')
    parser.add_argument('--train-only', action='store_true', help='仅训练模型')
    parser.add_argument('--evolve-only', action='store_true', help='仅运行进化（需已有模型）')
    parser.add_argument('--test-only', action='store_true', help='仅测试策略（需已有模型）')
    parser.add_argument('--rounds', type=int, default=10, help='进化轮数')
    parser.add_argument('--n-stocks', type=int, default=50, help='使用的股票数量（加速训练）')

    args = parser.parse_args()

    print("\n" + "#"*60)
    print("# ML策略训练与进化系统")
    print("#" + "#"*60)

    # 加载数据
    stock_data = load_stock_data(n_stocks=args.n_stocks)

    if not stock_data:
        print("没有加载到股票数据")
        return

    if args.train_only:
        # 仅训练
        train_models(stock_data)

    elif args.evolve_only:
        # 仅进化（需要模型）
        run_evolution(stock_data, args.rounds)

    elif args.test_only:
        # 仅测试
        test_ml_strategy()

    else:
        # 完整流程: 训练 + 进化
        train_models(stock_data)
        run_evolution(stock_data, args.rounds)
        test_ml_strategy()

    print("\n完成!")


if __name__ == '__main__':
    main()
