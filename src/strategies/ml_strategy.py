#!/usr/bin/env python3
"""
ML驱动策略
==========

基于机器学习预测生成交易信号
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys

sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.ml.prediction_service import PredictionService


class MLStrategy:
    """ML驱动策略"""

    def __init__(self,
                 prediction_service: PredictionService = None,
                 top_n: int = 10,
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.02):
        self.pred_service = prediction_service
        self.top_n = top_n
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_predictions = {}

    def set_params(self, top_n: int = None, buy_threshold: float = None,
                   sell_threshold: float = None):
        if top_n is not None:
            self.top_n = top_n
        if buy_threshold is not None:
            self.buy_threshold = buy_threshold
        if sell_threshold is not None:
            self.sell_threshold = sell_threshold

    def generate_signals(self,
                        stock_data: Dict[str, pd.DataFrame],
                        predictions: Dict[str, Dict[str, float]] = None,
                        start_date: str = None,
                        end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        生成交易信号

        基于模型预测的排序，对top N配置高评分买信号，其他持有
        """
        if predictions is None and self.pred_service is None:
            self.pred_service = PredictionService()
            if not self.pred_service.load_models():
                return {}

        if predictions is None and self.pred_service:
            predictions = self.pred_service.predict(stock_data)

        self.last_predictions = predictions

        # 按预测值排序
        sorted_codes = sorted(
            predictions.keys(),
            key=lambda c: predictions[c]['ensemble'],
            reverse=True
        )

        signals_dict = {}

        for code, df in stock_data.items():
            if df is None or len(df) < 20:
                continue

            df = df.copy()

            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]

            if len(df) == 0:
                continue

            df['signal'] = 'HOLD'
            df['score'] = 50.0

            if code in predictions:
                rank = sorted_codes.index(code)
                pred = predictions[code]['ensemble']

                # 根据排名和预测值决定信号
                # top N 且预测为正 -> BUY with high score
                # not in top N 且预测很低 -> SELL
                # 其他 -> HOLD with score based on rank

                if rank < self.top_n and pred > 0:
                    df['signal'] = 'BUY'
                    # 排名越高分数越高
                    df['score'] = 60 + (self.top_n - rank) * 4
                elif rank >= len(sorted_codes) - self.top_n // 2 and pred < -0.01:
                    df['signal'] = 'SELL'
                    df['score'] = max(20, 40 - rank * 2)
                else:
                    # 基于排名的分数
                    df['score'] = max(20, min(80, 50 + (self.top_n - rank) * 2))

            signals_dict[code] = df[['date', 'signal', 'score']].copy()

        return signals_dict

    def get_holdings(self, predictions: Dict[str, Dict[str, float]] = None) -> list:
        if predictions is None:
            predictions = self.last_predictions

        if not predictions:
            return []

        sorted_codes = sorted(
            predictions.keys(),
            key=lambda c: predictions[c]['ensemble'],
            reverse=True
        )

        return sorted_codes[:self.top_n]


class MLStrategyGenerator:
    """ML策略生成器"""

    @staticmethod
    def create_conservative_strategy(pred_service: PredictionService = None) -> MLStrategy:
        return MLStrategy(
            prediction_service=pred_service,
            top_n=5,
            buy_threshold=0.05,
            sell_threshold=-0.03
        )

    @staticmethod
    def create_aggressive_strategy(pred_service: PredictionService = None) -> MLStrategy:
        return MLStrategy(
            prediction_service=pred_service,
            top_n=15,
            buy_threshold=0.01,
            sell_threshold=-0.01
        )

    @staticmethod
    def create_balanced_strategy(pred_service: PredictionService = None) -> MLStrategy:
        return MLStrategy(
            prediction_service=pred_service,
            top_n=10,
            buy_threshold=0.02,
            sell_threshold=-0.02
        )


def main():
    """测试ML策略"""
    from src.data.data_pipeline import DataPipeline

    print("加载数据...")
    pipeline = DataPipeline()
    stock_data = {}
    codes = pipeline.get_stock_pool()[:20]

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 60:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    pred_service = PredictionService()
    if not pred_service.load_models():
        print("模型未加载")
        return

    print("\n生成预测...")
    predictions = pred_service.predict(stock_data)

    print("\n生成信号...")
    strategy = MLStrategyGenerator.create_balanced_strategy(pred_service)
    signals = strategy.generate_signals(stock_data, predictions,
                                       start_date='2024-07-01',
                                       end_date='2026-07-01')

    print(f"\n生成了 {len(signals)} 只股票的信号")
    print("\n持仓建议:")
    holdings = strategy.get_holdings(predictions)
    for i, code in enumerate(holdings):
        pred = predictions[code]['ensemble']
        print(f"  {i+1}. {code}: 预测 {pred:.2%}")


if __name__ == '__main__':
    main()
