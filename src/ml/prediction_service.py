#!/usr/bin/env python3
"""
预测服务模块
============

使用训练好的模型对股票进行预测
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys

sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')


class PredictionService:
    """预测服务"""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.xgb_model = None
        self.lstm_model = None
        self.config = None
        self.feature_names = None
        self.feature_engineer = None
        self.is_loaded = False

    def load_models(self) -> bool:
        """加载模型"""
        import xgboost as xgb

        xgb_path = os.path.join(self.model_dir, 'xgb_model.json')
        lstm_path = os.path.join(self.model_dir, 'lstm_model.keras')
        config_path = os.path.join(self.model_dir, 'model_config.json')

        if not os.path.exists(xgb_path):
            print("模型文件不存在，请先运行训练")
            return False

        # 加载XGBoost
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_path)
        print(f"XGBoost模型已加载: {xgb_path}")

        # LSTM暂不自动加载（TensorFlow初始化问题）
        self.lstm_model = None

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            self.feature_names = self.config.get('feature_names', [])

        # 初始化特征工程
        from .model_training import FeatureEngineer
        self.feature_engineer = FeatureEngineer(
            lookback=self.config.get('lookback_window', 20)
        )

        self.is_loaded = True
        print("模型加载完成 (XGBoost)")
        return True

    def calculate_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """为预测计算特征"""
        if self.feature_engineer is None:
            from .model_training import FeatureEngineer
            self.feature_engineer = FeatureEngineer()

        return self.feature_engineer.calculate_features(df)

    def create_feature_vector(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """从最新数据创建特征向量"""
        df = self.calculate_features_for_prediction(df)

        if not self.feature_names:
            print("特征名未加载")
            return None

        # 取最后一行
        last_row = df[self.feature_names].iloc[-1:]

        if last_row.isna().any().any():
            # 尝试填充NaN
            last_row = last_row.fillna(method='ffill').fillna(method='bfill')

        return last_row.values

    def create_sequence(self, df: pd.DataFrame, seq_length: int = 60) -> Optional[np.ndarray]:
        """为LSTM创建序列"""
        df = df.copy()
        df['return'] = df['close'].pct_change()

        if len(df) < seq_length:
            return None

        seq = df[['return', 'volume', 'close']].iloc[-seq_length:].values

        # 标准化
        seq_norm = seq.copy()
        for j in range(seq.shape[1]):
            col = seq[:, j]
            mean = np.nanmean(col)
            std = np.nanstd(col) + 1e-8
            seq_norm[:, j] = (col - mean) / std

        return seq_norm.reshape(1, seq_length, 3)

    def predict_single_xgb(self, df: pd.DataFrame) -> float:
        """用XGBoost预测单只股票"""
        if self.xgb_model is None:
            return 0.0

        features = self.create_feature_vector(df)
        if features is None:
            return 0.0

        try:
            pred = self.xgb_model.predict(features)[0]
            return float(pred)
        except Exception as e:
            print(f"XGBoost预测失败: {e}")
            return 0.0

    def predict_single_lstm(self, df: pd.DataFrame) -> float:
        """用LSTM预测单只股票"""
        if self.lstm_model is None:
            return 0.0

        seq = self.create_sequence(df)
        if seq is None:
            return 0.0

        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().set_level('ERROR')
            pred = self.lstm_model.predict(seq, verbose=0)[0][0]
            return float(pred)
        except Exception as e:
            return 0.0

    def predict_single(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict[str, float]:
        """
        预测单只股票的未来收益率

        Returns:
            {
                'xgb': XGBoost预测,
                'lstm': LSTM预测,
                'ensemble': 集成预测
            }
        """
        xgb_pred = self.predict_single_xgb(df) if self.xgb_model else 0.0
        lstm_pred = self.predict_single_lstm(df) if self.lstm_model else 0.0

        # 集成：XGBoost权重更高（更稳定）
        if self.lstm_model:
            ensemble_pred = 0.6 * xgb_pred + 0.4 * lstm_pred
        else:
            ensemble_pred = xgb_pred

        return {
            'xgb': xgb_pred,
            'lstm': lstm_pred,
            'ensemble': ensemble_pred
        }

    def predict(self, stock_data: Dict[str, pd.DataFrame],
                use_ensemble: bool = True) -> Dict[str, Dict[str, float]]:
        """
        对所有股票进行预测

        Args:
            stock_data: {code: DataFrame}
            use_ensemble: 是否使用集成预测

        Returns:
            {code: {'xgb': pred, 'lstm': pred, 'ensemble': pred}}
        """
        predictions = {}

        for code, df in stock_data.items():
            if df is None or len(df) < 60:
                continue

            try:
                preds = self.predict_single(df, use_ensemble=use_ensemble)
                predictions[code] = preds
            except Exception as e:
                print(f"预测 {code} 失败: {e}")
                continue

        return predictions

    def get_top_predictions(self, predictions: Dict[str, Dict[str, float]],
                           n: int = 10, by: str = 'ensemble') -> List[tuple]:
        """
        获取预测值最高/最低的股票

        Args:
            predictions: 预测结果
            n: 返回数量
            by: 'ensemble', 'xgb', 或 'lstm'

        Returns:
            [(code, prediction), ...] 按预测值降序
        """
        items = [(code, pred[by]) for code, pred in predictions.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]


def main():
    """测试预测服务"""
    from src.data.data_pipeline import DataPipeline

    print("加载模型...")
    pred_service = PredictionService()

    if not pred_service.load_models():
        print("模型加载失败，请先训练")
        return

    print("\n加载数据...")
    pipeline = DataPipeline()
    stock_data = {}
    codes = pipeline.get_stock_pool()[:10]

    for code in codes:
        df = pipeline.load(code)
        if df is not None:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    print("\n开始预测...")
    predictions = pred_service.predict(stock_data)

    print("\n预测结果 (按集成预测排序):")
    top_preds = pred_service.get_top_predictions(predictions, n=10)
    for code, pred in top_preds:
        print(f"  {code}: XGB={predictions[code]['xgb']:.4f}, "
              f"LSTM={predictions[code]['lstm']:.4f}, "
              f"Ensemble={predictions[code]['ensemble']:.4f}")


if __name__ == '__main__':
    main()
