#!/usr/bin/env python3
"""
ML模型训练模块
============

提供特征工程、模型训练、模型持久化功能
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')


@dataclass
class ModelConfig:
    """模型配置"""
    prediction_horizon: int = 5  # 预测未来N日
    lookback_window: int = 20   # 回看窗口
    train_ratio: float = 0.8    # 训练集比例
    xgb_params: dict = None
    lstm_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        if self.lstm_params is None:
            self.lstm_params = {
                'sequence_length': 60,
                'lstm_units': 64,
                'dropout': 0.2,
                'epochs': 20,
                'batch_size': 32,
            }


class FeatureEngineer:
    """特征工程"""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术因子作为特征

        Returns:
            DataFrame with features
        """
        df = df.copy()

        # 收益率
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # 动量因子
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_10d'] = df['close'].pct_change(10)
        df['momentum_20d'] = df['close'].pct_change(20)
        df['momentum_60d'] = df['close'].pct_change(60)

        # 波动率因子
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()

        # 均线
        for window in [5, 10, 20, 60]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_ma{window}_ratio'] = df['close'] / df[f'ma_{window}']

        # 均线多头排列
        df['ma_bull'] = (
            (df['ma_5'] > df['ma_10']) &
            (df['ma_10'] > df['ma_20']) &
            (df['ma_20'] > df['ma_60'])
        ).astype(int)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 布林带
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']

        # 量价特征
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['price_volume_corr'] = df['close'].rolling(10).corr(df['volume'])

        # 换手率变化
        df['turn_change'] = df['turn'].pct_change()

        return df

    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征矩阵"""
        df = self.calculate_features(df)

        # 选择特征列
        feature_cols = [
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio', 'price_ma60_ratio',
            'ma_bull',
            'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'atr_ratio',
            'volume_ratio', 'price_volume_corr', 'turn_change',
        ]

        # 只保留存在的列
        feature_cols = [c for c in feature_cols if c in df.columns]

        return df[feature_cols]

    def create_sequences(self, df: pd.DataFrame, seq_length: int = 60) -> np.ndarray:
        """为LSTM创建序列数据"""
        # 使用OHLCV数据创建序列
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        data = df[ohlcv_cols].values

        sequences = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i])

        return np.array(sequences)


class MLModelTrainer:
    """ML模型训练器"""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.feature_engineer = FeatureEngineer(lookback=self.config.lookback_window)
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None  # 用于LSTM的数据标准化器
        self.is_trained = False

        os.makedirs(MODEL_DIR, exist_ok=True)

    def prepare_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple:
        """
        准备训练数据

        Args:
            stock_data: {code: DataFrame}

        Returns:
            (X_train, y_train, X_valid, y_valid, feature_names)
        """
        all_features = []
        all_labels = []
        all_sequences = []  # LSTM序列数据
        sequence_labels = []

        for code, df in stock_data.items():
            if df is None or len(df) < 100:
                continue

            # 跳过北交所股票（数据量少）
            if code.startswith('bj.'):
                continue

            try:
                # 计算特征
                df_features = self.feature_engineer.calculate_features(df)

                # 创建特征矩阵
                feature_matrix = self.feature_engineer.create_feature_matrix(df)
                feature_names = feature_matrix.columns.tolist()

                # 创建目标变量：未来5日收益率
                df_features['future_return'] = df_features['close'].shift(-self.config.prediction_horizon) / df_features['close'] - 1

                # 移除NaN
                valid_idx = df_features.dropna(subset=['future_return']).index
                feature_matrix = feature_matrix.loc[valid_idx]
                labels = df_features.loc[valid_idx, 'future_return']

                all_features.append(feature_matrix.values)
                all_labels.append(labels.values)

                # 创建LSTM序列
                if len(df) >= self.config.lstm_params['sequence_length']:
                    df_ohlcv = df.copy()
                    seqs = self.feature_engineer.create_sequences(
                        df_ohlcv,
                        self.config.lstm_params['sequence_length']
                    )
                    # 对应的标签
                    seq_labels = []
                    for i in range(self.config.lstm_params['sequence_length'],
                                   len(df) - self.config.prediction_horizon + 1):
                        future_ret = (df['close'].iloc[i + self.config.prediction_horizon - 1] /
                                      df['close'].iloc[i - 1] - 1)
                        seq_labels.append(future_ret)
                    if len(seq_labels) > 0:
                        all_sequences.append(seqs[:len(seq_labels)])
                        sequence_labels.extend(seq_labels)

            except Exception as e:
                print(f"处理 {code} 时出错: {e}")
                continue

        # 合并所有数据
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        # 移除异常值
        valid_mask = (np.abs(y) < 0.5) & (~np.isnan(y)) & (~np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]

        # 时间序列分割
        n_train = int(len(X) * self.config.train_ratio)
        X_train, X_valid = X[:n_train], X[n_train:]
        y_train, y_valid = y[:n_train], y[n_train:]

        print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")
        print(f"特征数: {X_train.shape[1]}")

        return X_train, y_train, X_valid, y_valid, feature_names

    def prepare_lstm_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple:
        """准备LSTM训练数据"""
        all_sequences = []
        all_labels = []

        for code, df in stock_data.items():
            if df is None or len(df) < 100:
                continue
            if code.startswith('bj.'):
                continue

            try:
                # 计算收益率序列
                df = df.copy()
                df['return'] = df['close'].pct_change()

                # 创建序列
                seq_length = self.config.lstm_params['sequence_length']
                if len(df) < seq_length + self.config.prediction_horizon:
                    continue

                for i in range(seq_length, len(df) - self.config.prediction_horizon + 1):
                    seq = df[['return', 'volume', 'close']].iloc[i-seq_length:i].values
                    # 标准化
                    seq_norm = seq.copy()
                    for j in range(seq.shape[1]):
                        col = seq[:, j]
                        mean = np.nanmean(col)
                        std = np.nanstd(col) + 1e-8
                        seq_norm[:, j] = (col - mean) / std
                    all_sequences.append(seq_norm)

                    # 标签
                    future_ret = (df['close'].iloc[i + self.config.prediction_horizon - 1] /
                                  df['close'].iloc[i - 1] - 1)
                    all_labels.append(future_ret)

            except Exception as e:
                continue

        X = np.array(all_sequences)
        y = np.array(all_labels)

        # 移除异常值
        valid_mask = (np.abs(y) < 0.5) & (~np.isnan(y)) & (~np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]

        # 时间序列分割
        n_train = int(len(X) * self.config.train_ratio)
        X_train, X_valid = X[:n_train], X[n_train:]
        y_train, y_valid = y[:n_train], y[n_train:]

        print(f"LSTM 训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")

        return X_train, y_train, X_valid, y_valid

    def train_xgb_model(self, X_train, y_train, X_valid, y_valid):
        """训练XGBoost模型"""
        try:
            import xgboost as xgb

            params = self.config.xgb_params.copy()
            n_estimators = params.pop('n_estimators')

            self.xgb_model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                **params,
                random_state=42,
                n_jobs=-1,
            )

            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )

            # 评估
            train_pred = self.xgb_model.predict(X_train)
            valid_pred = self.xgb_model.predict(X_valid)

            train_ic = np.corrcoef(train_pred, y_train)[0, 1]
            valid_ic = np.corrcoef(valid_pred, y_valid)[0, 1]

            print(f"XGBoost - 训练IC: {train_ic:.4f}, 验证IC: {valid_ic:.4f}")

            return valid_ic

        except Exception as e:
            print(f"XGBoost训练失败: {e}")
            return None

    def train_lstm_model(self, X_train, y_train, X_valid, y_valid):
        """训练LSTM模型"""
        try:
            # 使用tensorflow/keras
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            tf.get_logger().setLevel('ERROR')

            seq_length = self.config.lstm_params['sequence_length']
            n_features = X_train.shape[2]

            # 构建模型
            self.lstm_model = keras.Sequential([
                layers.Input(shape=(seq_length, n_features)),
                layers.LSTM(self.config.lstm_params['lstm_units'], return_sequences=True),
                layers.Dropout(self.config.lstm_params['dropout']),
                layers.LSTM(self.config.lstm_params['lstm_units'] // 2),
                layers.Dropout(self.config.lstm_params['dropout']),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])

            self.lstm_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )

            # 训练
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = self.lstm_model.fit(
                X_train, y_train,
                validation_data=(X_valid, y_valid),
                epochs=self.config.lstm_params['epochs'],
                batch_size=self.config.lstm_params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )

            # 评估
            valid_pred = self.lstm_model.predict(X_valid, verbose=0).flatten()
            valid_ic = np.corrcoef(valid_pred, y_valid)[0, 1]

            print(f"LSTM - 验证IC: {valid_ic:.4f}")

            return valid_ic

        except Exception as e:
            print(f"LSTM训练失败: {e}")
            return None

    def train_ensemble(self, stock_data: Dict[str, pd.DataFrame]):
        """训练集成模型"""
        print("=" * 50)
        print("开始训练ML模型")
        print("=" * 50)

        # 准备数据
        X_train, y_train, X_valid, y_valid, feature_names = self.prepare_data(stock_data)

        # 训练XGBoost
        print("\n--- 训练 XGBoost ---")
        xgb_ic = self.train_xgb_model(X_train, y_train, X_valid, y_valid)

        # 尝试训练LSTM（可选）
        lstm_ic = None
        try:
            print("\n--- 准备 LSTM 数据 ---")
            X_lstm_train, y_lstm_train, X_lstm_valid, y_lstm_valid = self.prepare_lstm_data(stock_data)
            if len(X_lstm_train) > 100:
                print("\n--- 训练 LSTM ---")
                lstm_ic = self.train_lstm_model(X_lstm_train, y_lstm_train, X_lstm_valid, y_lstm_valid)
        except Exception as e:
            print(f"LSTM训练跳过: {e}")

        self.is_trained = True

        # 保存模型
        self.save_models(feature_names)

        return {
            'xgb_ic': xgb_ic,
            'lstm_ic': lstm_ic
        }

    def save_models(self, feature_names: List[str]):
        """保存模型到文件"""
        import xgboost as xgb

        # 保存XGBoost
        if self.xgb_model is not None:
            xgb_path = os.path.join(MODEL_DIR, 'xgb_model.json')
            self.xgb_model.save_model(xgb_path)
            print(f"XGBoost模型已保存: {xgb_path}")

        # 保存LSTM
        if self.lstm_model is not None:
            lstm_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
            self.lstm_model.save(lstm_path)
            print(f"LSTM模型已保存: {lstm_path}")

        # 保存配置
        config_path = os.path.join(MODEL_DIR, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'prediction_horizon': self.config.prediction_horizon,
                'lookback_window': self.config.lookback_window,
                'feature_names': feature_names,
                'xgb_params': self.config.xgb_params,
                'lstm_params': self.config.lstm_params,
            }, f, indent=2)
        print(f"模型配置已保存: {config_path}")

    def load_models(self) -> bool:
        """加载模型"""
        import xgboost as xgb

        xgb_path = os.path.join(MODEL_DIR, 'xgb_model.json')
        lstm_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
        config_path = os.path.join(MODEL_DIR, 'model_config.json')

        if not os.path.exists(xgb_path):
            print("模型文件不存在，请先训练模型")
            return False

        # 加载XGBoost
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_path)

        # 加载LSTM
        if os.path.exists(lstm_path):
            import tensorflow as tf
            self.lstm_model = tf.keras.models.load_model(lstm_path)

        # 加载配置
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            self.config.prediction_horizon = config_data['prediction_horizon']
            self.config.lookback_window = config_data['lookback_window']

        self.is_trained = True
        print("模型加载成功")
        return True


def main():
    """测试训练流程"""
    from src.data.data_pipeline import DataPipeline

    print("加载数据...")
    pipeline = DataPipeline()
    stock_data = {}
    codes = pipeline.get_stock_pool()[:50]  # 用50只股票测试

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 100:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    # 训练
    trainer = MLModelTrainer()
    results = trainer.train_ensemble(stock_data)

    print("\n训练完成!")
    print(f"XGBoost IC: {results['xgb_ic']:.4f}")
    print(f"LSTM IC: {results['lstm_ic']:.4f}")


if __name__ == '__main__':
    main()
