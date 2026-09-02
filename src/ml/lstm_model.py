#!/usr/bin/env python3
"""
LSTM 时序预测模型
================

用于捕捉价格序列的时序依赖模式

架构:
- 输入: 过去60天 [return, volume, close] 序列
- LSTM层: 64单元 x 2层 + Dropout
- Dense层: 输出未来5日收益预测
"""

import os
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import json

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')


class LSTMModel:
    """LSTM时序预测模型"""

    def __init__(self, seq_length: int = 60):
        self.seq_length = seq_length
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.is_loaded = False

    def build_model(self) -> 'keras.Model':
        """构建LSTM模型"""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.seq_length, 3)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='tanh')  # 输出 -1 到 1 (收益预测)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def prepare_sequence(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        从DataFrame准备LSTM输入序列

        Args:
            df: 包含 date, close, volume 的DataFrame

        Returns:
            标准化后的序列 (1, seq_length, 3)
        """
        if len(df) < self.seq_length + 5:
            return None

        df = df.copy().tail(self.seq_length + 5)

        # 计算收益率
        df['return'] = df['close'].pct_change()

        # Close标准化 (使用相对位置)
        df['close_norm'] = df['close'] / df['close'].iloc[0] - 1

        # Volume标准化
        vol = df['volume'].values
        vol_norm = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-8)

        # Return已经是标准化形式
        ret = df['return'].fillna(0).values

        # 构建序列 (只取最后seq_length个)
        seq = np.column_stack([
            ret[-self.seq_length:],
            vol_norm[-self.seq_length:],
            df['close_norm'].values[-self.seq_length:]
        ])

        return seq.reshape(1, self.seq_length, 3)

    def prepare_sequence_from_dict(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        使用全局统计量进行序列标准化

        Args:
            df: 包含 date, close, volume 的DataFrame

        Returns:
            标准化后的序列 (1, seq_length, 3)
        """
        if len(df) < self.seq_length + 5:
            return None

        df = df.copy().tail(self.seq_length + 5)

        # 计算收益率
        returns = df['close'].pct_change().fillna(0).values

        # Volume
        volumes = df['volume'].values

        # Close normalized
        close_norm = (df['close'].values / df['close'].iloc[0]) - 1

        if self.feature_mean is not None and self.feature_std is not None:
            # 使用全局统计量
            seq = np.column_stack([
                returns[-self.seq_length:],
                (volumes[-self.seq_length:] - self.feature_mean[0]) / (self.feature_std[0] + 1e-8),
                close_norm[-self.seq_length:]
            ])
        else:
            # 使用局部统计量
            seq = np.column_stack([
                returns[-self.seq_length:],
                (volumes[-self.seq_length:] - np.nanmean(volumes)) / (np.nanstd(volumes) + 1e-8),
                close_norm[-self.seq_length:]
            ])

        return seq.reshape(1, self.seq_length, 3)

    def load(self) -> bool:
        """加载模型"""
        model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
        stats_path = os.path.join(MODEL_DIR, 'lstm_stats.json')

        if not os.path.exists(model_path):
            print("LSTM模型不存在")
            return False

        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            from tensorflow import keras

            self.model = keras.models.load_model(model_path)
            self.is_loaded = True

            # 加载统计量
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    self.feature_mean = stats.get('mean', [1e6, 1e8, 1])
                    self.feature_std = stats.get('std', [0.02, 0.1, 0.1])

            print("LSTM模型加载成功")
            return True
        except Exception as e:
            print(f"LSTM模型加载失败: {e}")
            return False

    def save(self, stats: dict = None):
        """保存模型"""
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
        stats_path = os.path.join(MODEL_DIR, 'lstm_stats.json')

        if self.model:
            self.model.save(model_path)
            print(f"LSTM模型已保存: {model_path}")

        if stats:
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
            print(f"LSTM统计量已保存: {stats_path}")

    def predict(self, df: pd.DataFrame) -> float:
        """
        预测单只股票的未来收益

        Returns:
            预测的5日收益率 (-1 到 1)
        """
        if not self.is_loaded or self.model is None:
            return 0.0

        seq = self.prepare_sequence_from_dict(df)
        if seq is None:
            return 0.0

        try:
            pred = self.model.predict(seq, verbose=0)[0][0]
            # 限制在合理范围
            pred = np.clip(pred, -0.3, 0.3)
            return float(pred)
        except Exception as e:
            print(f"LSTM预测失败: {e}")
            return 0.0

    def predict_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """预测所有股票"""
        predictions = {}
        for code, df in data_dict.items():
            if df is not None and len(df) >= self.seq_length + 5:
                try:
                    predictions[code] = self.predict(df)
                except Exception as e:
                    predictions[code] = 0.0
            else:
                predictions[code] = 0.0
        return predictions

    def train(self, train_data: Dict[str, pd.DataFrame],
              validation_data: Dict[str, pd.DataFrame] = None,
              epochs: int = 20,
              batch_size: int = 32) -> dict:
        """
        训练LSTM模型

        Args:
            train_data: 训练数据 {code: DataFrame}
            validation_data: 验证数据
            epochs: 训练轮数
            batch_size: 批量大小

        Returns:
            训练历史
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow import keras

        # 准备训练序列
        X_train, y_train = [], []

        for code, df in train_data.items():
            if len(df) < self.seq_length + 10:
                continue

            # 生成多个样本 (滑动窗口)
            for i in range(10, len(df) - self.seq_length):
                window = df.iloc[i:i + self.seq_length]

                # 特征: [return, volume, close_norm]
                returns = window['close'].pct_change().fillna(0).values
                volumes = window['volume'].values
                close_norm = (window['close'].values / window['close'].iloc[0]) - 1

                seq = np.column_stack([returns, volumes, close_norm])
                X_train.append(seq)

                # 目标: 未来5日收益
                future_return = (df['close'].iloc[i + self.seq_length + 5] /
                               df['close'].iloc[i + self.seq_length]) - 1
                y_train.append(np.clip(future_return, -0.3, 0.3))

        if not X_train:
            print("没有足够的训练数据")
            return {}

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 计算统计量
        self.feature_mean = [np.mean(X_train[:, :, i]) for i in range(3)]
        self.feature_std = [np.std(X_train[:, :, i]) + 1e-8 for i in range(3)]

        # 标准化
        for i in range(3):
            X_train[:, :, i] = (X_train[:, :, i] - self.feature_mean[i]) / self.feature_std[i]

        print(f"训练数据: {len(X_train)} 样本")

        # 构建模型
        self.model = self.build_model()

        # 回调
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        # 训练
        validation_split = 0.2 if validation_data is None else None

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # 保存
        self.is_loaded = True
        self.save(stats={'mean': self.feature_mean, 'std': self.feature_std})

        return history.history


def train_lstm_model():
    """训练LSTM模型的脚本"""
    from src.simulation.evolving_portfolio import load_stock_data

    print("=" * 60)
    print("LSTM模型训练")
    print("=" * 60)

    # 加载数据
    import json
    pool_path = os.path.join(PROJECT_ROOT, 'data', 'expanded_stock_pool.json')
    with open(pool_path, 'r') as f:
        pool = json.load(f)
        codes = pool.get('stocks', [])[:50]  # 用50只股票训练

    print(f"\n加载 {len(codes)} 只股票数据...")
    data_dict = load_stock_data(codes)
    print(f"有效数据: {len(data_dict)} 只股票")

    # 划分训练/验证
    train_codes = list(data_dict.keys())[:40]
    val_codes = list(data_dict.keys())[40:50]

    train_data = {c: data_dict[c] for c in train_codes}
    val_data = {c: data_dict[c] for c in val_codes} if val_codes else None

    # 训练
    lstm = LSTMModel(seq_length=60)
    history = lstm.train(
        train_data,
        validation_data=val_data,
        epochs=20,
        batch_size=32
    )

    # 评估
    if val_data:
        print("\n验证集评估:")
        preds = lstm.predict_all(val_data)
        for code, pred in list(preds.items())[:5]:
            print(f"  {code}: {pred:.4f}")

    print("\n训练完成!")
    return lstm


def main():
    """测试LSTM模型"""
    lstm = LSTMModel()

    # 尝试加载
    if not lstm.load():
        print("开始训练新模型...")
        lstm = train_lstm_model()

    # 测试预测
    from src.simulation.evolving_portfolio import load_stock_data
    import json

    pool_path = os.path.join(PROJECT_ROOT, 'data', 'expanded_stock_pool.json')
    with open(pool_path, 'r') as f:
        pool = json.load(f)
        codes = pool.get('stocks', [])[:10]

    data_dict = load_stock_data(codes)

    print("\nLSTM预测结果:")
    preds = lstm.predict_all(data_dict)
    for code, pred in sorted(preds.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {code}: {pred:+.4f}")


if __name__ == '__main__':
    main()
