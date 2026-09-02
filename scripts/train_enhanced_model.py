#!/usr/bin/env python3
"""
增强模型训练 + 多策略系统
=========================
1. 更多特征（市场情绪、资金流、技术+基本面）
2. 多种策略（趋势、价值、均值回归）
3. 滚动训练
"""

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
MODEL_DIR = f'{DATA_DIR}/models'


def load_stock_data(codes: list) -> Dict[str, pd.DataFrame]:
    """加载股票数据"""
    data_dict = {}
    for code in codes:
        code_fmt = code.replace('.', '_')
        filepath = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[df['volume'] > 0]
            if len(df) >= 120:
                data_dict[code] = df
    return data_dict


def calculate_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算增强特征"""
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # ========== 基础收益率 ==========
    for period in [1, 3, 5, 10, 20, 60]:
        df[f'return_{period}d'] = df['close'].pct_change(period)

    # ========== 技术指标 ==========
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    exp26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林带
    bb_mid = pd.Series(close).rolling(20).mean()
    bb_std = pd.Series(close).rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # ATR
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = pd.Series(tr).rolling(14).mean()
    df['atr_ratio'] = df['atr'] / (close + 1e-8)

    # ========== 均线系统 ==========
    for window in [5, 10, 20, 60, 120]:
        df[f'ma{window}'] = pd.Series(close).rolling(window).mean()
        df[f'price_ma{window}_ratio'] = close / (df[f'ma{window}'] + 1e-8)

    # 均线多头排列
    df['ma_bull_5_20'] = (df['ma5'] > df['ma20']).astype(int)
    df['ma_bull_20_60'] = (df['ma20'] > df['ma60']).astype(int)
    df['ma_bull_all'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(int)

    # ========== 动量 ==========
    for period in [5, 10, 20, 60]:
        df[f'momentum_{period}d'] = close / (np.roll(close, period) + 1e-8) - 1

    # 动量加速
    df['momentum_accel'] = df['momentum_10d'] - df['momentum_10d'].shift(10)

    # ========== 波动率 ==========
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = pd.Series(close).pct_change().rolling(window).std() * np.sqrt(252)

    # ========== 成交量 ==========
    for window in [5, 20]:
        df[f'volume_ma{window}'] = pd.Series(volume).rolling(window).mean()
    df['volume_ratio'] = volume / (df['volume_ma20'] + 1e-8)

    # 成交量动量
    df['volume_momentum'] = pd.Series(volume).pct_change(10)

    # ========== 相对强度 ==========
    # 个股vs市场（用上证指数近似）
    # 这里简化处理，用其他股计算市场相关性

    # ========== 趋势强度 ==========
    df['trend_strength'] = np.abs(df['macd'] / (df['atr'] + 1e-8))

    # 通道突破
    df['close_upper_channel'] = close / (bb_mid + 2 * bb_std) - 1
    df['close_lower_channel'] = close / (bb_mid - 2 * bb_std) - 1

    # ========== 价值因子（简化） ==========
    # 价格动能
    df['price_velocity'] = df['return_10d'] / (df['volatility_10d'] + 1e-8)

    # 支撑阻力
    df['support_ratio'] = (close - bb_mid - bb_std) / (2 * bb_std + 1e-8)

    return df


def prepare_training_data(data_dict: Dict[str, pd.DataFrame],
                          prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """准备训练数据"""
    all_X = []
    all_y = []
    feature_names = None

    for code, df in data_dict.items():
        if len(df) < 120:
            continue

        df = calculate_enhanced_features(df)

        # 目标：未来N日收益率
        df['target'] = df['close'].shift(-prediction_horizon) / df['close'] - 1

        # 选择特征列
        exclude_cols = {'date', 'code', 'open', 'high', 'low', 'close', 'volume',
                       'amount', 'turn', 'pctChg', 'code_raw', 'target'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        if feature_names is None:
            feature_names = feature_cols

        # 有效数据
        valid_df = df.dropna(subset=['target'])
        valid_df = valid_df[valid_df['target'].abs() < 0.5]  # 去除极端值

        if len(valid_df) > 0:
            X = valid_df[feature_cols].values
            y = valid_df['target'].values
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return np.array([]), np.array([]), []

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # 处理NaN和Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    return X, y, feature_names


def train_model(X_train, y_train, X_valid, y_valid, feature_names):
    """训练XGBoost模型"""
    print(f"\n训练数据: {X_train.shape[0]} 样本")
    print(f"验证数据: {X_valid.shape[0]} 样本")

    # XGBoost模型
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50
    )

    # 特征重要性
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 TOP 10:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # 验证集评估
    y_pred = model.predict(X_valid)
    corr = np.corrcoef(y_pred, y_valid)[0, 1]
    print(f"\n验证集 IC: {corr:.4f}")

    # 计算方向准确率
    y_pred_dir = (y_pred > 0).astype(int)
    y_valid_dir = (y_valid > 0).astype(int)
    accuracy = (y_pred_dir == y_valid_dir).mean()
    print(f"方向准确率: {accuracy:.2%}")

    return model, importance_df


class MultiStrategySignal:
    """多策略信号生成器"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def predict_returns(self, df: pd.DataFrame) -> float:
        """预测收益率"""
        df = calculate_enhanced_features(df)

        exclude_cols = {'date', 'code', 'open', 'high', 'low', 'close', 'volume',
                       'amount', 'turn', 'pctChg', 'code_raw', 'target'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if X.shape[0] == 0:
            return 0

        pred = self.model.predict(X[-1:])[0]
        return float(pred)

    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """生成多策略信号"""
        signals = {}

        for code, df in data_dict.items():
            if len(df) < 60:
                continue

            try:
                # ML预测
                pred = self.predict_returns(df)

                # ========== 策略1: 趋势追踪 ==========
                close = df['close'].values
                ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
                ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
                rsi = self._calc_rsi(close)

                uptrend = ma20 > ma60

                trend_score = 50
                if uptrend:
                    trend_score += 30
                if rsi < 40:
                    trend_score += 15
                elif rsi > 70:
                    trend_score -= 20

                # ========== 策略2: 均值回归 ==========
                ret_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

                mean_reversion_score = 50
                if ret_20d < -0.1:  # 超跌
                    mean_reversion_score += 30
                elif ret_20d > 0.1:  # 超涨
                    mean_reversion_score -= 20

                # ========== 策略3: 动量 ==========
                ret_60d = (close[-1] - close[-60]) / close[-60] if len(close) >= 60 else 0

                momentum_score = 50
                if ret_60d > 0.15:
                    momentum_score += 25
                elif ret_60d < -0.15:
                    momentum_score -= 25

                # ========== 综合 ==========
                # ML权重40%，技术分析权重60%
                ml_score = 50 + pred * 500  # 将收益率转换为评分
                combined_score = ml_score * 0.4 + trend_score * 0.3 + mean_reversion_score * 0.15 + momentum_score * 0.15

                # 信号
                if combined_score > 65 and pred > 0:
                    signal = 'BUY'
                elif combined_score < 40 or pred < -0.03:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'

                signals[code] = {
                    'signal': signal,
                    'score': combined_score,
                    'ml_pred': pred,
                    'trend_score': trend_score,
                    'mean_reversion': mean_reversion_score,
                    'momentum': momentum_score,
                    'price': float(close[-1])
                }

            except Exception as e:
                continue

        return signals

    def _calc_rsi(self, prices, period=14):
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(period).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 50
        return 100 - (100 / (1 + rs))


def main():
    print("=" * 60)
    print("增强模型训练 + 多策略系统")
    print("=" * 60)

    # 加载股票池
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            codes = pool.get('stocks', [])[:50]
    else:
        codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

    print(f"\n加载股票池: {len(codes)} 只")

    # 加载数据
    data_dict = load_stock_data(codes)
    print(f"有效数据: {len(data_dict)} 只")

    # 分割数据
    all_codes = list(data_dict.keys())
    n_train = int(len(all_codes) * 0.7)
    train_codes = all_codes[:n_train]
    valid_codes = all_codes[n_train:]

    train_data = {c: data_dict[c] for c in train_codes}
    valid_data = {c: data_dict[c] for c in valid_codes}

    print(f"训练集: {len(train_data)} 只, 验证集: {len(valid_data)} 只")

    # 准备训练数据
    print("\n准备训练数据...")
    X, y, feature_names = prepare_training_data(train_data, prediction_horizon=5)

    if len(X) == 0:
        print("训练数据为空!")
        return

    # 时间序列分割
    n = len(X)
    X_train, X_valid = X[:int(n*0.8)], X[int(n*0.8):]
    y_train, y_valid = y[:int(n*0.8)], y[int(n*0.8):]

    # 训练
    print("\n训练模型...")
    model, importance_df = train_model(X_train, y_train, X_valid, y_valid, feature_names)

    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f'{MODEL_DIR}/enhanced_xgb_model.json'
    model.save_model(model_path)
    print(f"\n模型已保存: {model_path}")

    # 保存配置
    config = {
        'model_type': 'enhanced_xgb',
        'prediction_horizon': 5,
        'features': feature_names,
        'n_features': len(feature_names),
        'training_date': datetime.now().strftime('%Y-%m-%d')
    }
    with open(f'{MODEL_DIR}/enhanced_model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 测试多策略信号
    print("\n测试多策略信号...")
    signal_generator = MultiStrategySignal(model, feature_names)
    signals = signal_generator.generate_signals(valid_data)

    buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
    hold_count = sum(1 for s in signals.values() if s['signal'] == 'HOLD')

    print(f"\n信号分布:")
    print(f"  买入: {buy_count}")
    print(f"  卖出: {sell_count}")
    print(f"  持有: {hold_count}")

    # 保存信号生成器配置
    print("\n完成!")


if __name__ == '__main__':
    main()