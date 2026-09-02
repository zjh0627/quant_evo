#!/usr/bin/env python3
"""
混合预测服务
============

结合 LSTM 时序预测 + XGBoost 横截面预测 + 技术指标规则

架构:
Layer 1: LSTM (时序) → 输出预测收益
Layer 2: XGBoost (横截面) → 输出预测收益
Layer 3: 融合 → 综合信号
Layer 4: 规则过滤 → 止损/止盈/风控
"""

import os
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')


class HybridPredictionService:
    """
    混合预测服务

    结合多种模型的预测:
    - LSTM: 时序依赖
    - XGBoost: 横截面特征
    - 规则: 技术指标信号
    """

    def __init__(self,
                 lstm_weight: float = 0.4,
                 xgb_weight: float = 0.4,
                 rule_weight: float = 0.2):
        """
        Args:
            lstm_weight: LSTM预测权重
            xgb_weight: XGBoost预测权重
            rule_weight: 规则信号权重
        """
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.rule_weight = rule_weight

        self.lstm_model = None
        self.xgb_model = None
        self.is_loaded = False

        # 模型配置
        self.xgb_config = None
        self.lstm_seq_length = 60

    def load_models(self) -> bool:
        """加载所有模型"""
        print("=" * 60)
        print("加载混合预测模型")
        print("=" * 60)

        # 1. 加载XGBoost
        self._load_xgboost()

        # 2. 加载LSTM
        self._load_lstm()

        # 3. 加载配置
        self._load_config()

        self.is_loaded = True
        print("混合模型加载完成")
        return True

    def _load_xgboost(self):
        """加载XGBoost模型"""
        import xgboost as xgb

        xgb_path = os.path.join(MODEL_DIR, 'enhanced_xgb_model.json')

        if os.path.exists(xgb_path):
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(xgb_path)
            print(f"  ✓ XGBoost模型加载")
        else:
            print(f"  ✗ XGBoost模型不存在: {xgb_path}")

    def _load_lstm(self):
        """加载LSTM模型"""
        try:
            from src.ml.lstm_model import LSTMModel
            self.lstm_model = LSTMModel(seq_length=self.lstm_seq_length)
            if self.lstm_model.load():
                print(f"  ✓ LSTM模型加载")
            else:
                print(f"  ! LSTM模型未加载 (将只用XGBoost)")
                self.lstm_model = None
        except Exception as e:
            print(f"  ! LSTM加载失败: {e}")
            self.lstm_model = None

    def _load_config(self):
        """加载特征配置"""
        config_path = os.path.join(MODEL_DIR, 'enhanced_model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.xgb_config = json.load(f)
                print(f"  ✓ 模型配置加载")

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算XGBoost所需特征 - 使用统一的特征计算"""
        # 使用与训练时一致的增强特征计算
        from src.simulation.evolving_portfolio import calculate_enhanced_features
        return calculate_enhanced_features(df)

    def get_rule_signal(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        基于规则生成信号分数

        Returns:
            (score 0-100, signal_type)
        """
        if len(df) < 60:
            return 50, 'HOLD'

        close = df['close'].values
        volume = df['volume'].values

        score = 50  # 中性分数

        # 1. 趋势检查
        ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
        ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        ma60 = pd.Series(close).rolling(60).mean().iloc[-1]

        if ma5 > ma20 > ma60:
            score += 20  # 多头排列
        elif ma5 < ma20 < ma60:
            score -= 20  # 空头排列

        # 2. RSI检查
        rsi = self._calc_rsi(close)
        if rsi < 30:
            score += 15  # 超卖
        elif rsi > 70:
            score -= 15  # 超买

        # 3. 动量检查
        ret_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
        if ret_20d < -0.1:
            score += 10  # 超跌反弹可能
        elif ret_20d > 0.1:
            score -= 5   # 涨多回调可能

        # 4. 成交量检查
        vol_ma5 = pd.Series(volume).rolling(5).mean().iloc[-1]
        vol_ma20 = pd.Series(volume).rolling(20).mean().iloc[-1]
        if volume[-1] > vol_ma20 * 1.5:
            score += 5   # 放量

        # 限制范围
        score = max(0, min(100, score))

        # 信号类型
        if score >= 65:
            signal = 'BUY'
        elif score <= 35:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return score, signal

    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def predict_xgboost(self, df: pd.DataFrame) -> float:
        """XGBoost预测"""
        if self.xgb_model is None:
            return 0.0

        df_feat = self.calculate_features(df)

        exclude_cols = {'date', 'code', 'open', 'high', 'low', 'close', 'volume',
                       'amount', 'turn', 'pctChg', 'code_raw', 'target'}

        if self.xgb_config and 'features' in self.xgb_config:
            feature_cols = [c for c in self.xgb_config['features'] if c not in exclude_cols]
        else:
            feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

        feature_cols = [c for c in feature_cols if c in df_feat.columns]

        if not feature_cols:
            return 0.0

        X = df_feat[feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        try:
            pred = self.xgb_model.predict(X[-1:])[0]
            # 转换为0-100分数
            score = 50 + pred * 100
            return float(score)
        except:
            return 50.0

    def predict_lstm(self, df: pd.DataFrame) -> float:
        """LSTM预测"""
        if self.lstm_model is None:
            return 50.0

        try:
            # LSTM返回的是收益率预测 (-0.3 到 0.3)
            ret_pred = self.lstm_model.predict(df)
            # 转换为50中心的分数
            score = 50 + ret_pred * 166.67  # 0.3 * 166.67 ≈ 50
            return float(np.clip(score, 20, 80))
        except:
            return 50.0

    def predict_single(self, df: pd.DataFrame) -> Dict:
        """
        对单只股票进行综合预测

        Returns:
            {
                'xgb_score': XGBoost分数,
                'lstm_score': LSTM分数,
                'rule_score': 规则分数,
                'ensemble': 综合分数,
                'signal': 'BUY'/'SELL'/'HOLD',
                'confidence': 置信度
            }
        """
        if df is None or len(df) < 60:
            return self._default_prediction()

        # 各模型预测
        xgb_score = self.predict_xgboost(df)
        lstm_score = self.predict_lstm(df)
        rule_score, rule_signal = self.get_rule_signal(df)

        # 加权集成
        ensemble = (
            xgb_score * self.xgb_weight +
            lstm_score * self.lstm_weight +
            rule_score * self.rule_weight
        )

        # 最终信号
        if ensemble >= 60:
            signal = 'BUY'
        elif ensemble <= 40:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # 置信度 (分数距离50越远,置信度越高)
        confidence = abs(ensemble - 50) / 50

        return {
            'xgb_score': round(xgb_score, 2),
            'lstm_score': round(lstm_score, 2),
            'rule_score': round(rule_score, 2),
            'ensemble': round(ensemble, 2),
            'signal': signal,
            'confidence': round(confidence, 2),
            'trend': rule_signal
        }

    def predict_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """对所有股票进行预测"""
        predictions = {}

        for code, df in data_dict.items():
            if df is not None and len(df) >= 60:
                try:
                    predictions[code] = self.predict_single(df)
                except Exception as e:
                    print(f"预测 {code} 失败: {e}")
                    predictions[code] = self._default_prediction()
            else:
                predictions[code] = self._default_prediction()

        return predictions

    def _default_prediction(self) -> Dict:
        """默认预测"""
        return {
            'xgb_score': 50,
            'lstm_score': 50,
            'rule_score': 50,
            'ensemble': 50,
            'signal': 'HOLD',
            'confidence': 0,
            'trend': 'HOLD'
        }

    def get_top_signals(self, predictions: Dict[str, Dict],
                       n: int = 10,
                       by: str = 'ensemble',
                       signal_type: str = 'BUY') -> List[tuple]:
        """
        获取最佳信号

        Args:
            predictions: 预测结果
            n: 返回数量
            by: 'ensemble', 'xgb', 'lstm', 'rule'
            signal_type: 'BUY' 或 'SELL'

        Returns:
            [(code, prediction_dict), ...]
        """
        # 过滤信号类型
        filtered = {k: v for k, v in predictions.items() if v['signal'] == signal_type}

        # 排序
        if by == 'xgb':
            items = sorted(filtered.items(), key=lambda x: x[1]['xgb_score'], reverse=True)
        elif by == 'lstm':
            items = sorted(filtered.items(), key=lambda x: x[1]['lstm_score'], reverse=True)
        elif by == 'rule':
            items = sorted(filtered.items(), key=lambda x: x[1]['rule_score'], reverse=True)
        else:
            items = sorted(filtered.items(), key=lambda x: x[1]['ensemble'], reverse=True)

        return items[:n]


def test_hybrid_prediction():
    """测试混合预测"""
    from src.simulation.evolving_portfolio import load_stock_data

    print("=" * 60)
    print("混合预测服务测试")
    print("=" * 60)

    # 加载数据
    import json
    pool_path = os.path.join(PROJECT_ROOT, 'data', 'expanded_stock_pool.json')
    with open(pool_path, 'r') as f:
        pool = json.load(f)
        codes = pool.get('stocks', [])[:20]

    print(f"\n加载 {len(codes)} 只股票...")
    data_dict = load_stock_data(codes)
    print(f"有效数据: {len(data_dict)} 只股票")

    # 混合预测
    service = HybridPredictionService()
    service.load_models()

    print("\n预测结果 (按综合分数排序):")
    predictions = service.predict_all(data_dict)

    buy_signals = service.get_top_signals(predictions, n=10, signal_type='BUY')
    print("\n买入信号:")
    for code, pred in buy_signals[:5]:
        print(f"  {code}: 综合={pred['ensemble']:.1f}, "
              f"XGB={pred['xgb_score']:.1f}, LSTM={pred['lstm_score']:.1f}, "
              f"规则={pred['rule_score']:.1f}")

    sell_signals = service.get_top_signals(predictions, n=5, signal_type='SELL')
    print("\n卖出信号:")
    for code, pred in sell_signals[:3]:
        print(f"  {code}: 综合={pred['ensemble']:.1f}")


def main():
    test_hybrid_prediction()


if __name__ == '__main__':
    main()
