#!/usr/bin/env python3
"""
滚动训练模块
============

实现滚动窗口训练的ML策略，用于更真实的模型评估和部署

核心思想：
- 使用固定时间窗口的数据进行训练
- 随着时间向前滚动，窗口也跟着滚动
- 每次预测都使用截至当前时刻的历史数据
- 避免未来信息泄露，更真实地模拟实际交易场景

典型滚动训练流程：
1. 初始窗口：用 [T-180, T] 数据训练
2. 预测：用训练好的模型预测 T+5 的收益
3. 滚动：窗口向前移动60天，用 [T-120, T] 重新训练
4. 重复...
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.ml.model_training import FeatureEngineer, ModelConfig
from src.ml.prediction_service import PredictionService

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
ROLLING_CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache', 'rolling')


@dataclass
class RollingConfig:
    """滚动训练配置"""
    train_window: int = 180       # 训练窗口天数
    test_window: int = 60        # 测试窗口天数（或预测频率）
    step_size: int = 20          # 滚动步长（每隔多少天重新训练）
    min_train_samples: int = 100  # 最少训练样本数
    prediction_horizon: int = 5   # 预测未来N日收益
    lookback_window: int = 20    # 特征回看窗口

    # 模型参数
    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }


class RollingTrainer:
    """
    滚动训练器

    使用滚动窗口方式训练ML模型，更真实地评估模型在未来数据上的表现
    """

    def __init__(self, config: RollingConfig = None):
        self.config = config or RollingConfig()
        self.feature_engineer = FeatureEngineer(lookback=self.config.lookback_window)
        self.model = None
        self.scaler = None
        self.is_trained = False

        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(ROLLING_CACHE_DIR, exist_ok=True)

    def _get_training_dates(self, stock_data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        """获取所有股票数据的公共日期范围"""
        all_dates = None
        for code, df in stock_data.items():
            if df is None or len(df) < 50:
                continue
            dates = pd.to_datetime(df['date'])
            if all_dates is None:
                all_dates = set(dates)
            else:
                all_dates &= set(dates)

        if all_dates is None:
            return []
        return sorted(all_dates)

    def _filter_by_date_range(self, df: pd.DataFrame,
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp) -> pd.DataFrame:
        """按日期范围筛选数据"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    def _create_training_data(self, stock_data: Dict[str, pd.DataFrame],
                               end_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建训练数据（截止到指定日期）

        Args:
            stock_data: 股票数据
            end_date: 训练数据截止日期

        Returns:
            (X, y, feature_names)
        """
        all_features = []
        all_labels = []

        start_date = end_date - timedelta(days=self.config.train_window)

        for code, df in stock_data.items():
            if df is None or len(df) < 100:
                continue
            if code.startswith('bj.'):  # 跳过北交所
                continue

            try:
                # 筛选日期范围
                df_window = self._filter_by_date_range(df, start_date, end_date)
                if len(df_window) < 50:
                    continue

                # 计算特征（只计算一次）
                feature_matrix = self.feature_engineer.create_feature_matrix(df_window)
                feature_names = feature_matrix.columns.tolist()

                # 对特征矩阵进行填充处理NaN
                feature_matrix = feature_matrix.fillna(method='ffill').fillna(method='bfill')
                # 如果还有NaN，用0填充（极少数情况）
                feature_matrix = feature_matrix.fillna(0)

                # 在df_window上创建目标变量（用于获取有效索引）
                df_with_target = df_window.copy()
                df_with_target['future_return'] = (
                    df_with_target['close'].shift(-self.config.prediction_horizon) /
                    df_with_target['close'] - 1
                )

                # 移除NaN和异常值（只对目标变量过滤，不对特征过滤）
                valid_mask = ~df_with_target['future_return'].isna()
                valid_mask &= np.abs(df_with_target['future_return']) < 0.5
                valid_mask &= ~np.isinf(df_with_target['future_return'])

                feature_matrix = feature_matrix[valid_mask]
                labels = df_with_target.loc[valid_mask, 'future_return']

                if len(feature_matrix) > 0:
                    all_features.append(feature_matrix.values)
                    all_labels.append(labels.values)

            except Exception as e:
                continue

        if not all_features:
            return np.array([]), np.array([]), []

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        # 最终异常值过滤
        valid_mask = (
            (np.abs(y) < 0.5) &
            (~np.isnan(y)) &
            (~np.isinf(y))
        )
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y, feature_names

    def _create_prediction_data(self, df: pd.DataFrame,
                                 cutoff_date: pd.Timestamp) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        为预测创建特征向量（使用cutoff_date之前的数据）

        Args:
            df: 单只股票数据
            cutoff_date: 预测使用数据的截止日期

        Returns:
            (feature_vector, feature_names)
        """
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df_window = df[df['date'] <= cutoff_date]

            if len(df_window) < 30:
                return None, []

            # 计算特征
            df_features = self.feature_engineer.calculate_features(df_window)
            feature_matrix = self.feature_engineer.create_feature_matrix(df_features)
            feature_names = feature_matrix.columns.tolist()

            # 取最后一行
            last_row = feature_matrix.iloc[-1:]

            if last_row.isna().any().any():
                last_row = last_row.fillna(method='ffill').fillna(method='bfill')

            return last_row.values, feature_names

        except Exception as e:
            return None, []

    def train_at_timestamp(self, stock_data: Dict[str, pd.DataFrame],
                           train_end_date: pd.Timestamp) -> bool:
        """
        在指定时间点训练模型

        Args:
            stock_data: 股票数据
            train_end_date: 训练数据截止日期

        Returns:
            是否训练成功
        """
        print(f"\n训练时间点: {train_end_date.strftime('%Y-%m-%d')}")

        # 创建训练数据
        X, y, feature_names = self._create_training_data(stock_data, train_end_date)

        if len(X) < self.config.min_train_samples:
            print(f"训练样本不足: {len(X)} < {self.config.min_train_samples}")
            return False

        print(f"训练样本数: {len(X)}, 特征数: {X.shape[1]}")

        try:
            import xgboost as xgb

            params = self.config.xgb_params.copy()
            n_estimators = params.pop('n_estimators')

            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                **params,
                random_state=42,
                n_jobs=-1,
            )

            self.model.fit(X, y, verbose=False)

            # 保存特征名
            self.feature_names = feature_names
            self.train_date = train_end_date

            # 简单评估（用训练集上的表现作为参考）
            train_pred = self.model.predict(X)
            train_ic = np.corrcoef(train_pred, y)[0, 1] if len(y) > 1 else 0
            print(f"训练IC: {train_ic:.4f}")

            self.is_trained = True
            return True

        except Exception as e:
            print(f"训练失败: {e}")
            return False

    def predict_at_timestamp(self, stock_data: Dict[str, pd.DataFrame],
                              predict_date: pd.Timestamp) -> Dict[str, float]:
        """
        在指定时间点进行预测

        Args:
            stock_data: 股票数据
            predict_date: 预测时间点

        Returns:
            {code: predicted_return}
        """
        if not self.is_trained:
            return {}

        predictions = {}

        for code, df in stock_data.items():
            if df is None or len(df) < 60:
                continue

            try:
                features, feature_names = self._create_prediction_data(df, predict_date)

                if features is None or len(feature_names) == 0:
                    continue

                # 确保特征顺序一致
                if not hasattr(self, 'feature_names'):
                    continue

                # 如果特征名不完全匹配，使用位置索引
                pred = self.model.predict(features)[0]
                predictions[code] = float(pred)

            except Exception as e:
                continue

        return predictions

    def rolling_train_and_evaluate(self, stock_data: Dict[str, pd.DataFrame],
                                    start_date: pd.Timestamp,
                                    end_date: pd.Timestamp) -> Dict:
        """
        滚动训练和评估

        Args:
            stock_data: 股票数据
            start_date: 评估开始日期
            end_date: 评估结束日期

        Returns:
            滚动评估结果
        """
        print(f"\n{'='*60}")
        print(f"滚动训练评估")
        print(f"评估期间: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"训练窗口: {self.config.train_window}天")
        print(f"滚动步长: {self.config.step_size}天")
        print(f"{'='*60}")

        # 获取评估期间内的所有预测时间点
        all_dates = self._get_training_dates(stock_data)
        if not all_dates:
            print("无可用日期")
            return {}

        # 筛选在评估期间内的时间点
        eval_dates = [d for d in all_dates if start_date <= d <= end_date]
        if not eval_dates:
            print(f"评估期间内无数据: {start_date} ~ {end_date}")
            return {}

        # 滚动训练和预测
        results = []
        current_date_idx = 0

        while current_date_idx < len(eval_dates):
            predict_date = eval_dates[current_date_idx]

            # 训练数据截止到预测日期之前
            train_end = predict_date - timedelta(days=1)

            # 跳过训练数据不足的时间点
            if train_end < all_dates[0] + timedelta(days=self.config.train_window):
                current_date_idx += 1
                continue

            # 训练
            success = self.train_at_timestamp(stock_data, train_end)
            if not success:
                current_date_idx += self.config.step_size
                continue

            # 预测
            predictions = self.predict_at_timestamp(stock_data, predict_date)

            # 获取真实收益（用于评估）
            true_returns = {}
            for code, df in stock_data.items():
                if df is None:
                    continue
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                # 找到预测日期之后（含）的第N个交易日
                future_date = predict_date + timedelta(days=self.config.prediction_horizon)
                idx = df[df['date'] >= future_date]

                if len(idx) > 0:
                    # 用 <= predict_date 获取预测日之前（含）的最后一个交易日作为入场价
                    entry_df = df[df['date'] <= predict_date]
                    if len(entry_df) == 0:
                        continue
                    entry_price = entry_df.iloc[-1]['close']
                    exit_price = idx.iloc[0]['close']
                    ret = (exit_price / entry_price) - 1
                    true_returns[code] = ret

            # 计算IC（信息系数）
            valid_pairs = [(c, predictions.get(c), true_returns.get(c))
                          for c in set(predictions.keys()) & set(true_returns.keys())]

            if len(valid_pairs) >= 3:
                pred_vals = np.array([p[1] for p in valid_pairs])
                true_vals = np.array([p[2] for p in valid_pairs])

                # 过滤异常值
                mask = (np.abs(pred_vals) < 0.5) & (np.abs(true_vals) < 0.5)
                pred_vals = pred_vals[mask]
                true_vals = true_vals[mask]

                if len(pred_vals) >= 3:
                    ic = np.corrcoef(pred_vals, true_vals)[0, 1]
                    mae = np.mean(np.abs(pred_vals - true_vals))

                    results.append({
                        'date': predict_date.strftime('%Y-%m-%d'),
                        'n_stocks': len(pred_vals),
                        'ic': ic if not np.isnan(ic) else 0,
                        'mae': mae,
                        'pred_mean': np.mean(pred_vals),
                        'pred_std': np.std(pred_vals),
                        'true_mean': np.mean(true_vals),
                    })

                    if len(results) % 5 == 0:
                        print(f"  {predict_date.strftime('%Y-%m-%d')}: IC={ic:.4f}, "
                              f"MAE={mae:.4f}, 股票数={len(pred_vals)}")

            current_date_idx += self.config.step_size

        # 汇总结果
        if not results:
            print("无有效结果")
            return {}

        ic_values = [r['ic'] for r in results]
        mae_values = [r['mae'] for r in results]

        summary = {
            'n_periods': len(results),
            'mean_ic': np.mean(ic_values),
            'std_ic': np.std(ic_values),
            'ic_ir': np.mean(ic_values) / (np.std(ic_values) + 1e-8),
            'mean_mae': np.mean(mae_values),
            'positive_ic_ratio': sum(1 for ic in ic_values if ic > 0) / len(ic_values),
            'details': results
        }

        print(f"\n{'='*60}")
        print(f"滚动评估汇总")
        print(f"{'='*60}")
        print(f"评估期数: {summary['n_periods']}")
        print(f"平均IC: {summary['mean_ic']:.4f}")
        print(f"IC标准差: {summary['std_ic']:.4f}")
        print(f"IC_IR比率: {summary['ic_ir']:.4f}")
        print(f"正IC比例: {summary['positive_ic_ratio']:.1%}")
        print(f"平均MAE: {summary['mean_mae']:.4f}")

        return summary

    def generate_rolling_predictions(self, stock_data: Dict[str, pd.DataFrame],
                                     target_date: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        """
        生成滚动预测（用于实际交易决策）

        使用target_date之前的数据训练模型，然后对所有股票进行预测

        Args:
            stock_data: 股票数据
            target_date: 目标预测日期

        Returns:
            {code: {'prediction': float, 'confidence': float}}
        """
        train_end = target_date - timedelta(days=1)

        # 训练模型
        success = self.train_at_timestamp(stock_data, train_end)
        if not success:
            print(f"训练失败，无法生成预测")
            return {}

        # 预测
        predictions = self.predict_at_timestamp(stock_data, target_date)

        # 计算置信度（基于预测值的离散程度）
        if predictions:
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values) + 1e-8

            result = {}
            for code, pred in predictions.items():
                # 置信度：预测值距离均值多少个标准差
                z_score = abs(pred - np.mean(pred_values)) / pred_std
                confidence = 1 / (1 + z_score)  # 转换为0-1之间的置信度
                result[code] = {
                    'prediction': pred,
                    'confidence': confidence
                }

            return result

        return {}


class RollingBacktester:
    """
    滚动回测器

    在每个回测时点：
    1. 使用历史数据训练模型
    2. 生成预测
    3. 根据预测选择股票
    4. 执行交易
    5. 滚动到下一个时点
    """

    def __init__(self, initial_capital: float = 1000000,
                 rolling_config: RollingConfig = None):
        self.initial_capital = initial_capital
        self.rolling_config = rolling_config or RollingConfig()
        self.trainer = RollingTrainer(rolling_config)

    def run(self, stock_data: Dict[str, pd.DataFrame],
            start_date: str,
            end_date: str,
            top_n: int = 10,
            buy_threshold: float = 0.01,
            sell_threshold: float = -0.01) -> Dict:
        """
        运行滚动回测

        Args:
            stock_data: 股票数据
            start_date: 回测开始日期
            end_date: 回测结束日期
            top_n: 持仓股票数量
            buy_threshold: 买入阈值（预测收益率大于此值买入）
            sell_threshold: 卖出阈值（预测收益率小于此值卖出）

        Returns:
            回测结果
        """
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # 初始化
        capital = self.initial_capital
        position = {}  # {code: {'shares': int, 'entry_price': float}}
        equity_curve = []
        trades = []

        current_date = start_ts
        step = self.rolling_config.step_size

        # 获取所有可用日期
        all_dates = []
        for df in stock_data.values():
            dates = pd.to_datetime(df['date']).tolist()
            all_dates.extend(dates)
        all_dates = sorted(set(all_dates))
        all_dates = [d for d in all_dates if d >= start_ts and d <= end_ts]

        print(f"\n{'='*60}")
        print(f"滚动回测")
        print(f"期间: {start_date} ~ {end_date}")
        print(f"初始资金: {capital:,.2f}")
        print(f"持仓上限: {top_n}只")
        print(f"{'='*60}")

        while current_date < end_ts:
            # 训练模型
            train_end = current_date - timedelta(days=1)
            success = self.trainer.train_at_timestamp(stock_data, train_end)
            if not success:
                current_date += timedelta(days=step)
                continue

            # 获取预测
            predictions = self.trainer.predict_at_timestamp(stock_data, current_date)

            if not predictions:
                current_date += timedelta(days=step)
                continue

            # 排序预测
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

            # 目标持仓
            buy_list = [code for code, pred in sorted_preds[:top_n] if pred > buy_threshold]
            sell_list = [code for code, pos in position.items()
                        if predictions.get(code, 0) < sell_threshold]

            # 执行卖出
            for code in sell_list:
                if code in position:
                    shares = position[code]['shares']
                    entry = position[code]['entry_price']

                    # 获取当前价格
                    df = stock_data.get(code)
                    if df is not None:
                        df = df.copy()
                        df['date'] = pd.to_datetime(df['date'])
                        curr_price_df = df[df['date'] <= current_date]
                        if len(curr_price_df) > 0:
                            curr_price = curr_price_df.iloc[-1]['close']
                            pnl = (curr_price - entry) * shares
                            capital += curr_price * shares

                            trades.append({
                                'date': current_date.strftime('%Y-%m-%d'),
                                'action': 'sell',
                                'code': code,
                                'price': curr_price,
                                'shares': shares,
                                'pnl': pnl
                            })

                            del position[code]

            # 执行买入
            available_slot = top_n - len(position)
            for code in buy_list[:available_slot]:
                if code not in position and code in predictions:
                    df = stock_data.get(code)
                    if df is not None:
                        df = df.copy()
                        df['date'] = pd.to_datetime(df['date'])
                        curr_price_df = df[df['date'] <= current_date]
                        if len(curr_price_df) > 0:
                            curr_price = curr_price_df.iloc[-1]['close']
                            shares = int(capital * 0.8 / (top_n * curr_price))
                            if shares > 0:
                                cost = shares * curr_price
                                capital -= cost
                                position[code] = {
                                    'shares': shares,
                                    'entry_price': curr_price
                                }

                                trades.append({
                                    'date': current_date.strftime('%Y-%m-%d'),
                                    'action': 'buy',
                                    'code': code,
                                    'price': curr_price,
                                    'shares': shares
                                })

            # 记录权益
            total_value = capital
            for code, pos in position.items():
                df = stock_data.get(code)
                if df is not None:
                    df = df.copy()
                    df['date'] = pd.to_datetime(df['date'])
                    curr_price_df = df[df['date'] <= current_date]
                    if len(curr_price_df) > 0:
                        total_value += curr_price_df.iloc[-1]['close'] * pos['shares']

            equity_curve.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'value': total_value
            })

            current_date += timedelta(days=step)

        # 平仓
        final_value = capital
        for code, pos in position.items():
            df = stock_data.get(code)
            if df is not None:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                curr_price_df = df[df['date'] <= end_ts]
                if len(curr_price_df) > 0:
                    final_value += curr_price_df.iloc[-1]['close'] * pos['shares']

        # 计算指标
        df_eq = pd.DataFrame(equity_curve)
        df_eq['date'] = pd.to_datetime(df_eq['date'])
        returns = df_eq['value'].pct_change().dropna()

        total_return = (final_value - self.initial_capital) / self.initial_capital
        annual_return = total_return * 252 / max(len(df_eq), 1)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

        # 最大回撤
        cummax = df_eq['value'].cummax()
        drawdown = (df_eq['value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 统计
        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell']
        winners = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losers = [t for t in sell_trades if t.get('pnl', 0) <= 0]

        result = {
            'total_return': total_return,
            'final_value': final_value,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': len(winners) / len(sell_trades) if sell_trades else 0,
            'equity_curve': equity_curve,
            'trades': trades
        }

        print(f"\n{'='*60}")
        print(f"回测结果")
        print(f"{'='*60}")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"夏普比率: {sharpe:.3f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"总交易次数: {len(trades)}")
        print(f"胜率: {result['win_rate']:.1%}")
        print(f"最终净值: {final_value:,.2f}")
        print(f"{'='*60}")

        return result


def main():
    """测试滚动训练"""
    from src.data.data_pipeline import DataPipeline

    print("加载数据...")
    pipeline = DataPipeline()
    stock_data = {}
    codes = pipeline.get_stock_pool()[:20]  # 使用20只股票

    for code in codes:
        df = pipeline.load(code)
        if df is not None and len(df) > 200:
            stock_data[code] = df

    print(f"加载了 {len(stock_data)} 只股票")

    # 滚动评估
    config = RollingConfig(
        train_window=120,
        test_window=60,
        step_size=20,
        prediction_horizon=5
    )

    trainer = RollingTrainer(config)
    results = trainer.rolling_train_and_evaluate(
        stock_data,
        start_date=pd.Timestamp('2026-01-01'),
        end_date=pd.Timestamp('2026-07-01')
    )

    # 滚动回测
    print("\n运行滚动回测...")
    backtester = RollingBacktester(
        initial_capital=1000000,
        rolling_config=config
    )
    bt_result = backtester.run(
        stock_data,
        start_date='2026-01-01',
        end_date='2026-07-01',
        top_n=5,
        buy_threshold=0.005
    )


if __name__ == '__main__':
    main()