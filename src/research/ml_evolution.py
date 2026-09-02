#!/usr/bin/env python3
"""
ML策略进化循环
==============

通过超参数搜索和模型重训练来进化ML策略
"""

import os
import json
import copy
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.data.data_pipeline import DataPipeline
from src.ml.model_training import MLModelTrainer, ModelConfig
from src.ml.prediction_service import PredictionService
from src.strategies.ml_strategy import MLStrategy, MLStrategyGenerator
from src.backtest.light_backtest import LightBacktest

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')
EVOLUTION_LOG = os.path.join(CACHE_DIR, 'ml_evolution_log.json')
BEST_PARAMS_FILE = os.path.join(CACHE_DIR, 'best_ml_params.json')


@dataclass
class EvolutionConfig:
    """进化配置"""
    # 策略参数搜索空间
    top_n_range: Tuple[int, int] = (5, 20)
    buy_threshold_range: Tuple[float, float] = (0.005, 0.10)
    sell_threshold_range: Tuple[float, float] = (-0.10, -0.005)

    # 模型参数搜索空间
    prediction_horizon_range: Tuple[int, int, int] = (3, 5, 10)  # 改用tuple
    learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    max_depth_range: Tuple[int, int] = (3, 8)

    # 进化控制
    population_size: int = 10
    mutation_rate: float = 0.3
    elite_ratio: float = 0.2  # 保留最佳比例


@dataclass
class ExperimentResult:
    """单次实验结果"""
    experiment_id: int
    timestamp: str
    strategy_params: Dict
    model_params: Dict
    metrics: Dict
    score: float  # 综合评分
    improved: bool


class EvolutionState:
    """进化状态"""

    def __init__(self):
        self.current_round: int = 0
        self.best_score: float = 0
        self.best_params: Dict = {}
        self.best_metrics: Dict = {}
        self.experiments: List[ExperimentResult] = []
        self.population: List[Dict] = []

    def save(self):
        """保存状态"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        data = {
            'current_round': self.current_round,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'best_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in self.best_metrics.items()},
            'experiments': [
                {
                    'experiment_id': e.experiment_id,
                    'strategy_params': e.strategy_params,
                    'model_params': e.model_params,
                    'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                               for k, v in e.metrics.items()},
                    'score': float(e.score),
                    'improved': e.improved,
                }
                for e in self.experiments[-50:]  # 只保留最近50个
            ]
        }
        with open(EVOLUTION_LOG, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> 'EvolutionState':
        """加载状态"""
        if os.path.exists(EVOLUTION_LOG):
            with open(EVOLUTION_LOG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                state = cls()
                state.current_round = data.get('current_round', 0)
                state.best_score = data.get('best_score', 0)
                state.best_params = data.get('best_params', {})
                state.best_metrics = data.get('best_metrics', {})
                # 重构experiments
                for e in data.get('experiments', []):
                    state.experiments.append(ExperimentResult(
                        experiment_id=e['experiment_id'],
                        timestamp='',
                        strategy_params=e['strategy_params'],
                        model_params=e['model_params'],
                        metrics=e['metrics'],
                        score=e['score'],
                        improved=e['improved'],
                    ))
                return state
        return cls()


class MLEvolutionLoop:
    """ML策略进化循环"""

    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.state = EvolutionState()
        self.pipeline = DataPipeline()

    def _load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        import pandas as pd
        stock_data = {}
        codes = self.pipeline.get_stock_pool()

        for code in codes:
            df = self.pipeline.load(code)
            if df is not None and len(df) > 100:
                stock_data[code] = df

        print(f"加载了 {len(stock_data)} 只股票")
        return stock_data

    def _generate_strategy_params(self, base: Dict = None) -> Dict:
        """生成策略参数"""
        if base is None:
            # 随机生成
            return {
                'top_n': random.randint(*self.config.top_n_range),
                'buy_threshold': round(random.uniform(*self.config.buy_threshold_range), 3),
                'sell_threshold': round(random.uniform(*self.config.sell_threshold_range), 3),
            }
        else:
            # 变异
            params = copy.deepcopy(base)
            if random.random() < self.config.mutation_rate:
                param_to_mutate = random.choice(['top_n', 'buy_threshold', 'sell_threshold'])
                if param_to_mutate == 'top_n':
                    params['top_n'] = max(
                        self.config.top_n_range[0],
                        min(self.config.top_n_range[1],
                            params['top_n'] + random.randint(-3, 3))
                    )
                elif param_to_mutate == 'buy_threshold':
                    delta = random.uniform(-0.02, 0.02)
                    params['buy_threshold'] = round(
                        max(self.config.buy_threshold_range[0],
                            min(self.config.buy_threshold_range[1],
                                params['buy_threshold'] + delta)), 3)
                elif param_to_mutate == 'sell_threshold':
                    delta = random.uniform(-0.02, 0.02)
                    params['sell_threshold'] = round(
                        max(self.config.sell_threshold_range[0],
                            min(self.config.sell_threshold_range[1],
                                params['sell_threshold'] + delta)), 3)
            return params

    def _generate_model_params(self, base: Dict = None) -> Dict:
        """生成模型参数"""
        if base is None:
            return {
                'prediction_horizon': random.choice(list(self.config.prediction_horizon_range)),
                'learning_rate': round(random.uniform(*self.config.learning_rate_range), 2),
                'max_depth': random.randint(*self.config.max_depth_range),
            }
        else:
            params = copy.deepcopy(base)
            if random.random() < self.config.mutation_rate:
                param_to_mutate = random.choice(['prediction_horizon', 'learning_rate', 'max_depth'])
                if param_to_mutate == 'prediction_horizon':
                    params['prediction_horizon'] = random.choice(list(self.config.prediction_horizon_range))
                elif param_to_mutate == 'learning_rate':
                    delta = random.uniform(-0.05, 0.05)
                    params['learning_rate'] = round(
                        max(self.config.learning_rate_range[0],
                            min(self.config.learning_rate_range[1],
                                params['learning_rate'] + delta)), 2)
                elif param_to_mutate == 'max_depth':
                    params['max_depth'] = max(
                        self.config.max_depth_range[0],
                        min(self.config.max_depth_range[1],
                            params['max_depth'] + random.randint(-1, 1))
                    )
            return params

    def _run_experiment(self,
                       stock_data: Dict,
                       strategy_params: Dict,
                       model_params: Dict) -> Tuple[Dict, float]:
        """
        运行单次实验

        Returns:
            (metrics, composite_score)
        """
        try:
            # 1. 配置模型
            model_config = ModelConfig(
                prediction_horizon=model_params['prediction_horizon'],
                xgb_params={
                    'n_estimators': 100,
                    'max_depth': model_params['max_depth'],
                    'learning_rate': model_params['learning_rate'],
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                }
            )

            # 2. 训练模型
            trainer = MLModelTrainer(model_config)
            train_results = trainer.train_ensemble(stock_data)

            # 3. 预测
            pred_service = PredictionService()
            pred_service.load_models()
            predictions = pred_service.predict(stock_data)

            # 4. 生成信号
            strategy = MLStrategy(
                top_n=strategy_params['top_n'],
                buy_threshold=strategy_params['buy_threshold'],
                sell_threshold=strategy_params['sell_threshold']
            )
            signals = strategy.generate_signals(stock_data, predictions)

            # 5. 回测
            bt = LightBacktest(initial_capital=1000000)
            result = bt.run(
                stock_data,
                signals,
                start_date='2024-07-01',
                end_date='2026-07-01'
            )

            if result is None:
                return {}, 0

            # 6. 计算综合评分
            metrics = {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'xgb_ic': train_results.get('xgb_ic', 0),
                'lstm_ic': train_results.get('lstm_ic', 0),
            }

            # 综合评分: 收益30 + 夏普25 + 回撤20 + 胜率15 + 交易次数10
            score = 0
            score += min(max(result.annual_return * 10, 0), 30)
            score += min(max(result.sharpe_ratio * 5, 0), 25)
            score += min(max(-result.max_drawdown * 20, 0), 20) if result.max_drawdown < 0 else 0
            score += min(max(result.win_rate * 15, 0), 15)
            score += min(max(result.total_trades / 100 * 10, 0), 10)

            return metrics, score

        except Exception as e:
            print(f"实验失败: {e}")
            import traceback
            traceback.print_exc()
            return {}, 0

    def run_round(self, round_num: int,
                  strategy_params: Dict,
                  model_params: Dict,
                  stock_data: Dict) -> ExperimentResult:
        """运行单轮实验"""
        print(f"\n{'='*50}")
        print(f"第 {round_num} 轮")
        print(f"策略参数: top_n={strategy_params['top_n']}, "
              f"buy={strategy_params['buy_threshold']:.2%}, "
              f"sell={strategy_params['sell_threshold']:.2%}")
        print(f"模型参数: horizon={model_params['prediction_horizon']}, "
              f"lr={model_params['learning_rate']:.2f}, "
              f"depth={model_params['max_depth']}")
        print(f"{'='*50}")

        metrics, score = self._run_experiment(stock_data, strategy_params, model_params)

        improved = score > self.state.best_score

        print(f"\n回测结果:")
        print(f"  年化收益: {metrics.get('annual_return', 0):.2%}")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  胜率: {metrics.get('win_rate', 0):.1%}")
        print(f"  综合评分: {score:.1f}")
        print(f"  XGB IC: {metrics.get('xgb_ic', 0):.4f}")
        print(f"  LSTM IC: {metrics.get('lstm_ic', 0):.4f}")

        return ExperimentResult(
            experiment_id=round_num,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            strategy_params=strategy_params,
            model_params=model_params,
            metrics=metrics,
            score=score,
            improved=improved
        )

    def evolve(self, stock_data: Dict = None, max_rounds: int = 20) -> Dict:
        """
        运行进化搜索
        """
        # 加载数据
        if stock_data is None:
            print("加载股票数据...")
            stock_data = self._load_stock_data()

        print(f"\n{'#'*60}")
        print(f"# ML策略进化开始")
        print(f"# 最大迭代轮数: {max_rounds}")
        print(f"{'#'*60}\n")

        # 初始化参数
        if not self.state.best_params:
            self.state.best_params = {
                'strategy': self._generate_strategy_params(),
                'model': self._generate_model_params()
            }

        best_strategy = self.state.best_params.get('strategy', self._generate_strategy_params())
        best_model = self.state.best_params.get('model', self._generate_model_params())

        # 进化迭代
        for round_num in range(self.state.current_round + 1, max_rounds + 1):
            # 生成新参数
            if round_num == 1 or random.random() < 0.3:
                # 随机生成
                new_strategy = self._generate_strategy_params()
                new_model = self._generate_model_params()
            else:
                # 基于最佳参数变异
                new_strategy = self._generate_strategy_params(best_strategy)
                new_model = self._generate_model_params(best_model)

            # 运行实验
            result = self.run_round(round_num, new_strategy, new_model, stock_data)
            self.state.experiments.append(result)

            # 更新最佳
            if result.improved:
                print(f"\n✓ 改进! 评分从 {self.state.best_score:.1f} 提高到 {result.score:.1f}")
                self.state.best_score = result.score
                best_strategy = new_strategy
                best_model = new_model
                self.state.best_params = {
                    'strategy': best_strategy,
                    'model': best_model
                }
                self.state.best_metrics = result.metrics
            else:
                print(f"\n✗ 未改进，保持原参数 (评分 {result.score:.1f})")

            self.state.current_round = round_num
            self.state.save()

            # 早停
            if round_num > 5:
                recent_no_improve = sum(
                    1 for e in self.state.experiments[-5:]
                    if not e.improved
                )
                if recent_no_improve >= 5:
                    print("\n连续5轮无改进，提前终止")
                    break

        # 输出最终结果
        self._print_final_results()

        return self.state.best_params

    def _print_final_results(self):
        """打印最终结果"""
        print(f"\n{'#'*60}")
        print(f"# 进化完成")
        print(f"# 总实验次数: {len(self.state.experiments)}")
        print(f"# 最佳评分: {self.state.best_score:.1f}")
        print(f"# 最佳策略参数:")
        for k, v in self.state.best_params.get('strategy', {}).items():
            print(f"#   {k}: {v}")
        print(f"# 最佳模型参数:")
        for k, v in self.state.best_params.get('model', {}).items():
            print(f"#   {k}: {v}")
        print(f"{'#'*60}")

        if self.state.best_metrics:
            m = self.state.best_metrics
            print(f"\n最佳回测结果:")
            print(f"  年化收益: {m.get('annual_return', 0):.2%}")
            print(f"  夏普比率: {m.get('sharpe_ratio', 0):.3f}")
            print(f"  最大回撤: {m.get('max_drawdown', 0):.2%}")
            print(f"  胜率: {m.get('win_rate', 0):.1%}")

        # 保存最佳参数
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump({
                'best_params': self.state.best_params,
                'best_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                for k, v in self.state.best_metrics.items()},
                'best_score': float(self.state.best_score),
            }, f, indent=2)
        print(f"\n最佳参数已保存: {BEST_PARAMS_FILE}")


def main():
    """运行进化"""
    evolution = MLEvolutionLoop()
    best_params = evolution.evolve(max_rounds=10)
    print("\n最终最佳参数:")
    print(json.dumps(best_params, indent=2))


if __name__ == '__main__':
    main()
