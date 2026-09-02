#!/usr/bin/env python3
"""
Auto-Research 自主研究循环
核心模式:
1. LLM 提出策略参数修改建议
2. 运行回测
3. 评估结果
4. 决定保留或丢弃
5. 重复迭代
"""

import json
import copy
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any

# 项目路径
PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
import sys
sys.path.insert(0, PROJECT_ROOT)

from src.backtest.engine import BacktestEngine
from src.data.data_pipeline import DataPipeline
from src.evaluation.scorer import StrategyEvaluator, EvaluationMetrics
from src.ai_service import AIService

# 实验日志
EXPERIMENT_LOG = f'{PROJECT_ROOT}/data/cache/research_log.json'
BEST_PARAMS_FILE = f'{PROJECT_ROOT}/data/cache/best_params.json'

@dataclass
class ExperimentResult:
    """单次实验结果"""
    experiment_id: int
    timestamp: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    score: float
    improved: bool
    notes: str

@dataclass
class ResearchState:
    """研究状态"""
    current_round: int = 0
    max_rounds: int = 20
    best_score: float = 0
    best_params: Dict = field(default_factory=dict)
    best_metrics: Dict = field(default_factory=dict)
    experiments: List[ExperimentResult] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    def save(self):
        os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
        # 确保所有值都是原生Python类型
        experiments_data = []
        for e in self.experiments:
            experiments_data.append({
                'experiment_id': int(e.experiment_id),
                'timestamp': str(e.timestamp),
                'params': {k: v for k, v in e.params.items()},
                'metrics': {k: float(v) for k, v in e.metrics.items()},
                'score': float(e.score),
                'improved': bool(e.improved),
                'notes': str(e.notes)
            })
        data = {
            'current_round': int(self.current_round),
            'best_score': float(self.best_score),
            'best_params': {k: v for k, v in self.best_params.items()},
            'best_metrics': {k: float(v) for k, v in self.best_metrics.items()},
            'experiments': experiments_data,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date)
        }
        with open(EXPERIMENT_LOG, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> 'ResearchState':
        if os.path.exists(EXPERIMENT_LOG):
            with open(EXPERIMENT_LOG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                state = cls()
                state.current_round = data.get('current_round', 0)
                state.best_score = data.get('best_score', 0)
                state.best_params = data.get('best_params', {})
                state.best_metrics = data.get('best_metrics', {})
                state.start_date = data.get('start_date', '')
                state.end_date = data.get('end_date', '')
                experiments = data.get('experiments', [])
                state.experiments = [ExperimentResult(**e) for e in experiments]
                return state
        return cls()


class ParamGenerator:
    """参数生成器 - 基于规则生成参数"""

    # 参数范围
    PARAM_RANGES = {
        'min_score': (40, 80),        # 入场评分阈值
        'rebalance_days': (3, 20),    # 调仓周期
        'max_positions': (3, 8),       # 最大持仓数
        'stop_loss_pct': (0.03, 0.15),  # 止损
        'take_profit_pct': (0.10, 0.30), # 止盈
    }

    @classmethod
    def generate_variation(cls, current_params: Dict, strategy: str = 'explore') -> Dict:
        """
        基于当前参数生成变体

        Args:
            current_params: 当前参数
            strategy: 'explore' (探索新区域) 或 'refine' (精细调整)

        Returns:
            新的参数组合
        """
        import random

        new_params = copy.deepcopy(current_params)

        if strategy == 'explore':
            # 随机选择1-2个参数大幅变化
            params_to_change = random.sample(list(cls.PARAM_RANGES.keys()),
                                            min(random.randint(1, 2),
                                                len(cls.PARAM_RANGES)))
            for param in params_to_change:
                low, high = cls.PARAM_RANGES[param]
                if param in ['min_score', 'max_positions', 'rebalance_days']:
                    new_params[param] = random.randint(low, high)
                else:
                    new_params[param] = round(random.uniform(low, high), 2)

        else:  # refine
            # 小幅调整，选择1个参数微调
            param = random.choice(list(cls.PARAM_RANGES.keys()))
            low, high = cls.PARAM_RANGES[param]
            current_val = current_params.get(param, (low + high) / 2)

            # 在当前值附近小幅调整
            if param in ['min_score', 'max_positions', 'rebalance_days']:
                delta = random.randint(2, 5)
                direction = random.choice([-1, 1])
                new_val = int(current_val + delta * direction)
                new_params[param] = max(low, min(high, new_val))
            else:
                delta = random.uniform(0.01, 0.03)
                direction = random.choice([-1, 1])
                new_val = current_val + delta * direction * (high - low)
                new_params[param] = round(max(low, min(high, new_val)), 2)

        return new_params

    @classmethod
    def get_default_params(cls) -> Dict:
        """获取默认参数"""
        return {
            'min_score': 55,
            'rebalance_days': 5,
            'max_positions': 5,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.15,
        }


class AutoResearchLoop:
    """自主研究循环"""

    def __init__(self, use_ai: bool = True):
        self.state = ResearchState.load()
        self.pipeline = DataPipeline()
        self.evaluator = StrategyEvaluator()
        self.backtester = BacktestEngine(initial_capital=1000000)
        self.ai_service = AIService() if use_ai else None
        self.use_ai = use_ai

    def _load_stock_data(self) -> Dict:
        """加载股票数据"""
        stock_data = {}
        codes = self.pipeline.get_stock_pool()

        for code in codes:
            df = self.pipeline.load_from_csv(code)
            if df is not None:
                stock_data[code] = df

        return stock_data

    def _run_backtest(self, params: Dict, stock_data: Dict,
                     start_date: str, end_date: str) -> Optional[Dict]:
        """运行回测"""
        try:
            result = self.backtester.run(
                stock_data,
                start_date=start_date,
                end_date=end_date,
                rebalance_days=params.get('rebalance_days', 5),
                min_score=params.get('min_score', 55)
            )
            return result
        except Exception as e:
            print(f"回测失败: {e}")
            return None

    def _evaluate(self, result) -> tuple[EvaluationMetrics, float]:
        """评估回测结果"""
        metrics = self.evaluator.evaluate(result)
        score = self.evaluator.calculate_composite_score(metrics)
        return metrics, score

    def run_round(self, round_num: int, params: Dict,
                  stock_data: Dict, start_date: str, end_date: str) -> ExperimentResult:
        """运行单轮实验"""
        print(f"\n{'='*50}")
        print(f"第 {round_num} 轮")
        print(f"参数: {params}")
        print(f"{'='*50}")

        # 运行回测
        result = self._run_backtest(params, stock_data, start_date, end_date)

        if result is None:
            return ExperimentResult(
                experiment_id=round_num,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                params=params,
                metrics={},
                score=0,
                improved=False,
                notes="回测失败"
            )

        # 评估
        metrics, score = self._evaluate(result)

        # 打印结果
        print(f"\n回测结果:")
        print(f"  年化收益: {metrics.annual_return:.2f}%")
        print(f"  夏普比率: {metrics.sharpe_ratio:.3f}")
        print(f"  最大回撤: {metrics.max_drawdown:.2f}%")
        print(f"  胜率: {metrics.win_rate:.1f}%")
        print(f"  交易次数: {metrics.trade_count}")
        print(f"  综合评分: {score:.1f}")

        # 判断是否改进
        improved = score > self.state.best_score

        metrics_dict = {
            'total_return': metrics.total_return,
            'annual_return': metrics.annual_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'trade_count': metrics.trade_count,
        }

        return ExperimentResult(
            experiment_id=round_num,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            params=params,
            metrics=metrics_dict,
            score=score,
            improved=improved,
            notes=""
        )

    def run(self, start_date: str = '2025-01-01', end_date: str = '2025-06-30',
            max_rounds: int = 20):
        """
        运行自主研究循环
        """
        # 设置时间范围
        self.state.start_date = start_date
        self.state.end_date = end_date

        # 加载数据
        print("加载股票数据...")
        stock_data = self._load_stock_data()
        print(f"已加载 {len(stock_data)} 只股票\n")

        # 初始化参数
        if not self.state.best_params:
            self.state.best_params = ParamGenerator.get_default_params()

        print(f"{'#'*60}")
        print(f"# 自主策略研究开始")
        print(f"# 回测区间: {start_date} ~ {end_date}")
        print(f"# 最大迭代轮数: {max_rounds}")
        print(f"{'#'*60}\n")

        # 开始迭代
        for round_num in range(self.state.current_round + 1, max_rounds + 1):
            # 选择策略
            if round_num <= 3:
                strategy = 'explore'  # 前几轮探索
            elif self.state.experiments and not self.state.experiments[-1].improved:
                strategy = 'refine'  # 改进失败，精细调整
            else:
                strategy = 'explore' if round_num % 3 == 0 else 'refine'

            # 生成参数
            if round_num == 1:
                new_params = self.state.best_params
            else:
                # 尝试使用 AI 生成参数
                if self.use_ai and self.ai_service and self.state.experiments:
                    print("\n[AI] 正在分析回测结果...")
                    last_result = self.state.experiments[-1].metrics
                    ai_suggestion = self.ai_service.analyze_strategy(
                        last_result,
                        self.state.best_params
                    )
                    param_changes = ai_suggestion.get('param_changes', {})
                    if param_changes:
                        print(f"[AI] 建议修改: {param_changes}")
                        new_params = copy.deepcopy(self.state.best_params)
                        new_params.update(param_changes)
                    else:
                        print(f"[AI] 建议: {ai_suggestion.get('reason', '保持当前')}")
                        new_params = ParamGenerator.generate_variation(
                            self.state.best_params,
                            strategy=strategy
                        )
                else:
                    new_params = ParamGenerator.generate_variation(
                        self.state.best_params,
                        strategy=strategy
                    )

            # 运行实验
            result = self.run_round(round_num, new_params, stock_data, start_date, end_date)
            self.state.experiments.append(result)

            # 更新最优
            if result.improved:
                print(f"\n✓ 改进! 评分从 {self.state.best_score:.1f} 提高到 {result.score:.1f}")
                self.state.best_score = result.score
                self.state.best_params = new_params
                self.state.best_metrics = result.metrics
            else:
                print(f"\n✗ 未改进，保持原参数 (评分 {result.score:.1f})")

            self.state.current_round = round_num
            self.state.save()

            # 提前终止条件
            recent = self.state.experiments[-3:]
            if round_num > 5 and all(not e.improved for e in recent):
                print("\n连续 3 轮无改进，提前终止")
                break

        # 打印最终结果
        self.print_final_results()

        return self.state.best_params, self.state.best_metrics

    def print_final_results(self):
        """打印最终结果"""
        print(f"\n{'#'*60}")
        print(f"# 研究完成")
        print(f"# 总实验次数: {len(self.state.experiments)}")
        print(f"# 最佳评分: {self.state.best_score:.1f}")
        print(f"# 最佳参数:")
        for k, v in self.state.best_params.items():
            print(f"#   {k}: {v}")
        print(f"{'#'*60}")

        if self.state.best_metrics:
            m = self.state.best_metrics
            print(f"\n最佳回测结果:")
            print(f"  年化收益: {m.get('annual_return', 0):.2f}%")
            print(f"  夏普比率: {m.get('sharpe_ratio', 0):.3f}")
            print(f"  最大回撤: {m.get('max_drawdown', 0):.2f}%")
            print(f"  胜率: {m.get('win_rate', 0):.1f}%")

    def print_experiment_history(self, limit: int = 10):
        """打印实验历史"""
        print(f"\n{'='*60}")
        print(f"实验历史 (最近 {limit} 轮)")
        print(f"{'='*60}")
        print(f"{'轮次':<6} {'评分':<8} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'改进'}")
        print(f"{'-'*60}")

        for exp in self.state.experiments[-limit:]:
            m = exp.metrics
            print(f"{exp.experiment_id:<6} {exp.score:<8.1f} "
                  f"{m.get('annual_return', 0):<10.2f} "
                  f"{m.get('sharpe_ratio', 0):<10.3f} "
                  f"{m.get('max_drawdown', 0):<10.2f} "
                  f"{'✓' if exp.improved else '✗'}")


def main():
    """运行自主研究"""
    research = AutoResearchLoop()

    # 运行研究循环
    best_params, best_metrics = research.run(
        start_date='2025-01-01',
        end_date='2025-06-30',
        max_rounds=15
    )

    # 打印历史
    research.print_experiment_history()

    print("\n最终最佳参数:")
    print(json.dumps(best_params, indent=2))


if __name__ == '__main__':
    main()
