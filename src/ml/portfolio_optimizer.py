#!/usr/bin/env python3
"""
Mean-Variance组合优化器
=======================

使用cvxpy实现Markowitz Mean-Variance Optimization

功能:
1. 计算个股/行业协方差矩阵
2. Mean-Variance最优权重配置
3. 约束条件:
   - 单只股票仓位上限
   - 行业暴露上限
   - 目标收益或目标风险
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import os

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("警告: cvxpy未安装,Mean-Variance优化不可用")

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'


class CovarianceEstimator:
    """
    协方差矩阵估计器

    使用多种方法估计协方差:
    1. 样本协方差
    2. Shrinkage估计 (Ledoit-Wolf)
    3. 行业矩阵 (简化版)
    """

    def __init__(self, method: str = 'shrinkage'):
        """
        Args:
            method: 'sample', 'shrinkage', 或 'industry'
        """
        self.method = method

    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        估计协方差矩阵

        Args:
            returns: DataFrame of returns (dates x stocks)

        Returns:
            covariance matrix (stocks x stocks)
        """
        if self.method == 'sample':
            return self._sample_cov(returns)
        elif self.method == 'shrinkage':
            return self._ledoit_wolf(returns)
        elif self.method == 'industry':
            return self._industry_cov(returns)
        else:
            return self._sample_cov(returns)

    def _sample_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """样本协方差"""
        return np.cov(returns.T, ddof=1)

    def _ledoit_wolf(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Ledoit-Wolf Shrinkage估计

        将样本协方差向单位矩阵收缩,减少过拟合
        """
        n, p = returns.shape
        sample_cov = np.cov(returns.T, ddof=1)

        # 收缩目标(单位矩阵 * 样本方差均值)
        target = np.eye(p) * np.trace(sample_cov) / p

        # 收缩强度(简化版)
        delta = 1.0 / (n + p)

        cov = (1 - delta) * sample_cov + delta * target
        return cov

    def _industry_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """
        行业协方差矩阵

        假设同一行业股票收益相关,不同行业不相关
        """
        n, p = returns.shape
        sample_cov = np.cov(returns.T, ddof=1)

        # 简化:在样本协方差基础上添加行业结构
        # 对角线元素保持,非对角线元素根据行业相似性调整
        # 这里使用简化版本,直接返回样本协方差
        return sample_cov


class MeanVarianceOptimizer:
    """
    Mean-Variance组合优化器

    求解:
    max w'μ - γ * w'Σw
    s.t. sum(w) = 1
         w >= 0
         w <= max_weight per stock
         sector constraints
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.15,
                 min_weight: float = 0.0,
                 sector_max_weight: float = 0.30):
        """
        Args:
            risk_aversion: 风险厌恶系数 (γ)
            max_weight: 单只股票最大权重
            min_weight: 单只股票最小权重
            sector_max_weight: 单行业最大权重
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.sector_max_weight = sector_max_weight

        self.cov_estimator = CovarianceEstimator(method='shrinkage')
        self.last_weights = None

    def optimize(self,
                expected_returns: np.ndarray,
                cov_matrix: np.ndarray,
                stock_codes: List[str],
                sector_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        运行Mean-Variance优化

        Args:
            expected_returns: 预期收益数组 (n_stocks,)
            cov_matrix: 协方差矩阵 (n_stocks x n_stocks)
            stock_codes: 股票代码列表
            sector_map: {stock_code: sector_name} 映射

        Returns:
            {stock_code: weight}
        """
        if not CVXPY_AVAILABLE:
            return self._equal_weight(stock_codes)

        n = len(stock_codes)
        if n == 0:
            return {}

        if n == 1:
            return {stock_codes[0]: 1.0}

        # 检查有效性
        if np.any(np.isnan(expected_returns)) or np.any(np.isnan(cov_matrix)):
            print("  [MVO] 警告: 输入包含NaN,使用等权")
            return self._equal_weight(stock_codes)

        # 转换为numpy数组
        mu = np.array(expected_returns).flatten()
        Sigma = np.array(cov_matrix)

        # 确保矩阵是对称的
        Sigma = (Sigma + Sigma.T) / 2

        # 检查矩阵有效性
        try:
            np.linalg.cholesky(Sigma + 1e-6 * np.eye(n))
        except np.linalg.LinAlgError:
            print("  [MVO] 警告: 协方差矩阵不正定,添加正则化")
            Sigma = Sigma + 1e-6 * np.eye(n)

        # CVXPY优化
        w = cp.Variable(n)
        gamma = self.risk_aversion

        # 目标函数: 预期收益 - gamma * 方差
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)

        # 约束条件
        constraints = [
            cp.sum(w) == 1,           # 权重和=1
            w >= self.min_weight,     # 最小权重
            w <= self.max_weight      # 最大权重
        ]

        # 行业约束
        if sector_map:
            sectors = set(sector_map.values())
            for sector in sectors:
                sector_stocks = [i for i, code in enumerate(stock_codes)
                                if sector_map.get(code) == sector]
                if sector_stocks:
                    constraints.append(
                        cp.sum(w[sector_stocks]) <= self.sector_max_weight
                    )

        # 求解
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS, verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights = np.array(w.value).flatten()
                weights = np.maximum(weights, 0)  # 确保非负
                weights = weights / weights.sum()  # 归一化
                self.last_weights = weights
            else:
                print(f"  [MVO] 优化状态: {problem.status}, 使用等权")
                return self._equal_weight(stock_codes)
        except Exception as e:
            print(f"  [MVO] 求解失败: {e}, 使用等权")
            return self._equal_weight(stock_codes)

        # 返回结果
        result = {}
        for i, code in enumerate(stock_codes):
            if weights[i] > 0.001:  # 过滤太小的权重
                result[code] = float(weights[i])

        return result

    def _equal_weight(self, stock_codes: List[str]) -> Dict[str, float]:
        """等权分配"""
        n = len(stock_codes)
        if n == 0:
            return {}
        w = 1.0 / n
        return {code: w for code in stock_codes}


class MVOPortfolio:
    """
    Mean-Variance优化组合管理器

    用于在回测中动态调整仓位
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.15,
                 sector_max_weight: float = 0.30,
                 lookback_days: int = 60):
        """
        Args:
            risk_aversion: 风险厌恶系数
            max_weight: 单只股票最大权重
            sector_max_weight: 单行业最大权重
            lookback_days: 计算协方差的数据窗口
        """
        self.optimizer = MeanVarianceOptimizer(
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            sector_max_weight=sector_max_weight
        )
        self.lookback_days = lookback_days
        self.sector_map = {}

    def load_sector_map(self):
        """加载行业映射"""
        try:
            pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
            if os.path.exists(pool_path):
                with open(pool_path, 'r') as f:
                    pool = json.load(f)
                    self.sector_map = pool.get('industries', {})
        except Exception as e:
            print(f"  [MVO] 行业映射加载失败: {e}")

    def calculate_weights(self,
                        data_dict: Dict[str, pd.DataFrame],
                        scores: Dict[str, float]) -> Dict[str, float]:
        """
        根据数据和评分计算优化权重

        Args:
            data_dict: {code: DataFrame} 历史数据
            scores: {code: signal_score} 评分

        Returns:
            {code: weight} 优化后的权重
        """
        self.load_sector_map()

        if not data_dict:
            return {}

        # 收集股票代码(按评分排序)
        sorted_codes = sorted(scores.keys(), key=lambda x: scores.get(x, 0), reverse=True)

        # 取评分最高的前N只股票
        max_stocks = 10  # 最多10只
        selected_codes = sorted_codes[:max_stocks]

        if len(selected_codes) <= 1:
            return {selected_codes[0]: 1.0} if selected_codes else {}

        # 计算收益率矩阵
        returns_dict = {}
        for code in selected_codes:
            df = data_dict.get(code)
            if df is not None and len(df) >= self.lookback_days:
                prices = df['close'].values[-self.lookback_days:]
                rets = np.diff(prices) / prices[:-1]
                returns_dict[code] = rets

        # 再次过滤(确保有足够数据)
        selected_codes = [c for c in selected_codes if c in returns_dict]

        if len(selected_codes) < 2:
            return {selected_codes[0]: 1.0} if selected_codes else {}

        # 构建收益率DataFrame
        n = min(len(r) for r in returns_dict.values())
        returns_data = []
        for code in selected_codes:
            rets = returns_dict[code][-n:]
            returns_data.append(rets)
        returns_df = pd.DataFrame(returns_data, index=selected_codes).T

        # 计算预期收益(使用评分作为收益代理)
        expected_returns = np.array([scores.get(c, 50) / 100 for c in selected_codes])

        # 计算协方差矩阵
        cov_matrix = self.optimizer.cov_estimator.fit(returns_df)

        # 运行优化
        weights = self.optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            stock_codes=selected_codes,
            sector_map=self.sector_map
        )

        return weights


def test_mvo():
    """测试Mean-Variance优化器"""
    if not CVXPY_AVAILABLE:
        print("cvxpy未安装,跳过测试")
        return

    # 创建测试数据
    np.random.seed(42)
    n_stocks = 10
    n_days = 100

    # 模拟价格
    prices = np.cumprod(1 + np.random.randn(n_days, n_stocks) * 0.02, axis=0)
    returns = np.diff(prices, axis=0) / prices[:-1]

    # 模拟评分(越高越好)
    scores = 50 + np.random.randn(n_stocks) * 10

    stock_codes = [f'stock_{i}' for i in range(n_stocks)]

    # 创建优化器
    optimizer = MeanVarianceOptimizer(
        risk_aversion=1.0,
        max_weight=0.15,
        sector_max_weight=0.30
    )

    # 估计协方差
    cov_estimator = CovarianceEstimator(method='shrinkage')
    returns_df = pd.DataFrame(returns, columns=stock_codes)
    cov_matrix = cov_estimator.fit(returns_df)

    # 预期收益
    expected_returns = scores / 100

    # 优化
    weights = optimizer.optimize(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        stock_codes=stock_codes
    )

    print("\n" + "=" * 60)
    print("Mean-Variance优化结果")
    print("=" * 60)
    print(f"{'股票':<15} {'评分':>10} {'权重':>10}")
    print("-" * 40)
    for code, weight in sorted(weights.items(), key=lambda x: -x[1]):
        idx = stock_codes.index(code)
        print(f"{code:<15} {scores[idx]:>10.2f} {weight:>10.2%}")

    print(f"\n权重总和: {sum(weights.values()):.2%}")
    print(f"股票数量: {len(weights)}")


if __name__ == '__main__':
    test_mvo()