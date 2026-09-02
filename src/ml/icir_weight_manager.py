#!/usr/bin/env python3
"""
ICIR动态权重管理器
==================

基于滚动ICIR (Information Coefficient Information Ratio) 动态调整因子权重

ICIR = IC_mean / IC_std
- IC = Spearman相关系数(因子评分, 未来收益)
- ICIR > 0.5 表示因子稳定有效

权重调整策略:
- ICIR > 1.0: 高权重 (权重 × 1.5, 上限0.5)
- ICIR 0.5-1.0: 正常权重
- ICIR < 0.3: 低权重 (权重 × 0.5, 下限0.05)
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy.stats import spearmanr
import json
import os

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
MODEL_DIR = f'{DATA_DIR}/models'
CACHE_DIR = f'{DATA_DIR}/cache'


class ICIRRecord:
    """单条IC记录"""
    def __init__(self, date: str, factor_name: str, ic: float, ic_mean: float, ic_std: float, icir: float):
        self.date = date
        self.factor_name = factor_name
        self.ic = ic
        self.ic_mean = ic_mean
        self.ic_std = ic_std
        self.icir = icir

    def to_dict(self):
        return {
            'date': self.date,
            'factor_name': self.factor_name,
            'ic': self.ic,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'icir': self.icir
        }


class FactorICHistory:
    """单个因子的IC历史"""
    def __init__(self, name: str, window: int = 12):
        self.name = name
        self.window = window  # 滚动窗口大小
        self.ic_history = deque(maxlen=window)
        self.ic_records = []  # 所有历史记录

    def add_ic(self, date: str, ic: float) -> ICIRRecord:
        """添加新IC值,返回ICIR记录"""
        self.ic_history.append(ic)

        if len(self.ic_history) >= 3:
            ic_mean = np.mean(self.ic_history)
            ic_std = np.std(self.ic_history)
            icir = ic_mean / ic_std if ic_std > 0 else 0
        else:
            ic_mean = ic
            ic_std = 0
            icir = 0

        record = ICIRRecord(date, self.name, ic, ic_mean, ic_std, icir)
        self.ic_records.append(record)
        return record

    def get_icir(self) -> float:
        """获取当前ICIR"""
        if len(self.ic_history) < 3:
            return 0
        ic_mean = np.mean(self.ic_history)
        ic_std = np.std(self.ic_history)
        return ic_mean / ic_std if ic_std > 0 else 0

    def get_ic_mean(self) -> float:
        """获取IC均值"""
        if not self.ic_history:
            return 0
        return np.mean(self.ic_history)


class ICIRDynamicWeightManager:
    """
    ICIR动态权重管理器

    功能:
    1. 跟踪各因子IC值
    2. 计算滚动ICIR
    3. 动态调整因子权重
    """

    def __init__(self,
                 factors: List[str],
                 base_weights: Dict[str, float],
                 window: int = 12,
                 min_icir_threshold: float = 0.3,
                 max_weight: float = 0.5,
                 min_weight: float = 0.05):
        """
        Args:
            factors: 因子名称列表
            base_weights: 基础权重 (总和=1)
            window: 滚动窗口大小(周)
            min_icir_threshold: ICIR最低阈值,低于此值降低权重
            max_weight: 单因子最大权重
            min_weight: 单因子最小权重
        """
        self.factors = factors
        self.base_weights = base_weights.copy()
        self.window = window
        self.min_icir_threshold = min_icir_threshold
        self.max_weight = max_weight
        self.min_weight = min_weight

        # 每因子IC历史
        self.factor_histories: Dict[str, FactorICHistory] = {}
        for f in factors:
            self.factor_histories[f] = FactorICHistory(f, window)

        # IC缓存文件
        self.ic_cache_file = f'{CACHE_DIR}/icir_history.json'
        self._load_cache()

    def _load_cache(self):
        """加载IC缓存"""
        if os.path.exists(self.ic_cache_file):
            try:
                with open(self.ic_cache_file, 'r') as f:
                    data = json.load(f)
                    for factor_data in data.get('factor_histories', []):
                        name = factor_data['name']
                        if name in self.factor_histories:
                            for record in factor_data.get('records', []):
                                self.factor_histories[name].add_ic(
                                    record['date'], record['ic']
                                )
                print(f"  [ICIR] 已加载IC缓存")
            except Exception as e:
                print(f"  [ICIR] 缓存加载失败: {e}")

    def _save_cache(self):
        """保存IC缓存"""
        try:
            data = {
                'factor_histories': [
                    {
                        'name': name,
                        'records': [r.to_dict() for r in fh.ic_records]
                    }
                    for name, fh in self.factor_histories.items()
                ]
            }
            with open(self.ic_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [ICIR] 缓存保存失败: {e}")

    def calculate_ic(self, scores: Dict[str, float], future_returns: Dict[str, float]) -> Dict[str, float]:
        """
        计算各因子与未来收益的IC

        Args:
            scores: {code: factor_score}
            future_returns: {code: future_return}

        Returns:
            {factor_name: ic_value}
        """
        # 获取共同股票
        common_codes = set(scores.keys()) & set(future_returns.keys())
        if len(common_codes) < 10:
            return {f: 0 for f in self.factors}

        # 构建排名列表
        score_values = [scores[c] for c in common_codes]
        return_values = [future_returns[c] for c in common_codes]

        # 计算Spearman相关系数
        ic, _ = spearmanr(score_values, return_values)
        if np.isnan(ic):
            ic = 0

        return {f: ic for f in self.factors}

    def calculate_factor_ics(self,
                            factor_scores: Dict[str, Dict[str, float]],
                            future_returns: Dict[str, float]) -> Dict[str, float]:
        """
        计算多个因子的IC

        Args:
            factor_scores: {factor_name: {code: score}}
            future_returns: {code: future_return}

        Returns:
            {factor_name: ic_value}
        """
        ics = {}
        for factor_name, scores in factor_scores.items():
            common_codes = set(scores.keys()) & set(future_returns.keys())
            if len(common_codes) < 10:
                ics[factor_name] = 0
                continue

            score_values = [scores[c] for c in common_codes]
            return_values = [future_returns[c] for c in common_codes]

            ic, _ = spearmanr(score_values, return_values)
            ics[factor_name] = 0 if np.isnan(ic) else ic

        return ics

    def update_icir(self, date: str, ic_by_factor: Dict[str, float]) -> Dict[str, ICIRRecord]:
        """
        更新ICIR记录

        Args:
            date: 当前日期
            ic_by_factor: {因子名: IC值}

        Returns:
            {因子名: ICIRRecord}
        """
        records = {}
        for factor_name, ic in ic_by_factor.items():
            if factor_name in self.factor_histories:
                record = self.factor_histories[factor_name].add_ic(date, ic)
                records[factor_name] = record

        self._save_cache()
        return records

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        根据ICIR获取动态权重

        Returns:
            {因子名: 动态权重}
        """
        icirs = {f: self.factor_histories[f].get_icir() for f in self.factors}

        # 计算调整后权重
        raw_weights = {}
        for f in self.factors:
            icir = icirs[f]

            if icir > 1.0:
                # 高ICIR因子: 获得更高权重
                boost = 1.5
            elif icir < self.min_icir_threshold:
                # 低ICIR因子: 降低权重
                boost = 0.5
            else:
                boost = 1.0

            raw_weights[f] = self.base_weights[f] * boost

        # 归一化使总和=1
        total = sum(raw_weights.values())
        if total > 0:
            dynamic_weights = {f: raw_weights[f] / total for f in self.factors}
        else:
            dynamic_weights = self.base_weights.copy()

        # 限制单因子权重范围
        for f in self.factors:
            dynamic_weights[f] = max(self.min_weight,
                                     min(self.max_weight, dynamic_weights[f]))

        # 再次归一化
        total = sum(dynamic_weights.values())
        if total > 0:
            dynamic_weights = {f: dynamic_weights[f] / total for f in self.factors}

        return dynamic_weights

    def get_icir_report(self) -> Dict:
        """获取ICIR报告"""
        report = {
            'factors': {}
        }

        for f in self.factors:
            fh = self.factor_histories[f]
            report['factors'][f] = {
                'ic_mean': fh.get_ic_mean(),
                'icir': fh.get_icir(),
                'ic_count': len(fh.ic_history),
                'current_weight': self.base_weights.get(f, 0)
            }

        return report

    def print_icir_status(self):
        """打印ICIR状态"""
        print("\n" + "=" * 60)
        print("ICIR动态权重状态")
        print("=" * 60)

        icirs = {f: self.factor_histories[f].get_icir() for f in self.factors}
        dyn_weights = self.get_dynamic_weights()

        print(f"{'因子':<20} {'IC均值':>10} {'ICIR':>10} {'基础权重':>10} {'动态权重':>10} {'状态':>8}")
        print("-" * 70)

        for f in self.factors:
            ic_mean = self.factor_histories[f].get_ic_mean()
            icir = icirs[f]
            base_w = self.base_weights.get(f, 0)
            dyn_w = dyn_weights.get(f, 0)

            if icir > 1.0:
                status = "↑强"
            elif icir > 0.5:
                status = "✓中"
            elif icir > 0.3:
                status = "○弱"
            else:
                status = "✗无效"

            print(f"{f:<20} {ic_mean:>10.4f} {icir:>10.4f} {base_w:>10.2f} {dyn_w:>10.2f} {status:>8}")

        print("-" * 70)


def test_icir():
    """测试ICIR管理器"""
    # 测试数据
    factors = ['trend', 'mean_rev', 'momentum', 'new_factor']
    base_weights = {
        'trend': 0.20,
        'mean_rev': 0.30,
        'momentum': 0.25,
        'new_factor': 0.25
    }

    manager = ICIRDynamicWeightManager(factors, base_weights)

    # 模拟数据
    np.random.seed(42)
    codes = [f'code_{i}' for i in range(50)]

    # 模拟IC计算
    for week in range(15):
        date = f'2026-07-{week+1:02d}'
        ic_by_factor = {
            'trend': np.random.randn() * 0.1 + 0.2,
            'mean_rev': np.random.randn() * 0.1 + 0.4,
            'momentum': np.random.randn() * 0.1 + 0.1,
            'new_factor': np.random.randn() * 0.1 + 0.15
        }
        manager.update_icir(date, ic_by_factor)

    # 打印状态
    manager.print_icir_status()

    # 获取动态权重
    dyn_weights = manager.get_dynamic_weights()
    print("\n动态权重:")
    for f, w in dyn_weights.items():
        print(f"  {f}: {w:.2f}")


if __name__ == '__main__':
    test_icir()