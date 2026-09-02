#!/usr/bin/env python3
"""
走势相似度匹配引擎
基于DTW和相关性的走势模式匹配
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')


# ============================================================
# 匹配结果数据模型
# ============================================================

@dataclass
class PatternMatch:
    """走势匹配结果"""
    stock_code: str
    match_start_date: str
    match_end_date: str

    # 相似度
    similarity: float  # 0 ~ 1
    distance: float  # DTW距离

    # 匹配段的走势
    pattern_returns: List[float]  # 日收益率序列

    # 后续走势（匹配后的N天）
    future_1d: float  # 1天后涨跌幅
    future_5d: float  # 5天后涨跌幅
    future_20d: float  # 20天后涨跌幅

    # 统计
    avg_future_return: float  # 平均未来收益
    win_rate: float  # 胜率（正收益占比）

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PatternAnalysis:
    """模式分析结果"""
    current_code: str
    current_pattern: List[float]
    matches: List[PatternMatch]

    # 统计汇总
    avg_similarity: float
    avg_future_5d: float
    avg_future_20d: float

    # 建议
    recommendation: str  # 买入/卖出/观望
    confidence: str  # high / medium / low

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['matches'] = [m.to_dict() for m in self.matches]
        return d


# ============================================================
# 走势相似度匹配器
# ============================================================

class PatternMatcher:
    """走势相似度匹配引擎"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = '/Users/keira/project/claude/quant_evo/data/cache'

        self.data_dir = data_dir
        self.price_data: Dict[str, pd.DataFrame] = {}

    def load_stock_data(self, codes: List[str] = None):
        """加载股票数据"""
        if codes is None:
            # 加载所有
            files = os.listdir(self.data_dir)
            codes = [f.replace('kline_', '').replace('.csv', '').replace('_', '.')
                     for f in files if f.startswith('kline_') and f.endswith('.csv')]

        for code in codes:
            # 转换格式
            code_underscore = code.replace('.', '_')
            filepath = os.path.join(self.data_dir, f'kline_{code_underscore}.csv')

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, parse_dates=['date'])
                    self.price_data[code] = df
                except Exception as e:
                    print(f"[PatternMatcher] 加载 {code} 失败: {e}")

        print(f"[PatternMatcher] 加载 {len(self.price_data)} 只股票数据")

    def get_returns(self, code: str, days: int = 20) -> Optional[List[float]]:
        """获取最近N天的收益率序列"""
        if code not in self.price_data:
            return None

        df = self.price_data[code].sort_values('date')
        if len(df) < days:
            return None

        # 计算日收益率
        closes = df['close'].values[-days:]
        returns = np.diff(closes) / closes[:-1] * 100
        return returns.tolist()

    def get_future_returns(self, code: str, start_idx: int,
                          future_days: List[int] = [1, 5, 20]) -> Dict[str, float]:
        """获取未来收益"""
        if code not in self.price_data:
            return {}

        df = self.price_data[code].sort_values('date').reset_index(drop=True)
        if start_idx >= len(df) - max(future_days):
            return {}

        current_price = df.iloc[start_idx]['close']
        results = {}

        for days in future_days:
            if start_idx + days < len(df):
                future_price = df.iloc[start_idx + days]['close']
                results[f'future_{days}d'] = (future_price - current_price) / current_price * 100

        return results

    def calculate_correlation(self, seq1: List[float], seq2: List[float]) -> float:
        """计算皮尔逊相关系数"""
        if len(seq1) != len(seq2):
            return 0.0

        n = len(seq1)
        if n < 5:
            return 0.0

        mean1 = sum(seq1) / n
        mean2 = sum(seq2) / n

        numerator = sum((a - mean1) * (b - mean2) for a, b in zip(seq1, seq2))
        denom1 = sum((a - mean1) ** 2 for a in seq1) ** 0.5
        denom2 = sum((b - mean2) ** 2 for b in seq2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            return 0.0

        return numerator / (denom1 * denom2)

    def calculate_dtw_distance(self, seq1: List[float], seq2: List[float]) -> float:
        """计算DTW距离（简化版）"""
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return float('inf')

        # 构建距离矩阵
        dtw = np.zeros((n + 1, m + 1))
        dtw[:, 0] = np.inf
        dtw[0, :] = np.inf
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

        return dtw[n, m]

    def normalize_sequence(self, seq: List[float]) -> List[float]:
        """归一化序列（去除均值，标准差归一）"""
        arr = np.array(seq)
        if len(arr) == 0:
            return []
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return [0.0] * len(arr)
        return ((arr - mean) / std).tolist()

    def find_similar_patterns(self,
                             code: str,
                             pattern_days: int = 20,
                             top_n: int = 5,
                             min_history_days: int = 50) -> PatternAnalysis:
        """
        查找相似走势

        Args:
            code: 目标股票代码
            pattern_days: 用于匹配的走势天数
            top_n: 返回前N个最相似
            min_history_days: 最小历史数据要求

        Returns:
            PatternAnalysis
        """
        if code not in self.price_data:
            return PatternAnalysis(
                current_code=code,
                current_pattern=[],
                matches=[],
                avg_similarity=0,
                avg_future_5d=0,
                avg_future_20d=0,
                recommendation='无数据',
                confidence='low'
            )

        # 获取当前走势
        current_pattern = self.get_returns(code, pattern_days)
        if current_pattern is None:
            return PatternAnalysis(
                current_code=code,
                current_pattern=[],
                matches=[],
                avg_similarity=0,
                avg_future_5d=0,
                avg_future_20d=0,
                recommendation='数据不足',
                confidence='low'
            )

        # 归一化
        norm_pattern = self.normalize_sequence(current_pattern)

        df = self.price_data[code].sort_values('date').reset_index(drop=True)
        current_end_idx = len(df) - 1

        matches: List[PatternMatch] = []

        # 在历史中滑动窗口查找相似
        for start_idx in range(min_history_days, current_end_idx - pattern_days):
            # 获取历史段
            hist_segment = df.iloc[start_idx:start_idx + pattern_days]
            hist_returns = np.diff(hist_segment['close'].values) / hist_segment['close'].values[:-1] * 100

            if len(hist_returns) < pattern_days - 1:
                continue

            norm_hist = self.normalize_sequence(hist_returns.tolist())

            # 计算相似度
            corr = self.calculate_correlation(norm_pattern, norm_hist)
            dtw_dist = self.calculate_dtw_distance(norm_pattern, norm_hist)

            # DTW距离转相似度（距离越小越相似）
            similarity = 1.0 / (1.0 + dtw_dist / 100)

            # 综合得分（相关性权重更高）
            final_similarity = 0.7 * max(0, corr) + 0.3 * similarity

            if final_similarity < 0.3:  # 相似度阈值
                continue

            # 获取后续收益
            future = self.get_future_returns(code, start_idx + pattern_days - 1)

            match = PatternMatch(
                stock_code=code,
                match_start_date=hist_segment['date'].iloc[0].strftime('%Y-%m-%d'),
                match_end_date=hist_segment['date'].iloc[-1].strftime('%Y-%m-%d'),
                similarity=final_similarity,
                distance=dtw_dist,
                pattern_returns=hist_returns.tolist(),
                future_1d=future.get('future_1d', 0),
                future_5d=future.get('future_5d', 0),
                future_20d=future.get('future_20d', 0),
                avg_future_return=future.get('future_5d', 0),
                win_rate=1.0 if future.get('future_5d', 0) > 0 else 0.0
            )
            matches.append(match)

        # 排序并取top_n
        matches.sort(key=lambda x: x.similarity, reverse=True)
        matches = matches[:top_n]

        # 统计
        if matches:
            avg_sim = sum(m.similarity for m in matches) / len(matches)
            avg_5d = sum(m.future_5d for m in matches) / len(matches)
            avg_20d = sum(m.future_20d for m in matches) / len(matches)
        else:
            avg_sim = avg_5d = avg_20d = 0

        # 生成建议
        recommendation, confidence = self._generate_recommendation(avg_sim, avg_5d, avg_20d, matches)

        return PatternAnalysis(
            current_code=code,
            current_pattern=current_pattern,
            matches=matches,
            avg_similarity=avg_sim,
            avg_future_5d=avg_5d,
            avg_future_20d=avg_20d,
            recommendation=recommendation,
            confidence=confidence
        )

    def _generate_recommendation(self, avg_sim: float, avg_5d: float, avg_20d: float,
                                 matches: List[PatternMatch]) -> Tuple[str, str]:
        """生成建议"""
        if avg_sim < 0.5:
            return '观望', 'low'

        # 统计胜率
        positive_5d = sum(1 for m in matches if m.future_5d > 0) / len(matches) if matches else 0
        positive_20d = sum(1 for m in matches if m.future_20d > 0) / len(matches) if matches else 0

        if avg_20d > 5 and positive_20d > 0.6:
            return '买入', 'high'
        elif avg_20d > 2 and positive_20d > 0.5:
            return '买入', 'medium'
        elif avg_20d < -5 and positive_20d < 0.4:
            return '卖出', 'high'
        elif avg_20d < -2 and positive_20d < 0.5:
            return '卖出', 'medium'
        else:
            return '观望', 'medium'

    def compare_stocks(self, code1: str, code2: str, days: int = 20) -> float:
        """比较两只股票的走势相似度"""
        returns1 = self.get_returns(code1, days)
        returns2 = self.get_returns(code2, days)

        if returns1 is None or returns2 is None:
            return 0.0

        norm1 = self.normalize_sequence(returns1)
        norm2 = self.normalize_sequence(returns2)

        return max(0, self.calculate_correlation(norm1, norm2))


# 全局单例
_matcher: Optional[PatternMatcher] = None


def get_pattern_matcher() -> PatternMatcher:
    global _matcher
    if _matcher is None:
        _matcher = PatternMatcher()
        _matcher.load_stock_data()
    return _matcher


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("测试走势相似度匹配")
    print("=" * 60)

    matcher = PatternMatcher()
    matcher.load_stock_data()

    # 测试单只股票
    test_code = 'sh.600519'  # 贵州茅台
    print(f"\n[1] 分析 {test_code} 走势模式...")

    analysis = matcher.find_similar_patterns(test_code, pattern_days=20, top_n=5)

    print(f"\n  相似匹配数: {len(analysis.matches)}")
    print(f"  平均相似度: {analysis.avg_similarity:.2%}")
    print(f"  平均5日收益: {analysis.avg_future_5d:+.2f}%")
    print(f"  平均20日收益: {analysis.avg_future_20d:+.2f}%")
    print(f"  建议: {analysis.recommendation} (置信度: {analysis.confidence})")

    if analysis.matches:
        print(f"\n[2] 最相似的历史走势:")
        best = analysis.matches[0]
        print(f"  时间段: {best.match_start_date} ~ {best.match_end_date}")
        print(f"  相似度: {best.similarity:.2%}")
        print(f"  后续走势: 1日{best.future_1d:+.2f}%, 5日{best.future_5d:+.2f}%, 20日{best.future_20d:+.2f}%")

    # 测试股票对比
    print(f"\n[3] 对比两只股票走势:")
    sim = matcher.compare_stocks('sh.600519', 'sh.600036')
    print(f"  贵州茅台 vs 招商银行 相似度: {sim:.2%}")
