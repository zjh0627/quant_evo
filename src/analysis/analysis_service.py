#!/usr/bin/env python3
"""
历史模式匹配服务
整合走势匹配和事件匹配，提供统一的分析接口
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.analysis.pattern_matcher import PatternMatcher, PatternAnalysis, get_pattern_matcher
from src.analysis.event_matcher import EventMatcher, EventAnalysis, get_event_matcher
from src.analysis.event_db import get_event_database


@dataclass
class CombinedAnalysis:
    """综合分析结果"""
    stock_code: str
    analysis_time: str

    # 走势分析
    pattern_analysis: PatternAnalysis

    # 事件分析
    event_analysis: EventAnalysis

    # 综合建议
    recommendation: str
    confidence: str
    reasoning: List[str]

    # 输出格式
    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'analysis_time': self.analysis_time,
            'pattern': self.pattern_analysis.to_dict() if self.pattern_analysis else {},
            'event': self.event_analysis.to_dict() if self.event_analysis else {},
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }

    def to_summary(self) -> str:
        """生成简洁摘要"""
        lines = [
            f"【{self.stock_code} 综合分析】",
            f"分析时间: {self.analysis_time}",
            f"",
            f"--- 走势模式分析 ---",
            f"相似匹配: {len(self.pattern_analysis.matches) if self.pattern_analysis else 0} 个",
            f"平均相似度: {self.pattern_analysis.avg_similarity:.1%}" if self.pattern_analysis else "",
            f"历史后续走势: 5日 {self.pattern_analysis.avg_future_5d:+.2f}%, 20日 {self.pattern_analysis.avg_future_20d:+.2f}%" if self.pattern_analysis else "",
            f"",
            f"--- 事件匹配分析 ---",
            f"相似事件: {len(self.event_analysis.matches) if self.event_analysis else 0} 个",
            f"预测一周: {self.event_analysis.avg_predicted_1w:+.2f}%" if self.event_analysis else "",
            f"预测一月: {self.event_analysis.avg_predicted_1m:+.2f}%" if self.event_analysis else "",
            f"",
            f"--- 综合建议 ---",
            f"建议: {self.recommendation}",
            f"置信度: {self.confidence}",
            f"理由: {'; '.join(self.reasoning[:3])}",
        ]
        return '\n'.join(lines)


class PatternMatchService:
    """历史模式匹配服务"""

    def __init__(self):
        self.pattern_matcher: Optional[PatternMatcher] = None
        self.event_matcher: Optional[EventMatcher] = None

    def _ensure_pattern_matcher(self):
        if self.pattern_matcher is None:
            self.pattern_matcher = get_pattern_matcher()
        return self.pattern_matcher

    def _ensure_event_matcher(self):
        if self.event_matcher is None:
            self.event_matcher = get_event_matcher()
        return self.event_matcher

    def analyze_stock(self,
                     stock_code: str,
                     pattern_days: int = 20) -> CombinedAnalysis:
        """
        综合分析股票

        Args:
            stock_code: 股票代码，如 'sh.600519'
            pattern_days: 走势匹配天数

        Returns:
            CombinedAnalysis
        """
        # 走势分析
        matcher = self._ensure_pattern_matcher()
        pattern_analysis = matcher.find_similar_patterns(
            stock_code,
            pattern_days=pattern_days,
            top_n=5
        )

        # 事件分析（基于最近新闻自动分析）
        event_analysis = None
        # 这里可以接入实时新闻分析

        # 综合建议
        recommendation, confidence, reasoning = self._combine_recommendations(
            pattern_analysis, event_analysis
        )

        return CombinedAnalysis(
            stock_code=stock_code,
            analysis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            pattern_analysis=pattern_analysis,
            event_analysis=event_analysis,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning
        )

    def analyze_event(self,
                     event_type: str,
                     title: str,
                     sentiment: float,
                     intensity: str = 'medium',
                     markets: List[str] = None,
                     sectors: List[str] = None) -> EventAnalysis:
        """分析特定事件"""
        matcher = self._ensure_event_matcher()
        return matcher.analyze_event(
            event_type=event_type,
            title=title,
            sentiment=sentiment,
            intensity=intensity,
            affected_markets=markets,
            affected_sectors=sectors
        )

    def analyze_news(self, news_title: str, news_content: str = '') -> EventAnalysis:
        """分析新闻事件（自动识别）"""
        matcher = self._ensure_event_matcher()
        return matcher.analyze_news_event(news_title, news_content)

    def batch_analyze_stocks(self,
                            stock_codes: List[str],
                            pattern_days: int = 20) -> List[CombinedAnalysis]:
        """批量分析多只股票"""
        results = []
        for code in stock_codes:
            try:
                analysis = self.analyze_stock(code, pattern_days)
                results.append(analysis)
            except Exception as e:
                print(f"[PatternMatchService] 分析 {code} 失败: {e}")
        return results

    def get_top_opportunities(self,
                             stock_codes: List[str],
                             pattern_days: int = 20,
                             top_n: int = 5) -> List[CombinedAnalysis]:
        """获取最佳机会股票（按预测收益排序）"""
        analyses = self.batch_analyze_stocks(stock_codes, pattern_days)

        # 过滤掉无法分析的
        valid = [a for a in analyses if a.pattern_analysis and a.pattern_analysis.matches]

        # 按预测收益排序
        valid.sort(key=lambda a: a.pattern_analysis.avg_future_20d, reverse=True)

        return valid[:top_n]

    def _combine_recommendations(self,
                                pattern: PatternAnalysis,
                                event: EventAnalysis) -> Tuple[str, str, List[str]]:
        """综合走势和事件分析生成建议"""
        reasoning = []

        # 基于走势的建议
        if pattern and pattern.matches:
            if pattern.recommendation == '买入':
                reasoning.append(f'走势形态相似历史，历史上后续上涨')
            elif pattern.recommendation == '卖出':
                reasoning.append(f'走势形态与历史顶部相似')

        # 基于事件的建议
        if event and event.matches:
            if event.recommendation in ['强烈买入', '买入']:
                reasoning.append(f'当前事件与历史{event.matches[0].historical_event.title}相似')
            elif event.recommendation in ['强烈卖出', '卖出']:
                reasoning.append(f'类似事件历史上导致下跌')

        # 综合判断
        buy_signals = 0
        sell_signals = 0

        if pattern:
            if pattern.avg_future_20d > 5:
                buy_signals += 1
            elif pattern.avg_future_20d < -5:
                sell_signals += 1

        if event:
            if event.avg_predicted_1m > 5:
                buy_signals += 1
            elif event.avg_predicted_1m < -5:
                sell_signals += 1

        if buy_signals > sell_signals:
            recommendation = '买入'
            confidence = 'high' if buy_signals >= 2 else 'medium'
        elif sell_signals > buy_signals:
            recommendation = '卖出'
            confidence = 'high' if sell_signals >= 2 else 'medium'
        else:
            recommendation = '观望'
            confidence = 'medium'

        return recommendation, confidence, reasoning

    def get_event_statistics(self) -> Dict:
        """获取事件统计"""
        db = get_event_database()
        return db.get_statistics_by_type()


# 全局单例
_service: Optional[PatternMatchService] = None


def get_pattern_match_service() -> PatternMatchService:
    global _service
    if _service is None:
        _service = PatternMatchService()
    return _service


if __name__ == '__main__':
    print("=" * 60)
    print("测试历史模式匹配服务")
    print("=" * 60)

    service = PatternMatchService()

    # 分析单只股票
    print("\n[1] 综合分析贵州茅台:")
    analysis = service.analyze_stock('sh.600519', pattern_days=20)
    print(analysis.to_summary())

    # 分析新闻事件
    print("\n[2] 分析新闻事件:")
    news_analysis = service.analyze_news('央行宣布降息10个基点，A股有望迎来反弹')
    print(f"  事件类型: {news_analysis.input_event_type}")
    print(f"  情绪: {news_analysis.input_sentiment:+.2f}")
    print(f"  预测一月: {news_analysis.avg_predicted_1m:+.2f}%")
    print(f"  建议: {news_analysis.recommendation}")

    # 批量分析
    print("\n[3] 批量分析股票池:")
    stock_pool = ['sh.600519', 'sh.600036', 'sz.300750', 'sh.688012']
    analyses = service.batch_analyze_stocks(stock_pool)

    print(f"\n  分析了 {len(analyses)} 只股票:")
    for a in analyses:
        if a.pattern_analysis and a.pattern_analysis.matches:
            print(f"  {a.stock_code}: {a.pattern_analysis.recommendation} "
                  f"(20日预测: {a.pattern_analysis.avg_future_20d:+.2f}%)")

    # 事件统计
    print("\n[4] 历史事件统计:")
    stats = service.get_event_statistics()
    for event_type, s in stats.items():
        print(f"  {event_type}: {s['count']}起, 一月平均{s['avg_next_month']:+.2f}%")
