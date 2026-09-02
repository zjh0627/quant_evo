#!/usr/bin/env python3
"""
事件相似度匹配引擎
基于当前事件查找历史相似事件，预测市场反应
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import re

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.analysis.event_db import EventDatabase, MarketEvent, get_event_database, EVENT_TYPES


@dataclass
class EventMatch:
    """事件匹配结果"""
    historical_event: MarketEvent
    similarity: float  # 0 ~ 1

    # 预测市场反应
    predicted_immediate: float  # 预测立即反应 %
    predicted_1d: float  # 预测次日 %
    predicted_1w: float  # 预测一周 %
    predicted_1m: float  # 预测一月 %

    # 统计
    confidence: str  # high / medium / low
    match_reasons: List[str]  # 匹配原因

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['historical_event'] = self.historical_event.to_dict()
        return d


@dataclass
class EventAnalysis:
    """事件分析结果"""
    # 输入事件
    input_event_type: str
    input_title: str
    input_sentiment: float
    input_intensity: str
    input_markets: List[str]
    input_sectors: List[str]

    # 匹配结果
    matches: List[EventMatch]

    # 统计汇总
    avg_predicted_1w: float
    avg_predicted_1m: float
    confidence: str

    # 建议
    recommendation: str
    reasoning: str

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['matches'] = [m.to_dict() for m in self.matches]
        return d


class EventMatcher:
    """事件相似度匹配引擎"""

    def __init__(self):
        self.event_db = get_event_database()

    def analyze_event(self,
                     event_type: str,
                     title: str,
                     sentiment: float,
                     intensity: str = 'medium',
                     affected_markets: List[str] = None,
                     affected_sectors: List[str] = None) -> EventAnalysis:
        """
        分析事件，查找历史相似事件

        Args:
            event_type: 事件类型 (central_bank/geopolitics/policy等)
            title: 事件标题
            sentiment: 情绪 -1 ~ +1
            intensity: 强度 high/medium/low
            affected_markets: 影响市场
            affected_sectors: 影响板块

        Returns:
            EventAnalysis
        """
        if affected_markets is None:
            affected_markets = []
        if affected_sectors is None:
            affected_sectors = []

        # 从标题提取关键词
        keywords = self._extract_keywords(title)

        # 查找相似事件
        matches = self._find_similar_events(
            event_type=event_type,
            sentiment=sentiment,
            intensity=intensity,
            affected_markets=affected_markets,
            keywords=keywords
        )

        # 统计预测
        if matches:
            avg_1w = sum(m.predicted_1w for m in matches) / len(matches)
            avg_1m = sum(m.predicted_1m for m in matches) / len(matches)
            confidence_score = sum(m.similarity for m in matches) / len(matches)
        else:
            avg_1w = avg_1m = 0
            confidence_score = 0

        # 置信度
        if confidence_score > 0.7:
            confidence = 'high'
        elif confidence_score > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'

        # 生成建议
        recommendation, reasoning = self._generate_recommendation(
            event_type, sentiment, matches, avg_1w, avg_1m
        )

        return EventAnalysis(
            input_event_type=event_type,
            input_title=title,
            input_sentiment=sentiment,
            input_intensity=intensity,
            input_markets=affected_markets,
            input_sectors=affected_sectors,
            matches=matches,
            avg_predicted_1w=avg_1w,
            avg_predicted_1m=avg_1m,
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本提取关键词"""
        # 常见关键词
        keyword_patterns = {
            '降息': ['降息', '宽松', '宽松货币'],
            '加息': ['加息', '紧缩', '收紧货币'],
            '战争': ['战争', '冲突', '袭击', '军事'],
            '制裁': ['制裁', '禁运', '封锁'],
            '政策': ['政策', '出台', '发布', '调整'],
            '业绩': ['业绩', '利润', '营收', '增长', '下滑'],
            'CPI': ['CPI', '通胀', '物价'],
            'GDP': ['GDP', '经济', '增长'],
            '非农': ['非农', '就业', '失业率'],
        }

        keywords = []
        for category, words in keyword_patterns.items():
            for word in words:
                if word in text:
                    keywords.append(word)

        return keywords

    def _find_similar_events(self,
                            event_type: str,
                            sentiment: float,
                            intensity: str,
                            affected_markets: List[str],
                            keywords: List[str]) -> List[EventMatch]:
        """查找相似事件"""
        matches = []

        for event in self.event_db.events:
            similarity = self._calculate_similarity(
                event, event_type, sentiment, intensity, affected_markets, keywords
            )

            if similarity < 0.3:  # 阈值
                continue

            # 加权预测（根据相似度）
            weight = similarity
            predicted_1w = event.next_week_change * weight
            predicted_1m = event.next_month_change * weight

            # 匹配原因
            reasons = []
            if event.event_type == event_type:
                reasons.append(f'同类型事件({EVENT_TYPES.get(event_type, event_type)})')
            if abs(event.sentiment - sentiment) < 0.3:
                reasons.append('情绪相似')
            if event.intensity == intensity:
                reasons.append('强度相近')
            market_overlap = set(affected_markets) & set(event.affected_markets)
            if market_overlap:
                reasons.append(f'共同影响: {",".join(market_overlap)}')

            match = EventMatch(
                historical_event=event,
                similarity=similarity,
                predicted_immediate=event.immediate_reaction * weight,
                predicted_1d=event.next_day_change * weight,
                predicted_1w=predicted_1w,
                predicted_1m=predicted_1m,
                confidence='high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low',
                match_reasons=reasons
            )
            matches.append(match)

        # 排序
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[:5]

    def _calculate_similarity(self,
                             event: MarketEvent,
                             event_type: str,
                             sentiment: float,
                             intensity: str,
                             affected_markets: List[str],
                             keywords: List[str]) -> float:
        """计算相似度"""
        score = 0.0

        # 类型匹配 (30%)
        if event.event_type == event_type:
            score += 0.3

        # 情绪匹配 (25%)
        sentiment_diff = abs(event.sentiment - sentiment)
        score += (1 - sentiment_diff) * 0.25

        # 强度匹配 (15%)
        intensity_map = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
        intensity_diff = abs(intensity_map.get(event.intensity, 0) - intensity_map.get(intensity, 0))
        score += (1 - intensity_diff) * 0.15

        # 市场重叠 (20%)
        if affected_markets and event.affected_markets:
            overlap = len(set(affected_markets) & set(event.affected_markets))
            union = len(set(affected_markets) | set(event.affected_markets))
            score += (overlap / union) * 0.2 if union > 0 else 0

        # 关键词匹配 (10%)
        event_text = event.title + event.description
        keyword_match = sum(1 for kw in keywords if kw in event_text)
        score += min(keyword_match * 0.05, 0.1)

        return min(score, 1.0)

    def _generate_recommendation(self,
                                 event_type: str,
                                 sentiment: float,
                                 matches: List[EventMatch],
                                 avg_1w: float,
                                 avg_1m: float) -> Tuple[str, str]:
        """生成建议"""
        if not matches:
            return '信息不足', '无足够历史数据参考'

        # 计算加权情绪
        weighted_sentiment = sum(m.similarity * m.historical_event.sentiment for m in matches) / sum(m.similarity for m in matches)

        # 基于预测和情绪判断
        if avg_1m > 8 and weighted_sentiment > 0.3:
            return '强烈买入', f'历史相似事件发生后平均上涨{avg_1m:.1f}%，市场情绪偏多'
        elif avg_1m > 3 and weighted_sentiment > 0:
            return '买入', f'历史相似事件发生后平均上涨{avg_1m:.1f}%，短期看好'
        elif avg_1m < -8 and weighted_sentiment < -0.3:
            return '强烈卖出', f'历史相似事件发生后平均下跌{abs(avg_1m):.1f}%，市场情绪偏空'
        elif avg_1m < -3 and weighted_sentiment < 0:
            return '卖出', f'历史相似事件发生后平均下跌{abs(avg_1m):.1f}%，短期谨慎'
        elif abs(avg_1m) < 3:
            return '观望', f'历史相似事件发生后走势分化，平均涨跌幅{avg_1m:+.1f}%'
        else:
            return '中性', f'事件影响中性，历史平均收益{avg_1m:+.1f}%'

    def analyze_news_event(self, news_title: str, news_content: str = '') -> EventAnalysis:
        """
        分析新闻事件（自动识别类型和情绪）

        Args:
            news_title: 新闻标题
            news_content: 新闻内容

        Returns:
            EventAnalysis
        """
        text = news_title + ' ' + news_content

        # 自动识别事件类型
        event_type = self._detect_event_type(text)

        # 自动识别情绪
        sentiment = self._detect_sentiment(text)

        # 自动识别强度
        intensity = self._detect_intensity(text, sentiment)

        # 自动识别影响市场
        markets = self._detect_affected_markets(text)

        # 自动识别影响板块
        sectors = self._detect_affected_sectors(text)

        return self.analyze_event(
            event_type=event_type,
            title=news_title,
            sentiment=sentiment,
            intensity=intensity,
            affected_markets=markets,
            affected_sectors=sectors
        )

    def _detect_event_type(self, text: str) -> str:
        """识别事件类型"""
        patterns = {
            'central_bank': ['央行', '美联储', '加息', '降息', '利率', '货币政策', 'FOMC', 'LPR'],
            'geopolitics': ['战争', '冲突', '制裁', '外交', '峰会', '俄罗斯', '乌克兰', '中东'],
            'policy': ['政策', '财政部', '证监会', '银保监会', '监管', '出台', '征求意见'],
            'economic_data': ['GDP', 'CPI', 'PPI', '非农', '就业', '制造业', 'PMI', '经济数据'],
            'company': ['业绩', '利润', '营收', '财报', '预亏', '预盈', '重组', '并购'],
            'disaster': ['地震', '洪水', '疫情', '台风', '空难', '事故'],
            'figure': ['耶伦', '鲍威尔', '习近平', '拜登', '普京', '讲话', '演讲'],
        }

        for event_type, keywords in patterns.items():
            if any(kw in text for kw in keywords):
                return event_type

        return 'market'  # 默认市场事件

    def _detect_sentiment(self, text: str) -> float:
        """识别情绪"""
        bullish = ['涨', '大涨', '利好', '增长', '突破', '创新高', '超预期', '增持', '买入', '上调', '回升', '反弹', '正增长']
        bearish = ['跌', '大跌', '利空', '下降', '跌破', '创新低', '不及预期', '减持', '卖出', '下调', '萎缩', '亏损', '预警', '风险']

        bullish_count = sum(1 for kw in bullish if kw in text)
        bearish_count = sum(1 for kw in bearish if kw in text)

        if bullish_count + bearish_count == 0:
            return 0

        return (bullish_count - bearish_count) / (bullish_count + bearish_count)

    def _detect_intensity(self, text: str, sentiment: float) -> str:
        """识别强度"""
        strong_words = ['重磅', '史上', '首次', '大规模', '全面', '紧急', '超预期', '大幅']
        weak_words = ['小幅', '微调', '略', '轻微']

        if any(w in text for w in strong_words) or abs(sentiment) > 0.5:
            return 'high'
        elif any(w in text for w in weak_words) or abs(sentiment) < 0.2:
            return 'low'
        else:
            return 'medium'

    def _detect_affected_markets(self, text: str) -> List[str]:
        """识别影响市场"""
        markets = []
        market_keywords = {
            'A股': ['A股', '上证', '深证', '沪深', '创业板', '科创板', '中国股市'],
            '港股': ['港股', '恒生', 'H股'],
            '美股': ['美股', '纳斯达克', '道琼斯', '标普', '华尔街'],
            '黄金': ['黄金', '金价', 'COMEX'],
            '原油': ['原油', '油价', '布伦特', 'WTI'],
            '外汇': ['外汇', '美元', '人民币', '欧元', '汇率'],
            '债券': ['债券', '国债', '收益率'],
        }

        for market, keywords in market_keywords.items():
            if any(kw in text for kw in keywords):
                markets.append(market)

        return markets if markets else ['全球']

    def _detect_affected_sectors(self, text: str) -> List[str]:
        """识别影响板块"""
        sectors = []
        sector_keywords = {
            '半导体': ['半导体', '芯片', '集成电路', '光刻'],
            '新能源': ['新能源', '锂电', '光伏', '电动车', '宁德', '比亚迪'],
            '消费': ['消费', '白酒', '食品', '饮料', '茅台'],
            '金融': ['银行', '券商', '保险', '金融'],
            '房地产': ['房地产', '地产', '万科', '恒大', '碧桂园'],
            '医药': ['医药', '医疗', '疫苗', '创新药'],
            '军工': ['军工', '国防', '航天'],
        }

        for sector, keywords in sector_keywords.items():
            if any(kw in text for kw in keywords):
                sectors.append(sector)

        return sectors


# 全局单例
_matcher: Optional[EventMatcher] = None


def get_event_matcher() -> EventMatcher:
    global _matcher
    if _matcher is None:
        _matcher = EventMatcher()
    return _matcher


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("测试事件相似度匹配")
    print("=" * 60)

    matcher = EventMatcher()

    # 测试1: 已知类型事件
    print("\n[1] 分析 '美联储降息50基点' 事件:")
    analysis = matcher.analyze_event(
        event_type='central_bank',
        title='美联储宣布降息50个基点',
        sentiment=0.8,
        intensity='high',
        affected_markets=['美股', '港股', 'A股'],
        affected_sectors=['金融', '科技']
    )

    print(f"  匹配到 {len(analysis.matches)} 个相似历史事件")
    print(f"  预测一周收益: {analysis.avg_predicted_1w:+.2f}%")
    print(f"  预测一月收益: {analysis.avg_predicted_1m:+.2f}%")
    print(f"  置信度: {analysis.confidence}")
    print(f"  建议: {analysis.recommendation} - {analysis.reasoning}")

    if analysis.matches:
        print(f"\n  最相似历史事件:")
        best = analysis.matches[0]
        print(f"    {best.historical_event.title}")
        print(f"    相似度: {best.similarity:.1%}")
        print(f"    匹配原因: {', '.join(best.match_reasons)}")

    # 测试2: 从新闻标题自动识别
    print("\n[2] 自动分析新闻标题:")
    test_titles = [
        '重磅！央行宣布降息10个基点，A股港股双双大涨',
        '俄乌冲突升级！原油黄金暴涨，全球股市下跌',
        '宁德时代业绩超预期，一季度净利润增长超200%',
    ]

    for title in test_titles:
        print(f"\n  标题: {title[:30]}...")
        analysis = matcher.analyze_news_event(title)
        print(f"  识别类型: {EVENT_TYPES.get(analysis.input_event_type, analysis.input_event_type)}")
        print(f"  识别情绪: {analysis.input_sentiment:+.2f}")
        print(f"  影响市场: {', '.join(analysis.input_markets)}")
        print(f"  预测一月: {analysis.avg_predicted_1m:+.2f}%")
        print(f"  建议: {analysis.recommendation}")
