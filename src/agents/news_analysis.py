#!/usr/bin/env python3
"""
News Analysis Agent - 新闻分析Agent
监控财经舆情、生成情绪因子
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.agents.base import Agent, Message


@dataclass
class SentimentScore:
    """情绪评分"""
    market: float      # 市场整体情绪 -1 ~ +1
    sector: Dict[str, float]  # 板块情绪
    hot_topics: List[Dict]    # 热点话题
    last_update: str


class NewsAnalysisAgent(Agent):
    """新闻分析Agent"""

    def __init__(self):
        super().__init__("NewsAnalysis")
        self.sentiment_cache: Optional[SentimentScore] = None
        self.news_cache: List[Dict] = []
        self._news_service = None

    @property
    def news_service(self):
        """延迟加载新闻服务"""
        if self._news_service is None:
            from src.data.news_service import get_news_service
            self._news_service = get_news_service()
        return self._news_service

    def process(self, message: Optional[Message] = None) -> Optional[Message]:
        """处理消息"""
        if message and message.content.get('action') == 'analyze':
            return self.analyze_news(message.content)

        # 默认：分析舆情
        return self.analyze_news({})

    def analyze_news(self, context: Dict[str, Any]) -> Optional[Message]:
        """分析新闻舆情"""
        print("[NewsAnalysis] 分析舆情...")

        # 获取市场热点（真实数据）
        hot_topics = self.news_service.get_hot_topics(limit=10)

        # 计算情绪评分
        sentiment = self._calculate_sentiment(hot_topics)

        # 提取情绪因子
        sentiment_factors = self._extract_sentiment_factors(sentiment)

        # 获取最新新闻
        market_news = self.news_service.get_market_news(days=3)
        self.news_cache = market_news

        self.sentiment_cache = sentiment
        self.set_state('last_sentiment', {
            'market': sentiment.market,
            'hot_count': len(hot_topics)
        })

        return self.send('Orchestrator', {
            'action': 'analysis_complete',
            'sentiment': {
                'market': sentiment.market,
                'hot_topics': hot_topics,
                'sector': sentiment.sector
            },
            'factors': sentiment_factors,
            'news_count': len(market_news)
        }, 'RESPONSE')

    def _calculate_sentiment(self, topics: List[Dict]) -> SentimentScore:
        """计算情绪评分"""
        if not topics:
            return SentimentScore(
                market=0,
                sector={},
                hot_topics=[],
                last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        # 加权平均
        total_weight = 0
        weighted_sum = 0
        sector_scores: Dict[str, List[float]] = {}

        for topic in topics:
            # 使用情绪绝对值作为权重（强情绪话题更重要）
            weight = abs(topic.get('sentiment', 0)) + 0.5
            sentiment_val = topic.get('sentiment', 0)
            weighted_sum += sentiment_val * weight
            total_weight += weight

            # 累加板块情绪
            for stock in topic.get('related_stocks', []):
                sector = self._get_sector(stock)
                if sector not in sector_scores:
                    sector_scores[sector] = []
                sector_scores[sector].append(sentiment_val)

        market_sentiment = weighted_sum / total_weight if total_weight > 0 else 0

        # 计算板块平均
        sector_avg = {}
        for sector, scores in sector_scores.items():
            sector_avg[sector] = sum(scores) / len(scores)

        return SentimentScore(
            market=market_sentiment,
            sector=sector_avg,
            hot_topics=topics,
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

    def _extract_sentiment_factors(self, sentiment: SentimentScore) -> Dict[str, float]:
        """提取情绪因子"""
        factors = {
            'market_sentiment': sentiment.market,
            'hot_topic_count': len(sentiment.hot_topics),
            'bullish_count': sum(1 for t in sentiment.hot_topics if t.get('sentiment', 0) > 0),
            'bearish_count': sum(1 for t in sentiment.hot_topics if t.get('sentiment', 0) < 0),
            'neutral_count': sum(1 for t in sentiment.hot_topics if t.get('sentiment', 0) == 0),
        }

        # 板块情绪因子
        for sector in ['半导体', '新能源', '消费', '金融', '光模块', '医药']:
            factors[f'{sector}_sentiment'] = sentiment.sector.get(sector, 0)

        return factors

    def _get_sector(self, code: str) -> str:
        """获取股票所属板块"""
        sector_map = {
            # 半导体
            'sh.688012': '半导体', 'sh.603986': '半导体',
            'sz.002409': '半导体', 'sh.600584': '半导体',
            # 光模块
            'sz.300502': '光模块', 'sz.300308': '光模块',
            # 新能源
            'sz.300750': '新能源', 'sz.002594': '新能源',
            # 消费
            'sh.600519': '消费', 'sh.600887': '消费',
            'sh.601888': '消费',
            # 金融
            'sh.600036': '金融', 'sh.601318': '金融',
            'sz.000001': '金融',
            # 医药
            'sh.600276': '医药',
            # 房地产
            'sz.000002': '房地产',
            # 电子
            'sz.000021': '电子', 'sz.300408': '电子',
        }
        return sector_map.get(code, '其他')

    def get_market_sentiment(self) -> float:
        """获取市场情绪"""
        if self.sentiment_cache:
            return self.sentiment_cache.market
        return 0

    def get_sentiment_factor(self, factor_name: str) -> float:
        """获取特定情绪因子"""
        if self.sentiment_cache:
            factors = self._extract_sentiment_factors(self.sentiment_cache)
            return factors.get(factor_name, 0)
        return 0

    def get_recent_news(self, limit: int = 20) -> List[Dict]:
        """获取最近新闻"""
        return self.news_cache[:limit]

    def generate_market_report(self) -> str:
        """生成市场简报"""
        if not self.sentiment_cache:
            return "暂无数据"

        sentiment_label = '偏多' if self.sentiment_cache.market > 0.2 else '偏空' if self.sentiment_cache.market < -0.2 else '中性'

        report = f"""【市场舆情简报 - {self.sentiment_cache.last_update}】

市场情绪: {sentiment_label} ({self.sentiment_cache.market:+.2f})

热点话题:
"""
        for i, topic in enumerate(self.sentiment_cache.hot_topics[:5], 1):
            sentiment_emoji = '↑' if topic.get('sentiment', 0) > 0 else '↓' if topic.get('sentiment', 0) < 0 else '→'
            report += f"{i}. #{topic['topic']}# {sentiment_emoji}\n"

        report += f"\n板块情绪:"
        for sector, score in list(self.sentiment_cache.sector.items())[:5]:
            emoji = '↑' if score > 0 else '↓' if score < 0 else '→'
            report += f"\n  {sector}: {score:+.2f} {emoji}"

        return report


if __name__ == '__main__':
    # 测试
    agent = NewsAnalysisAgent()

    print("[测试] 新闻分析Agent")
    print("=" * 50)

    msg = agent.analyze_news({})

    if msg:
        print("\n舆情分析结果:")
        print(f"市场情绪: {msg.content['sentiment']['market']:.2f}")
        print(f"热点话题: {len(msg.content['sentiment']['hot_topics'])}")
        print(f"新闻数量: {msg.content.get('news_count', 0)}")

        print(f"\n情绪因子:")
        for k, v in msg.content['factors'].items():
            print(f"  {k}: {v:.2f}")

        print(f"\n{agent.generate_market_report()}")
