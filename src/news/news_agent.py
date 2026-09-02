#!/usr/bin/env python3
"""
新闻Agent v2 - 新闻监控、情感分析、事件驱动信号
================================================

功能:
1. 定时抓取财经新闻 (多源备用)
2. AI情感分析 - 判断对A股的影响
3. 事件驱动信号 - 基于重大新闻生成交易信号
4. 推送通知 - 重要新闻及时推送
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import os
import json
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request
import urllib.error

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
CACHE_DIR = f'{PROJECT_ROOT}/data/cache'
NEWS_CACHE = f'{CACHE_DIR}/news_cache.json'
NEWS_CONFIG = f'{CACHE_DIR}/news_config.json'
ALERT_HISTORY = f'{CACHE_DIR}/news_alerts.json'

# 新闻源配置 - 多源备用
NEWS_SOURCES = {
    'ft_china': {
        'name': 'Financial Times China',
        'url': 'https://www.ft.com/china?format=rss',
        'keywords': ['china', 'chinese', 'beijing', 'economy', 'trade', 'market'],
        'weight': 0.9,
        'enabled': True
    },
    'ft_markets': {
        'name': 'Financial Times Markets',
        'url': 'https://www.ft.com/markets?format=rss',
        'keywords': ['market', 'stock', 'trading', 'fed', 'china'],
        'weight': 0.8,
        'enabled': True
    },
    'cnbc_china': {
        'name': 'CNBC China',
        'url': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
        'keywords': ['china', 'chinese', 'asian', 'market', 'economy'],
        'weight': 0.8,
        'enabled': True
    },
    'marketwatch': {
        'name': 'MarketWatch',
        'url': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        'keywords': ['china', 'stock', 'market', 'economy', 'fed'],
        'weight': 0.7,
        'enabled': True
    },
    'seeking_alpha': {
        'name': 'Seeking Alpha China',
        'url': 'https://seekingalpha.com/market_currents.xml',
        'keywords': ['china', 'chinese', 'a-share', 'hong kong', 'economy'],
        'weight': 0.7,
        'enabled': True
    },
    'bbc_business': {
        'name': 'BBC Business',
        'url': 'http://feeds.bbci.co.uk/news/business/rss.xml',
        'keywords': ['china', 'trade', 'economy', 'market', 'global'],
        'weight': 0.6,
        'enabled': True
    },
}

# A股相关关键词
A_SHARE_KEYWORDS = {
    'tech': ['半导体', '芯片', '光刻', 'AI', '人工智能', '服务器', '云计算', '科技', 'semiconductor', 'chip', 'ai'],
    'data_center': ['数据中心', '光模块', '光通信', '光纤', '交换机', '服务器', 'optical', 'fiber', 'data center'],
    'new_energy': ['新能源', '光伏', '锂电', '储能', '电动车', '动力电池', 'solar', 'lithium', 'battery'],
    'finance': ['银行', '券商', '保险', '金融科技', ' fintech', 'bank', 'insurance'],
    'consumer': ['消费', '食品', '饮料', '家电', '汽车', '零售', 'consumer', 'retail'],
    'medical': ['医药', '医疗', '疫苗', '医疗器械', '中药', '创新药', 'medical', 'pharma', 'health'],
    'military': ['军工', '国防', '航空航天', '军用', 'military', 'defense', 'aviation'],
    'trade': ['关税', '贸易战', '制裁', '出口管制', '实体清单', 'tariff', 'sanction', 'export control'],
    'macro': ['GDP', 'CPI', 'PPI', 'PMI', '进出口', '外汇', '汇率', 'gdp', 'inflation', 'trade'],
}

SENTIMENT_KEYWORDS = {
    'positive': ['surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'boost', 'growth', 'beat', 'exceed',
                 '上涨', '利好', '增长', '突破', '创新高', '超预期', '业绩增长', '强劲', '复苏'],
    'negative': ['fall', 'drop', 'plunge', 'tumble', 'decline', 'slump', 'crash', 'loss', 'miss',
                 '下跌', '利空', '暴跌', '亏损', '风险', '违约', '制裁', '禁止', '衰退', '疲软'],
    'neutral': ['unchanged', 'steady', 'stable', 'flat', 'wait', 'monitor', 'neutral',
                '观望', '等待', '稳定', '持平', '符合预期']
}

IMPACT_KEYWORDS = {
    'high': ['ban', 'block', 'sanction', 'embargo', 'export control', 'prohibit', 'halt', 'emergency',
             'crisis', 'war', 'tariff', 'retaliation', '禁止', '制裁', '管制', '封锁', '紧急', '危机', '关税'],
    'medium': ['plan', 'propose', 'draft', 'report', 'investigate', 'review', 'study',
               '计划', '提议', '草案', '调查', '审查', '考虑', '评估'],
}

STOCK_POOL = None

def load_stock_pool():
    """加载股票池"""
    global STOCK_POOL
    if STOCK_POOL is None:
        pool_path = f'{PROJECT_ROOT}/data/expanded_stock_pool.json'
        if os.path.exists(pool_path):
            with open(pool_path, 'r') as f:
                STOCK_POOL = json.load(f)
    return STOCK_POOL


@dataclass
class NewsItem:
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: str
    sentiment: str
    sentiment_score: float
    related_sectors: List[str]
    impact_level: str
    a_share_impact: str
    processed_at: str


@dataclass
class NewsSignal:
    code: str
    name: str
    signal: str
    score: float
    news_id: str
    news_title: str
    reason: str
    sentiment: str
    confidence: float
    timestamp: str


class NewsAgent:
    def __init__(self, push_enabled: bool = True):
        self.push_enabled = push_enabled
        self.news_cache = self._load_cache()
        self.config = self._load_config()
        self.alerts = self._load_alerts()
        self.stock_names = {}

    def _load_cache(self) -> Dict:
        if os.path.exists(NEWS_CACHE):
            with open(NEWS_CACHE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'items': [], 'last_update': ''}

    def _save_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(NEWS_CACHE, 'w', encoding='utf-8') as f:
            json.dump(self.news_cache, f, ensure_ascii=False, indent=2)

    def _load_config(self) -> Dict:
        default = {
            'enabled': True,
            'fetch_interval': 300,
            'min_impact_score': 60,
            'max_news_per_source': 30,
            'lark_webhook': '',
            'watched_sectors': ['tech', 'data_center', 'trade'],
        }
        if os.path.exists(NEWS_CONFIG):
            with open(NEWS_CONFIG, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default, **config}
        return default

    def _save_config(self):
        with open(NEWS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def _load_alerts(self) -> List:
        if os.path.exists(ALERT_HISTORY):
            with open(ALERT_HISTORY, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_alerts(self):
        with open(ALERT_HISTORY, 'w', encoding='utf-8') as f:
            json.dump(self.alerts[-100:], f, ensure_ascii=False, indent=2)

    def _get_stock_name(self, code: str) -> str:
        if not self.stock_names:
            pool = load_stock_pool()
            if pool:
                self.stock_names = pool.get('names', {})
        return self.stock_names.get(code, code)

    def fetch_rss_feed(self, source_name: str, source_config: Dict) -> List[Dict]:
        news_items = []
        try:
            url = source_config['url']
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode('utf-8', errors='ignore')

            items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
            max_items = source_config.get('max_items', source_config.get('max_news_per_source', 30))

            for item in items[:max_items]:
                try:
                    # 提取标题
                    title_match = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>', item)
                    if not title_match:
                        title_match = re.search(r'<title>(.*?)</title>', item)
                    title = title_match.group(1).strip() if title_match else ''
                    if not title:
                        continue

                    # 提取链接
                    link_match = re.search(r'<link>(.*?)</link>', item)
                    if not link_match:
                        link_match = re.search(r'<link><!\[CDATA\[(.*?)\]\]></link>', item)
                    link = link_match.group(1).strip() if link_match else ''

                    # 提取描述
                    desc_match = re.search(r'<description><!\[CDATA\[(.*?)\]\]></description>', item)
                    if not desc_match:
                        desc_match = re.search(r'<description>(.*?)</description>', item)
                    description = ''
                    if desc_match:
                        description = desc_match.group(1).strip()
                        description = re.sub(r'<[^>]+>', '', description)
                        description = description[:600]

                    # 提取日期
                    date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
                    pub_date = date_match.group(1).strip() if date_match else datetime.now().isoformat()

                    if link:
                        news_id = hashlib.md5(f"{link}".encode()).hexdigest()[:12]
                        news_items.append({
                            'id': news_id,
                            'title': title,
                            'content': description,
                            'source': source_config['name'],
                            'url': link,
                            'published_at': pub_date,
                        })
                except Exception as e:
                    continue

        except Exception as e:
            print(f"  ✗ {source_config['name']}: {str(e)[:50]}")

        return news_items

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()

        pos_count = sum(1 for kw in SENTIMENT_KEYWORDS['positive'] if kw.lower() in text_lower)
        neg_count = sum(1 for kw in SENTIMENT_KEYWORDS['negative'] if kw.lower() in text_lower)

        if pos_count > neg_count:
            sentiment = 'positive'
            score = min(50 + pos_count * 12, 95)
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = max(50 - neg_count * 12, 5)
        else:
            sentiment = 'neutral'
            score = 50

        return sentiment, max(5, min(95, score))

    def find_related_sectors(self, title: str, content: str) -> List[str]:
        text = (title + ' ' + content).lower()
        related = []

        for sector, keywords in A_SHARE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    if sector not in related:
                        related.append(sector)
                    break

        return related

    def assess_impact(self, title: str, content: str) -> str:
        text = (title + ' ' + content).lower()

        for kw in IMPACT_KEYWORDS['high']:
            if kw in text:
                return 'high'

        for kw in IMPACT_KEYWORDS['medium']:
            if kw in text:
                return 'medium'

        return 'low'

    def assess_a_share_impact(self, title: str, content: str, sentiment: str, sectors: List[str]) -> str:
        text = (title + ' ' + content).lower()

        china_keywords = ['china', 'chinese', 'beijing', 'shanghai', 'a-share', 'a share',
                         'hong kong', 'asian', 'pacific']

        has_china = any(kw in text for kw in china_keywords)

        if not has_china and not sectors:
            return 'none'

        if sentiment == 'positive' and (has_china or sectors):
            return 'bullish'
        elif sentiment == 'negative' and (has_china or sectors):
            return 'bearish'
        elif sentiment == 'neutral' and sectors:
            return 'mixed'

        return 'none'

    def process_news(self, raw_news: List[Dict]) -> List[NewsItem]:
        processed = []
        seen_ids = {n['id'] for n in self.news_cache.get('items', [])}

        for news in raw_news:
            news_id = news.get('id', '')
            if news_id in seen_ids:
                continue

            title = news.get('title', '')
            content = news.get('content', '')

            sentiment, sentiment_score = self.analyze_sentiment(title + ' ' + content)
            sectors = self.find_related_sectors(title, content)
            impact = self.assess_impact(title, content)
            a_share = self.assess_a_share_impact(title, content, sentiment, sectors)

            item = NewsItem(
                id=news_id,
                title=title,
                content=content,
                source=news.get('source', ''),
                url=news.get('url', ''),
                published_at=news.get('published_at', ''),
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                related_sectors=sectors,
                impact_level=impact,
                a_share_impact=a_share,
                processed_at=datetime.now().isoformat()
            )
            processed.append(item)
            seen_ids.add(news_id)

        return processed

    def fetch_all_news(self) -> List[NewsItem]:
        print("=" * 60)
        print("新闻Agent v2 - 抓取中...")
        print("=" * 60)

        all_raw_news = []

        for source_key, source_config in NEWS_SOURCES.items():
            if not source_config.get('enabled', True):
                continue

            print(f"抓取 {source_config['name']}...", end=' ')
            items = self.fetch_rss_feed(source_key, source_config)
            all_raw_news.extend(items)
            print(f"✓ {len(items)}条")

        print(f"\n总计获取 {len(all_raw_news)} 条原始新闻")

        processed = self.process_news(all_raw_news)
        print(f"新增处理 {len(processed)} 条")

        if processed:
            self.news_cache['items'].extend([asdict(n) for n in processed])
            self.news_cache['items'] = self.news_cache['items'][-500:]
            self.news_cache['last_update'] = datetime.now().isoformat()
            self._save_cache()

        return processed

    def get_important_news(self, min_score: int = 60, limit: int = 20) -> List[NewsItem]:
        items = self.news_cache.get('items', [])
        important = []

        for item in items:
            score = item.get('sentiment_score', 50)
            impact = item.get('impact_level', 'low')

            impact_mult = {'high': 1.4, 'medium': 1.1, 'low': 1.0, 'none': 0.8}
            final_score = score * impact_mult.get(impact, 1.0)

            if final_score >= min_score or impact == 'high':
                important.append(NewsItem(**item))

        important.sort(key=lambda x: (x.sentiment_score * (1.3 if x.impact_level == 'high' else 1.0), x.published_at), reverse=True)

        return important[:limit]

    def generate_signals(self) -> List[NewsSignal]:
        signals = []
        important_news = self.get_important_news(min_score=60)

        # 板块相关龙头股
        sector_stocks = {
            'tech': ['sh.601012', 'sz.002371', 'sh.603501', 'sz.688981'],
            'data_center': ['sz.300308', 'sh.002463', 'sh.600588', 'sz.300502'],
            'new_energy': ['sh.601012', 'sz.300750', 'sh.600900', 'sh.600274'],
            'finance': ['sh.601398', 'sh.601318', 'sh.600036', 'sh.601166'],
            'consumer': ['sh.600519', 'sz.000858', 'sh.600887', 'sz.002304'],
            'medical': ['sh.600276', 'sz.300760', 'sh.601607', 'sz.300015'],
            'military': ['sh.601989', 'sh.600893', 'sz.000733', 'sh.600760'],
            'trade': ['sh.601398', 'sh.600036', 'sz.002475', 'sh.601012'],
            'macro': ['sh.601398', 'sh.600036', 'sh.601318', 'sz.399001'],
        }

        for news in important_news:
            if news.a_share_impact == 'none' or not news.related_sectors:
                continue

            related_codes = []
            for sector in news.related_sectors:
                if sector in sector_stocks:
                    related_codes.extend(sector_stocks[sector])

            related_codes = list(set(related_codes))[:5]

            for code in related_codes:
                if news.a_share_impact == 'bullish':
                    signal_type = 'BUY'
                    base_score = 65
                elif news.a_share_impact == 'bearish':
                    signal_type = 'SELL'
                    base_score = 65
                else:
                    signal_type = 'HOLD'
                    base_score = 50

                if news.sentiment == 'positive':
                    final_score = base_score + news.sentiment_score * 0.35
                elif news.sentiment == 'negative':
                    final_score = base_score - (100 - news.sentiment_score) * 0.35
                else:
                    final_score = base_score

                final_score = max(15, min(95, final_score))
                confidence = (news.sentiment_score / 100) * (1.0 if news.impact_level == 'high' else 0.75)

                signals.append(NewsSignal(
                    code=code,
                    name=self._get_stock_name(code),
                    signal=signal_type,
                    score=final_score,
                    news_id=news.id,
                    news_title=news.title[:60],
                    reason=f"{news.source}: {news.title[:70]}",
                    sentiment=news.sentiment,
                    confidence=confidence,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))

        seen = {}
        for sig in signals:
            key = sig.code
            if key not in seen or sig.score > seen[key].score:
                seen[key] = sig

        return sorted(list(seen.values()), key=lambda x: -x.score)[:10]

    def send_alert(self, news: NewsItem):
        if not self.push_enabled:
            return

        if any(a.get('news_id') == news.id for a in self.alerts):
            return

        sentiment_emoji = '🟢' if news.sentiment == 'positive' else '🔴' if news.sentiment == 'negative' else '🟡'
        impact_emoji = '⚠️' if news.impact_level == 'high' else '📢' if news.impact_level == 'medium' else '📋'
        a_share_emoji = '📈' if news.a_share_impact == 'bullish' else '📉' if news.a_share_impact == 'bearish' else '➡️'

        message = f"""📰 **财经新闻速报**

**来源**: {news.source}
**时间**: {news.published_at}

{impact_emoji} **{news.title}**

{news.content[:200]}...

{sentiment_emoji} 情感: {'利好' if news.sentiment == 'positive' else '利空' if news.sentiment == 'negative' else '中性'}
{a_share_emoji} A股影响: {'看多' if news.a_share_impact == 'bullish' else '看空' if news.a_share_impact == 'bearish' else '中性'}
📂 关联板块: {', '.join(news.related_sectors) if news.related_sectors else '通用'}

🔗 {news.url}"""

        if self.config.get('lark_webhook'):
            try:
                self._send_lark_message(message)
                print(f"✓ 推送: {news.title[:40]}...")
            except Exception as e:
                print(f"✗ 推送失败: {e}")

        self.alerts.append({
            'news_id': news.id,
            'title': news.title,
            'sent': datetime.now().isoformat()
        })
        self._save_alerts()

    def _send_lark_message(self, message: str):
        import requests
        webhook = self.config.get('lark_webhook')
        if not webhook:
            return

        payload = {"msg_type": "text", "content": {"text": message}}
        requests.post(webhook, json=payload, timeout=10)

    def run(self) -> Dict:
        print("\n" + "=" * 60)
        print("新闻Agent v2 启动")
        print("=" * 60)

        new_news = self.fetch_all_news()
        important = self.get_important_news(min_score=self.config.get('min_impact_score', 60))
        signals = self.generate_signals()

        print(f"\n📊 重要新闻: {len(important)} 条")
        print(f"📈 生成信号: {len(signals)} 条")

        for news in important[:5]:
            self.send_alert(news)

        signals_file = f'{CACHE_DIR}/news_signals.json'
        with open(signals_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(s) for s in signals], f, ensure_ascii=False, indent=2)

        return {
            'new_news': len(new_news),
            'important_news': len(important),
            'signals': len(signals),
            'signals_list': signals
        }


def main():
    agent = NewsAgent(push_enabled=True)
    result = agent.run()

    print("\n" + "=" * 60)
    print("运行结果")
    print("=" * 60)
    print(f"新新闻: {result['new_news']} 条")
    print(f"重要新闻: {result['important_news']} 条")
    print(f"生成信号: {result['signals']} 条")

    if result['signals_list']:
        print("\n信号列表:")
        for sig in result['signals_list'][:5]:
            emoji = '🟢' if sig.signal == 'BUY' else '🔴' if sig.signal == 'SELL' else '🟡'
            print(f"  {emoji} {sig.signal} {sig.code} {sig.name}: {sig.score:.1f}分")
            print(f"     原因: {sig.reason[:60]}...")


if __name__ == '__main__':
    main()
