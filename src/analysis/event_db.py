#!/usr/bin/env python3
"""
历史事件库
存储历史重大事件及其市场影响
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import os

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')


# ============================================================
# 事件类型定义
# ============================================================

EVENT_TYPES = {
    'central_bank': '央行政策',
    'geopolitics': '地缘冲突',
    'policy': '行业政策',
    'economic_data': '经济数据',
    'company': '公司事件',
    'disaster': '重大灾害',
    'figure': '重要人物讲话',
    'market': '市场事件',
}


# ============================================================
# 事件数据模型
# ============================================================

@dataclass
class MarketEvent:
    """市场事件"""
    id: str
    date: str  # YYYY-MM-DD
    event_type: str  # central_bank / geopolitics / policy 等
    title: str  # 事件标题
    description: str  # 事件描述

    # 影响范围
    affected_markets: List[str]  # ['A股', '港股', '美股', '黄金', '原油', '外汇']
    affected_sectors: List[str]  # ['半导体', '新能源', '消费']
    affected_stocks: List[str]  # 具体股票代码

    # 市场反应
    immediate_reaction: float  # 立即反应 (%)
    next_day_change: float  # 次日涨跌幅 (%)
    next_week_change: float  # 未来一周涨跌幅 (%)
    next_month_change: float  # 未来一月涨跌幅 (%)

    # 事件特征（用于相似度匹配）
    sentiment: float  # -1 ~ +1 情绪
    intensity: str  # high / medium / low 强度
    duration_days: int  # 持续天数

    # 原始来源
    source: str
    source_url: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# 历史事件库
# ============================================================

class EventDatabase:
    """历史事件数据库"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = '/Users/keira/project/claude/quant_evo/data/event_db.csv'

        self.db_path = db_path
        self.events: List[MarketEvent] = []
        self._load()

    def _load(self):
        """加载事件库"""
        if os.path.exists(self.db_path):
            try:
                df = pd.read_csv(self.db_path)
                for _, row in df.iterrows():
                    self.events.append(MarketEvent(
                        id=str(row['id']),
                        date=row['date'],
                        event_type=row['event_type'],
                        title=row['title'],
                        description=row['description'],
                        affected_markets=eval(row['affected_markets']),
                        affected_sectors=eval(row['affected_sectors']),
                        affected_stocks=eval(row['affected_stocks']),
                        immediate_reaction=row['immediate_reaction'],
                        next_day_change=row['next_day_change'],
                        next_week_change=row['next_week_change'],
                        next_month_change=row['next_month_change'],
                        sentiment=row['sentiment'],
                        intensity=row['intensity'],
                        duration_days=row['duration_days'],
                        source=row['source'],
                        source_url=row.get('source_url', '')
                    ))
                print(f"[EventDB] 加载 {len(self.events)} 个历史事件")
            except Exception as e:
                print(f"[EventDB] 加载失败: {e}")
                self._init_sample_data()
        else:
            self._init_sample_data()
            self.save()

    def _init_sample_data(self):
        """初始化示例数据"""
        self.events = [
            # 央行政策
            MarketEvent(
                id='evt_001',
                date='2024-09-19',
                event_type='central_bank',
                title='美联储降息50基点',
                description='美联储宣布降息50个基点，为2020年来首次降息',
                affected_markets=['美股', '港股', 'A股'],
                affected_sectors=['金融', '科技'],
                affected_stocks=[],
                immediate_reaction=1.5,
                next_day_change=2.1,
                next_week_change=3.5,
                next_month_change=5.2,
                sentiment=0.8,
                intensity='high',
                duration_days=5,
                source='FOMC'
            ),
            MarketEvent(
                id='evt_002',
                date='2024-07-22',
                event_type='central_bank',
                title='中国央行降息10基点',
                description='LPR利率下调，1年期降10基点至3.35%',
                affected_markets=['A股', '港股'],
                affected_sectors=['金融', '房地产'],
                affected_stocks=[],
                immediate_reaction=0.8,
                next_day_change=1.2,
                next_week_change=2.0,
                next_month_change=3.1,
                sentiment=0.6,
                intensity='medium',
                duration_days=3,
                source='中国人民银行'
            ),

            # 地缘冲突
            MarketEvent(
                id='evt_003',
                date='2024-04-14',
                event_type='geopolitics',
                title='伊朗以色列冲突',
                description='伊朗对以色列发动无人机和导弹袭击，中东局势升级',
                affected_markets=['原油', '黄金', '港股'],
                affected_sectors=['能源', '军工'],
                affected_stocks=[],
                immediate_reaction=-1.5,
                next_day_change=-0.8,
                next_week_change=0.5,
                next_month_change=-2.0,
                sentiment=-0.6,
                intensity='high',
                duration_days=10,
                source='新华社'
            ),
            MarketEvent(
                id='evt_004',
                date='2022-02-24',
                event_type='geopolitics',
                title='俄乌战争爆发',
                description='俄罗斯对乌克兰发动特别军事行动',
                affected_markets=['原油', '黄金', '小麦', '全球股市'],
                affected_sectors=['能源', '农业', '军工'],
                affected_stocks=[],
                immediate_reaction=-3.5,
                next_day_change=-2.8,
                next_week_change=-5.0,
                next_month_change=-8.0,
                sentiment=-0.9,
                intensity='high',
                duration_days=30,
                source='BBC'
            ),

            # 行业政策
            MarketEvent(
                id='evt_005',
                date='2023-08-27',
                event_type='policy',
                title='印花税减半征收',
                description='A股印花税减半征收，港股同步下调',
                affected_markets=['A股', '港股'],
                affected_sectors=['券商', '金融'],
                affected_stocks=[],
                immediate_reaction=3.2,
                next_day_change=1.5,
                next_week_change=2.8,
                next_month_change=5.0,
                sentiment=0.9,
                intensity='high',
                duration_days=7,
                source='财政部'
            ),
            MarketEvent(
                id='evt_006',
                date='2024-05-20',
                event_type='policy',
                title='房地产重磅政策出台',
                description='取消全国层面房贷利率下限，多地取消限购',
                affected_markets=['A股', '港股'],
                affected_sectors=['房地产', '银行', '家电'],
                affected_stocks=[],
                immediate_reaction=2.5,
                next_day_change=1.8,
                next_week_change=3.0,
                next_month_change=4.5,
                sentiment=0.7,
                intensity='high',
                duration_days=14,
                source='央行'
            ),

            # 经济数据
            MarketEvent(
                id='evt_007',
                date='2024-06-12',
                event_type='economic_data',
                title='美国CPI超预期回落',
                description='美国5月CPI同比上涨3.3%，低于预期',
                affected_markets=['美股', '黄金', '外汇'],
                affected_sectors=[],
                affected_stocks=[],
                immediate_reaction=1.2,
                next_day_change=0.8,
                next_week_change=1.5,
                next_month_change=2.5,
                sentiment=0.7,
                intensity='medium',
                duration_days=3,
                source='美国劳工部'
            ),
            MarketEvent(
                id='evt_008',
                date='2024-07-15',
                event_type='economic_data',
                title='中国GDP超预期',
                description='二季度GDP同比增长6.3%，高于预期',
                affected_markets=['A股', '港股'],
                affected_sectors=['消费', '制造'],
                affected_stocks=[],
                immediate_reaction=1.5,
                next_day_change=0.9,
                next_week_change=1.8,
                next_month_change=2.5,
                sentiment=0.6,
                intensity='medium',
                duration_days=2,
                source='国家统计局'
            ),

            # 公司事件
            MarketEvent(
                id='evt_009',
                date='2024-04-19',
                event_type='company',
                title='宁德时代业绩超预期',
                description='一季度净利润同比增长近10倍',
                affected_markets=['A股', '创业板'],
                affected_sectors=['新能源', '锂电池'],
                affected_stocks=['sz.300750'],
                immediate_reaction=5.0,
                next_day_change=3.2,
                next_week_change=8.0,
                next_month_change=12.0,
                sentiment=0.9,
                intensity='high',
                duration_days=10,
                source='公司公告'
            ),

            # 市场事件
            MarketEvent(
                id='evt_010',
                date='2024-01-22',
                event_type='market',
                title='量化基金爆仓引发踩踏',
                description='多家量化基金触及止损线，引发连锁抛售',
                affected_markets=['A股'],
                affected_sectors=[],
                affected_stocks=[],
                immediate_reaction=-3.8,
                next_day_change=-1.5,
                next_week_change=0.5,
                next_month_change=2.0,
                sentiment=-0.8,
                intensity='high',
                duration_days=5,
                source='市场观察'
            ),
        ]
        print(f"[EventDB] 初始化 {len(self.events)} 个示例事件")

    def save(self):
        """保存事件库"""
        if not self.events:
            return

        df = pd.DataFrame([e.to_dict() for e in self.events])
        df.to_csv(self.db_path, index=False)
        print(f"[EventDB] 保存 {len(self.events)} 个事件到 {self.db_path}")

    def add_event(self, event: MarketEvent):
        """添加事件"""
        self.events.append(event)

    def get_events_by_type(self, event_type: str) -> List[MarketEvent]:
        """按类型获取事件"""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_date_range(self, start: str, end: str) -> List[MarketEvent]:
        """按日期范围获取事件"""
        return [e for e in self.events if start <= e.date <= end]

    def get_events_by_market(self, market: str) -> List[MarketEvent]:
        """按市场获取相关事件"""
        return [e for e in self.events if market in e.affected_markets]

    def get_similar_events(self, event: MarketEvent, top_n: int = 5) -> List[Tuple[MarketEvent, float]]:
        """
        查找相似事件

        Returns:
            [(相似事件, 相似度分数), ...]
        """
        similarities = []

        for e in self.events:
            if e.id == event.id:
                continue

            sim = self._calculate_similarity(event, e)
            similarities.append((e, sim))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def _calculate_similarity(self, e1: MarketEvent, e2: MarketEvent) -> float:
        """计算两个事件的相似度"""
        score = 0.0

        # 类型相同 (30%)
        if e1.event_type == e2.event_type:
            score += 0.3

        # 情绪相似 (20%)
        sentiment_diff = abs(e1.sentiment - e2.sentiment)
        score += (1 - sentiment_diff) * 0.2

        # 强度相似 (20%)
        intensity_map = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
        intensity_diff = abs(intensity_map.get(e1.intensity, 0) - intensity_map.get(e2.intensity, 0))
        score += (1 - intensity_diff) * 0.2

        # 市场重叠 (30%)
        overlap = len(set(e1.affected_markets) & set(e2.affected_markets))
        union = len(set(e1.affected_markets) | set(e2.affected_markets))
        if union > 0:
            score += (overlap / union) * 0.3

        return score

    def get_statistics_by_type(self) -> Dict:
        """按类型统计事件平均表现"""
        stats = defaultdict(lambda: {
            'count': 0,
            'immediate_reaction': [],
            'next_week_change': [],
            'next_month_change': []
        })

        for e in self.events:
            event_type = EVENT_TYPES.get(e.event_type, e.event_type)
            stats[event_type]['count'] += 1
            stats[event_type]['immediate_reaction'].append(e.immediate_reaction)
            stats[event_type]['next_week_change'].append(e.next_week_change)
            stats[event_type]['next_month_change'].append(e.next_month_change)

        # 计算平均值
        result = {}
        for event_type, data in stats.items():
            result[event_type] = {
                'count': data['count'],
                'avg_immediate': sum(data['immediate_reaction']) / len(data['immediate_reaction']),
                'avg_next_week': sum(data['next_week_change']) / len(data['next_week_change']),
                'avg_next_month': sum(data['next_month_change']) / len(data['next_month_change']),
            }

        return result


# 全局单例
_event_db: Optional[EventDatabase] = None


def get_event_database() -> EventDatabase:
    global _event_db
    if _event_db is None:
        _event_db = EventDatabase()
    return _event_db


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("测试历史事件库")
    print("=" * 60)

    db = EventDatabase()

    # 统计
    print("\n[1] 按类型统计事件平均表现:")
    stats = db.get_statistics_by_type()
    for event_type, s in stats.items():
        print(f"  {event_type}: {s['count']}起, "
              f"平均次日反应{s['avg_immediate']:+.2f}%, "
              f"一周{s['avg_next_week']:+.2f}%, "
              f"一月{s['avg_next_month']:+.2f}%")

    # 查找相似事件
    print("\n[2] 查找与'美联储降息'最相似的事件:")
    fed_event = db.events[0]  # 美联储降息事件
    similar = db.get_similar_events(fed_event, top_n=3)
    for e, sim in similar:
        print(f"  [{sim:.1%}] {e.title}")
        print(f"       隔日: {e.next_day_change:+.2f}%, 一周: {e.next_week_change:+.2f}%, 一月: {e.next_month_change:+.2f}%")
