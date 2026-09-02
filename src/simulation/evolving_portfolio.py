#!/usr/bin/env python3
"""
进化型模拟盘 v7 - 混合模型版
==============================
使用混合预测: LSTM + XGBoost + 规则引擎
"""

import sys
import os
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict
import random
import xgboost as xgb

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
DATA_DIR = f'{PROJECT_ROOT}/data'
CACHE_DIR = f'{DATA_DIR}/cache'
MODEL_DIR = f'{DATA_DIR}/models'

PORTFOLIO_FILE = f'{CACHE_DIR}/virtual_portfolio.json'
POSITION_FILE = f'{CACHE_DIR}/positions.json'
EVOLUTION_LOG = f'{CACHE_DIR}/ml_evolution_log.json'

DEFAULT_PARAMS = {
    'top_n': 8,               # Plan A: 2→8 分散持仓
    'ml_weight': 0.0,         # ML在当前市场无效，保持关闭
    'trend_weight': 0.20,       # 趋势因子权重（IC=0.23）
    'mean_rev_weight': 0.30,    # 均值回归权重（IC=0.46，最高）
    'momentum_weight': 0.25,    # 动量因子权重（IC=0.13）
    'new_factor_weight': 0.25,   # 新因子权重
    'buy_threshold': 55,
    'sell_threshold': 42,
    'stop_loss': 0.10,         # 止损10%
    'take_profit': 0.20,       # 止盈20% (网格搜索最优)
    'position_size': 0.10,      # Plan A: 0.18→0.10 每只仓位缩小
    'rsi_oversold': 30,
    'rsi_overbought': 65,
    'vol_threshold': 2.0,
    'news_weight': 0.0,
    # 动态阈值参数
    'use_dynamic_threshold': False,   # 是否使用动态阈值（基于分数分布）
    'dynamic_mode': 'percentile',     # 'std' 或 'percentile'
    'dynamic_buy_std': 0.5,          # std模式: 买入阈值 = 均值 + std * 此系数
    'dynamic_sell_std': 0.3,         # std模式: 卖出阈值 = 均值 - std * 此系数
    'dynamic_buy_percentile': 80,    # percentile模式: 买入阈值 = 第N百分位
    'dynamic_sell_percentile': 30,   # percentile模式: 卖出阈值 = 第N百分位
    # 防回转参数
    'sell_cooldown_days': 5,        # 卖出后冷却天数
    'stop_loss_cooldown_days': 10,   # 止损后冷却天数
    'min_hold_days': 3,             # 最小持仓天数（持有多久才考虑卖出）
    # MVO仓位优化参数
    'use_mvo_allocation': False,    # 是否使用Mean-Variance优化仓位分配
    # Plan C: 移动止损参数
    'use_trailing_stop': False,       # Plan C: 移动止损(当前市场无效)
    'trailing_stop_pct': 0.15,      # 移动止损: 从最高价回落15%止损(放宽)
    # Plan D: 趋势确认参数(当前市场无效,保持关闭)
    'require_uptrend': False,         # Plan D: 只在上升趋势中买入(过于严格)
    'uptrend_ma_period': 20,        # 趋势确认MA周期
}


@dataclass
class Position:
    code: str
    name: str
    shares: int
    entry_date: str
    entry_price: float
    current_price: float
    pnl_pct: float
    max_price: float = 0.0  # 移动止损用：持仓期间最高价
    trend_intact: bool = True
    stop_loss: float = 0.10  # 止损10%
    take_profit: float = 0.20  # 止盈20% (网格搜索最优)
    reason: str = ""


@dataclass
class Portfolio:
    cash: float
    total_value: float
    positions: List[dict]
    last_update: str
    equity_curve: List[dict]


# 全局股票名称映射（从expanded_stock_pool.json加载）
STOCK_NAMES = {}

def load_stock_names():
    """从股票池加载名称映射"""
    global STOCK_NAMES
    pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
    if os.path.exists(pool_path):
        with open(pool_path, 'r') as f:
            pool = json.load(f)
            STOCK_NAMES = pool.get('names', {})
    return STOCK_NAMES

def get_stock_name(code: str) -> str:
    if not STOCK_NAMES:
        load_stock_names()
    return STOCK_NAMES.get(code, code)


def load_positions() -> List[Position]:
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            data = json.load(f)
            return [Position(**p) for p in data]
    return []


def save_positions(positions: List[Position]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(POSITION_FILE, 'w') as f:
        json.dump([asdict(p) for p in positions], f, ensure_ascii=False, indent=2)


def load_portfolio() -> Portfolio:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            data['equity_curve'] = data.get('equity_curve', [])
            return Portfolio(**data)
    return Portfolio(
        cash=2000000,
        total_value=2000000,
        positions=[],
        last_update='',
        equity_curve=[]
    )


def save_portfolio(portfolio: Portfolio):
    portfolio.last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(asdict(portfolio), f, ensure_ascii=False, indent=2)


def load_stock_data(codes: list) -> Dict[str, pd.DataFrame]:
    data_dict = {}
    for code in codes:
        # 优先从Parquet缓存加载（最新数据）
        cache_file = f'{CACHE_DIR}/kline_{code.replace(".", "_")}.parquet'
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[df['volume'] > 0]
            if len(df) >= 120:
                data_dict[code] = df
            continue

        # 回退到CSV文件
        code_fmt = code.replace('.', '_')
        filepath = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[df['volume'] > 0]
            if len(df) >= 120:
                data_dict[code] = df
    return data_dict


def calculate_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算增强特征 - 与模型训练时一致"""
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # 收益率特征
    for period in [1, 3, 5, 10, 20, 60]:
        df[f'return_{period}d'] = df['close'].pct_change(period)

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    exp26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林带
    bb_mid = pd.Series(close).rolling(20).mean()
    bb_std = pd.Series(close).rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr'] = pd.Series(tr).rolling(14).mean()
    df['atr_ratio'] = df['atr'] / (close + 1e-8)

    # 均线
    for window in [5, 10, 20, 60, 120]:
        df[f'ma{window}'] = pd.Series(close).rolling(window).mean()
        df[f'price_ma{window}_ratio'] = close / (df[f'ma{window}'] + 1e-8)

    # 均线多头排列
    df['ma_bull_5_20'] = (df['ma5'] > df['ma20']).astype(int)
    df['ma_bull_20_60'] = (df['ma20'] > df['ma60']).astype(int)
    df['ma_bull_all'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(int)
    df['ma_bull'] = df['ma_bull_all']  # 兼容

    # 动量
    for period in [5, 10, 20, 60]:
        df[f'momentum_{period}d'] = close / (np.roll(close, period) + 1e-8) - 1

    df['momentum_accel'] = df['momentum_10d'] - df['momentum_10d'].shift(10)

    # 波动率
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = pd.Series(close).pct_change().rolling(window).std() * np.sqrt(252)

    # 成交量
    df['volume_ma5'] = pd.Series(volume).rolling(5).mean()
    df['volume_ma20'] = pd.Series(volume).rolling(20).mean()
    df['volume_ratio'] = volume / (df['volume_ma20'] + 1e-8)
    df['volume_momentum'] = df['volume'].pct_change(5)

    # 趋势强度
    adx = calculate_adx(df, high, low, close)
    df['trend_strength'] = adx

    # 价格通道
    df['close_upper_channel'] = close / (df['ma20'] + 2 * df['volatility_20d'] / np.sqrt(252) * close + 1e-8)
    df['close_lower_channel'] = close / (df['ma20'] - 2 * df['volatility_20d'] / np.sqrt(252) * close + 1e-8)

    # 价格速度
    df['price_velocity'] = pd.Series(close).pct_change().rolling(5).mean() * 252

    # 支撑位比率
    df['support_ratio'] = close / (df['bb_lower'] + 1e-8)

    # ========== 新增因子 ==========

    # 1. 成交量异动因子
    # 放量突破：今日成交量 > 20日均量的2倍 且 价格上涨
    df['volume_surge'] = ((df['volume_ratio'] > 2.0) & (df['return_1d'] > 0)).astype(int)
    # 缩量整理：成交量 < 20日均量的50%
    df['volume_shrink'] = (df['volume_ratio'] < 0.5).astype(int)
    # 综合评分
    df['volume_score'] = np.clip(df['volume_ratio'] * df['return_1d'] * 10 + 50, 0, 100)

    # 2. 布林带收窄因子（波动率收缩）
    # 布林带宽度标准化
    bb_width = (df['bb_upper'] - df['bb_lower']) / (df['ma20'] + 1e-8)
    bb_width_ma = bb_width.rolling(20).mean()
    df['bb_squeeze'] = bb_width / (bb_width_ma + 1e-8)  # < 0.8 表示收窄
    # 收窄后突破信号
    df['bb_breakout'] = ((bb_width < bb_width_ma * 0.8) & (df['return_1d'] > 0.02)).astype(int)

    # 3. 动量加速因子
    # 10日动量的变化率（加速度）
    mom10 = df['momentum_10d']
    mom10_change = mom10 - mom10.shift(5)
    df['momentum_accel'] = mom10_change
    # 动量强度：当前动量 vs 历史动量
    mom10_hist_max = mom10.rolling(60).max()
    mom10_hist_min = mom10.rolling(60).min()
    df['momentum_strength'] = (mom10 - mom10_hist_min) / (mom10_hist_max - mom10_hist_min + 1e-8)

    # 综合新因子评分（0-100）
    df['new_factor_score'] = (
        df['volume_score'] * 0.3 +
        (100 - np.clip(df['bb_squeeze'] * 50, 0, 100)) * 0.3 +
        df['momentum_strength'] * 100 * 0.4
    )

    # ========== 新增技术因子 ==========

    # 1. KDJ随机指标
    low_n = pd.Series(low).rolling(9).min()
    high_n = pd.Series(high).rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-8) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    df['kdj金叉'] = ((df['kdj_k'] > df['kdj_d']) & (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))).astype(int)
    df['kdj死叉'] = ((df['kdj_k'] < df['kdj_d']) & (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))).astype(int)

    # 2. CCI商品通道指数
    tp = pd.Series((high + low + close) / 3)
    sma_tp = tp.rolling(14).mean()
    mad = (tp - sma_tp).abs().rolling(14).mean()
    df['cci'] = (tp - sma_tp) / (mad * 0.015 + 1e-8)

    # 3. OBV能量潮
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    obv = close_s.diff()
    obv_sign = np.where(obv > 0, 1, -1)
    df['obv'] = (volume_s * pd.Series(obv_sign)).cumsum()
    df['obv_ma5'] = df['obv'].rolling(5).mean()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['obv多头'] = (df['obv'] > df['obv_ma20']).astype(int)

    # 4. MFI资金流量指数
    tp_mfi = pd.Series((high + low + close) / 3)
    mf_raw = tp_mfi * volume_s
    mf_pos = mf_raw.where(tp_mfi > tp_mfi.shift(1), 0).rolling(14).sum()
    mf_neg = mf_raw.where(tp_mfi < tp_mfi.shift(1), 0).rolling(14).sum()
    mr = mf_pos / (mf_neg + 1e-8)
    df['mfi'] = 100 - (100 / (1 + mr))
    df['mfi超卖'] = (df['mfi'] < 20).astype(int)
    df['mfi超买'] = (df['mfi'] > 80).astype(int)

    # 5. ROC变动率
    df['roc_12'] = (close_s - close_s.shift(12)) / (close_s.shift(12) + 1e-8) * 100
    df['roc_6'] = (close_s - close_s.shift(6)) / (close_s.shift(6) + 1e-8) * 100

    # 6. TRIX三重指数平滑移动平均
    ema1 = close_s.ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    df['trix'] = ema3.pct_change() * 100
    df['trix_signal'] = df['trix'].ewm(span=9, adjust=False).mean()

    # 7. 乖离率
    df['bias_5'] = (close - df['ma5']) / (df['ma5'] + 1e-8) * 100
    df['bias_10'] = (close - df['ma10']) / (df['ma10'] + 1e-8) * 100
    df['bias_20'] = (close - df['ma20']) / (df['ma20'] + 1e-8) * 100

    # 8. 威廉指标
    df['wr_14'] = -100 * (high_n - close) / (high_n - low_n + 1e-8)
    df['wr_28'] = -100 * (pd.Series(high).rolling(28).max() - close) / (pd.Series(high).rolling(28).max() - pd.Series(low).rolling(28).min() + 1e-8)

    # 9. 估波指标
    df['coppock'] = (close_s.pct_change(10) + close_s.pct_change(14)).rolling(10).sum() / 2 * 100

    # 10. 布林带位置（与现有bb_position相同但更明确）
    df['boll_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['boll_width'] = (df['bb_upper'] - df['bb_lower']) / (df['ma20'] + 1e-8)

    # 11. 成交量量比
    df['vol_ratio_5'] = volume / (pd.Series(volume).rolling(5).mean() + 1e-8)

    # 12. 相对强弱指标（RS）
    df['rs_5_20'] = df['momentum_5d'] - df['momentum_20d']

    return df


def calculate_adx(df: pd.DataFrame, high, low, close, period: int = 14) -> pd.Series:
    """计算ADX趋势强度指标"""
    high_diff = pd.Series(high).diff()
    low_diff = -pd.Series(low).diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = pd.Series(tr).rolling(14).mean()

    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-8))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-8))

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(14).mean()

    return adx


class EnhancedModel:
    """增强模型预测器"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_loaded = False

    def load(self):
        """加载增强模型"""
        model_path = f'{MODEL_DIR}/enhanced_xgb_model.json'
        config_path = f'{MODEL_DIR}/enhanced_model_config.json'

        if not os.path.exists(model_path):
            print("增强模型不存在")
            return False

        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.feature_names = config.get('features', [])

        self.is_loaded = True
        print("增强模型加载成功")
        return True

    def predict(self, df: pd.DataFrame) -> float:
        """预测单只股票"""
        if not self.is_loaded or self.model is None:
            return 0

        df = calculate_enhanced_features(df)

        exclude_cols = {'date', 'code', 'open', 'high', 'low', 'close', 'volume',
                       'amount', 'turn', 'pctChg', 'code_raw', 'target'}

        if not self.feature_names:
            feature_cols = [c for c in df.columns if c not in exclude_cols]
        else:
            feature_cols = self.feature_names

        feature_cols = [c for c in feature_cols if c in df.columns]

        if not feature_cols:
            return 0

        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if X.shape[0] == 0:
            return 0

        try:
            pred = self.model.predict(X[-1:])[0]
            return float(pred)
        except:
            return 0

    def predict_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """预测所有股票"""
        predictions = {}
        for code, df in data_dict.items():
            if len(df) >= 60:
                try:
                    predictions[code] = self.predict(df)
                except:
                    predictions[code] = 0
        return predictions


class MultiStrategySignal:
    """
    多策略信号生成器 v7 - 混合模型版

    使用混合预测:
    - LSTM: 时序预测
    - XGBoost: 横截面预测
    - 规则引擎: 技术指标
    - 行业轮动: 强势行业优先
    - ICIR动态权重: 根据ICIR调整因子权重
    """

    def __init__(self, model: EnhancedModel = None, use_hybrid: bool = True):
        self.model = model
        self.use_hybrid = use_hybrid
        self.hybrid_service = None
        self.industry_map = {}
        self.sector_momentum_cache = {}
        self.icir_manager = None  # ICIR动态权重管理器

        # 加载行业数据
        self._load_industry_data()

        # 初始化ICIR动态权重管理器
        self._init_icir_manager()

        # 初始化CFFEX机构信号
        self._init_cffex_signal()

        # 初始化融资融券情绪信号
        self._init_margin_signal()

        if use_hybrid:
            try:
                from src.ml.hybrid_prediction import HybridPredictionService
                self.hybrid_service = HybridPredictionService()
                self.hybrid_service.load_models()
                print("混合预测服务已加载")
            except Exception as e:
                print(f"混合预测加载失败: {e}")
                self.use_hybrid = False

    def _load_industry_data(self):
        """加载行业分类数据"""
        try:
            pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
            if os.path.exists(pool_path):
                with open(pool_path, 'r') as f:
                    pool = json.load(f)
                    self.industry_map = pool.get('industries', {})
                    print(f"  [行业] 加载了 {len(self.industry_map)} 只股票的行业数据")
        except Exception as e:
            print(f"  [行业] 加载失败: {e}")

    def _init_icir_manager(self):
        """初始化ICIR动态权重管理器"""
        try:
            from src.ml.icir_weight_manager import ICIRDynamicWeightManager

            factors = ['trend', 'mean_rev', 'momentum', 'new_factor']
            base_weights = {
                'trend': 0.20,
                'mean_rev': 0.30,
                'momentum': 0.25,
                'new_factor': 0.25
            }

            self.icir_manager = ICIRDynamicWeightManager(
                factors=factors,
                base_weights=base_weights,
                window=12,  # 12周滚动窗口
                min_icir_threshold=0.3,
                max_weight=0.5,
                min_weight=0.05
            )
            print("  [ICIR] ICIR动态权重管理器已初始化")
        except Exception as e:
            print(f"  [ICIR] ICIR管理器初始化失败: {e}")
            self.icir_manager = None

    def _init_cffex_signal(self):
        """初始化CFFEX机构持仓信号"""
        try:
            from src.signals.cffex_signal import generate_signal, get_recent_data
            self.cffex_signal = generate_signal
            self.cffex_recent = get_recent_data
            print(f"  [CFFEX] 机构持仓信号已加载")
            print(f"  [CFFEX] 信号: {self.cffex_signal()['signal']}, 描述: {self.cffex_signal()['description']}")
        except Exception as e:
            print(f"  [CFFEX] 信号加载失败: {e}")
            self.cffex_signal = None
            self.cffex_recent = None

    def _init_margin_signal(self):
        """初始化融资融券情绪信号"""
        try:
            from src.signals.margin_sentiment_signal import generate_margin_signal
            self.margin_signal = generate_margin_signal
            result = self.margin_signal()
            print(f"  [Margin] 融资融券信号已加载")
            print(f"  [Margin] 信号: {result['signal']}, 融资余额: {result['margin_balance']:.2f}万亿")
        except Exception as e:
            print(f"  [Margin] 信号加载失败: {e}")
            self.margin_signal = None

    def _calculate_sector_momentum(self, data_dict: Dict[str, pd.DataFrame], lookback: int = 20) -> Dict[str, float]:
        """计算各行业动量"""
        sector_returns = {}
        sector_counts = {}

        for code, df in data_dict.items():
            if len(df) < lookback:
                continue

            industry = self.industry_map.get(code, '未知')
            if industry == '未知':
                continue

            close = df['close'].values
            ret_20d = (close[-1] - close[-lookback]) / close[-lookback] if len(close) >= lookback else 0

            if industry not in sector_returns:
                sector_returns[industry] = 0
                sector_counts[industry] = 0

            sector_returns[industry] += ret_20d
            sector_counts[industry] += 1

        # 计算行业平均动量
        sector_momentum = {}
        for industry in sector_returns:
            if sector_counts[industry] >= 2:  # 至少2只股票
                sector_momentum[industry] = sector_returns[industry] / sector_counts[industry]

        return sector_momentum

    def get_effective_weights(self, params: dict) -> Dict[str, float]:
        """
        获取有效权重（ICIR动态权重或基础权重）

        Returns:
            {因子名: 权重值}
        """
        # 如果有ICIR管理器且有足够数据，使用动态权重
        if self.icir_manager is not None:
            icirs = {f: self.icir_manager.factor_histories[f].get_icir() for f in ['trend', 'mean_rev', 'momentum', 'new_factor']}
            # 只有当ICIR数据足够时才使用动态权重
            if any(icirs.values()):
                dyn_weights = self.icir_manager.get_dynamic_weights()
                return dyn_weights

        # 回退到基础权重
        return {
            'ml': params.get('ml_weight', 0),
            'trend': params.get('trend_weight', 0.2),
            'mean_rev': params.get('mean_rev_weight', 0.3),
            'momentum': params.get('momentum_weight', 0.2),
            'new_factor': params.get('new_factor_weight', 0.1)
        }

    def calculate_signals(self, data_dict: Dict[str, pd.DataFrame], params: dict) -> Dict[str, dict]:
        """计算信号"""

        # 如果使用混合预测
        if self.use_hybrid and self.hybrid_service:
            return self._calculate_hybrid_signals(data_dict, params)

        # 回退到原始方法
        return self._calculate_legacy_signals(data_dict, params)

    def _load_news_signals(self) -> Dict[str, dict]:
        """加载新闻驱动信号"""
        news_signals = {}
        try:
            from src.news.news_agent import NewsAgent
            agent = NewsAgent(push_enabled=False)
            signals = agent.generate_signals()
            for sig in signals:
                news_signals[sig.code] = {
                    'score': sig.score,
                    'signal': sig.signal,
                    'sentiment': sig.sentiment,
                    'reason': sig.reason[:80]
                }
        except Exception as e:
            print(f"新闻信号加载失败: {e}")
        return news_signals

    def _calculate_hybrid_signals(self, data_dict: Dict[str, pd.DataFrame], params: dict) -> Dict[str, dict]:
        """使用混合预测计算信号 + 新闻驱动 + 行业轮动 + 动态阈值"""
        # 获取混合预测
        hybrid_preds = self.hybrid_service.predict_all(data_dict)

        # 计算行业动量
        sector_momentum = self._calculate_sector_momentum(data_dict)
        if sector_momentum:
            top_sectors = sorted(sector_momentum.items(), key=lambda x: -x[1])[:5]
            print(f"  [行业] 强势行业: {[(s[0][:10], f'{s[1]*100:.1f}%') for s in top_sectors]}")

        # 获取新闻信号
        news_signals = self._load_news_signals()
        print(f"  [新闻] 获取 {len(news_signals)} 条新闻信号")

        # ========== 第一步：计算所有股票的评分（不分配信号）==========
        temp_scores = {}  # code -> final_score
        temp_data = {}    # code -> 所有中间数据

        for code, df in data_dict.items():
            if len(df) < 60:
                continue

            # 计算增强特征（确保新因子存在）
            df = calculate_enhanced_features(df.copy())

            close = df['close'].values
            hybrid = hybrid_preds.get(code, {})

            # 综合分数
            ensemble = hybrid.get('ensemble', 50)

            # 获取各分量分数
            xgb_score = hybrid.get('xgb_score', 50)
            lstm_score = hybrid.get('lstm_score', 50)
            rule_score = hybrid.get('rule_score', 50)

            # ========== 趋势追踪 ==========
            ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
            ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
            rsi = self._calc_rsi(close)
            uptrend = ma20 > ma60
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)

            trend_score = 50
            if uptrend:
                trend_score += 30
            if rsi < rsi_oversold:
                trend_score += 20
            elif rsi > rsi_overbought:
                trend_score -= 20

            # ========== 均值回归 ==========
            ret_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
            mean_rev_score = 50
            if ret_20d < -0.08:
                mean_rev_score += 30
            elif ret_20d > 0.08:
                mean_rev_score -= 20

            # ========== 动量 ==========
            ret_60d = (close[-1] - close[-60]) / close[-60] if len(close) >= 60 else 0
            momentum_score = 50
            if ret_60d > 0.10:
                momentum_score += 25
            elif ret_60d < -0.10:
                momentum_score -= 25

            # ========== 新因子 ==========
            vol_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
            ret_1d = df['return_1d'].iloc[-1] if 'return_1d' in df.columns else 0
            volume_score = min(100, max(0, 50 + (vol_ratio - 1) * 20 + ret_1d * 100))

            bb_squeeze = df['bb_squeeze'].iloc[-1] if 'bb_squeeze' in df.columns else 1.0
            bb_breakout = df['bb_breakout'].iloc[-1] if 'bb_breakout' in df.columns else 0
            momentum_accel = df['momentum_accel'].iloc[-1] if 'momentum_accel' in df.columns else 0
            momentum_strength = df['momentum_strength'].iloc[-1] if 'momentum_strength' in df.columns else 0.5

            new_factor_score = 50
            if vol_ratio > 1.5 and ret_1d > 0:
                new_factor_score += 15
            if bb_squeeze < 0.8:
                new_factor_score += 10
            if bb_breakout:
                new_factor_score += 15
            if momentum_accel > 0.02:
                new_factor_score += 10
            new_factor_score += momentum_strength * 15

            # ========== 综合评分（使用ICIR动态权重）==========
            ml_score = (xgb_score + lstm_score) / 2

            # 获取有效权重（ICIR动态权重或基础权重）
            weights = self.get_effective_weights(params)

            combined = (
                ml_score * weights.get('ml', 0) +
                trend_score * weights.get('trend', 0.2) +
                mean_rev_score * weights.get('mean_rev', 0.3) +
                momentum_score * weights.get('momentum', 0.2) +
                new_factor_score * weights.get('new_factor', 0.1)
            )

            if params.get('ml_weight', 0) > 0:
                final_score = (combined + ensemble) / 2
            else:
                final_score = combined

            # ========== 新闻信号加成 ==========
            news_boost = 0
            news_reason = ''
            news_weight = params.get('news_weight', 0.05)
            if news_weight > 0 and code in news_signals:
                ns = news_signals[code]
                if ns['score'] > 70:
                    news_boost = (ns['score'] - 70) * news_weight
                    news_reason = f"📰{ns['reason'][:40]}..."

            final_score = final_score + news_boost

            # ========== CFFEX机构信号加成 ==========
            cffex_boost = 0
            cffex_description = ''
            if self.cffex_signal:
                try:
                    cffex = self.cffex_signal()
                    if cffex['signal'] == 'BULLISH':
                        cffex_boost = 3
                        cffex_description = '机构净空单增加→多头加成'
                    elif cffex['signal'] == 'BEARISH':
                        cffex_boost = -3
                        cffex_description = '机构净空单减少→空头加成'
                    if cffex_boost != 0:
                        final_score += cffex_boost
                except:
                    pass

            # ========== 融资融券情绪信号加成 ==========
            margin_boost = 0
            margin_description = ''
            if self.margin_signal:
                try:
                    margin = self.margin_signal()
                    if margin['signal'] == 'BULLISH':
                        margin_boost = 2
                        margin_description = '融资余额下降→情绪利好'
                    elif margin['signal'] == 'BEARISH':
                        margin_boost = -2
                        margin_description = '融资余额上升→情绪利空'
                    if margin_boost != 0:
                        final_score += margin_boost
                except:
                    pass

            # ========== 行业轮动加成 ==========
            sector_boost = 0
            industry = self.industry_map.get(code, '未知')
            if params.get('use_sector_rotation', True):
                if industry in sector_momentum and sector_momentum[industry] > 0:
                    max_mom = max(sector_momentum.values()) if sector_momentum else 0.01
                    sector_boost = (sector_momentum[industry] / max_mom) * 5
                    final_score += sector_boost
                elif industry in sector_momentum and sector_momentum[industry] < -0.05:
                    final_score -= 3

            # 存储评分
            temp_scores[code] = final_score
            temp_data[code] = {
                'ml_score': ml_score,
                'xgb_score': xgb_score,
                'lstm_score': lstm_score,
                'rule_score': rule_score,
                'ensemble': ensemble,
                'trend_score': trend_score,
                'mean_rev': mean_rev_score,
                'momentum': momentum_score,
                'price': float(close[-1]),
                'news_boost': news_boost,
                'news_reason': news_reason,
                'cffex_boost': cffex_boost,
                'cffex_description': cffex_description,
                'margin_boost': margin_boost,
                'margin_description': margin_description,
                'sector_boost': sector_boost,
                'industry': industry,
                'uptrend': uptrend  # Plan D: 趋势确认
            }

        # ========== 第二步：计算动态阈值 ==========
        signals = {}
        use_dynamic = params.get('use_dynamic_threshold', False)

        if use_dynamic and len(temp_scores) > 5:
            scores = np.array(list(temp_scores.values()))
            mode = params.get('dynamic_mode', 'percentile')

            if mode == 'percentile':
                # 百分位模式：更稳定，不受均值漂移影响
                buy_pct = params.get('dynamic_buy_percentile', 80)
                sell_pct = params.get('dynamic_sell_percentile', 30)
                dynamic_buy_threshold = np.percentile(scores, buy_pct)
                dynamic_sell_threshold = np.percentile(scores, sell_pct)
                print(f"  [动态阈值-百分位] 买入前{100-buy_pct}%={dynamic_buy_threshold:.1f}, 卖出后{sell_pct}%={dynamic_sell_threshold:.1f}")
            else:
                # 标准差模式：跟随均值
                score_mean = np.mean(scores)
                score_std = np.std(scores)
                buy_std = params.get('dynamic_buy_std', 0.5)
                sell_std = params.get('dynamic_sell_std', 0.3)
                dynamic_buy_threshold = score_mean + buy_std * score_std
                dynamic_sell_threshold = score_mean - sell_std * score_std
                print(f"  [动态阈值-标准差] 均值={score_mean:.1f}, 买入={dynamic_buy_threshold:.1f}, 卖出={dynamic_sell_threshold:.1f}")
        else:
            # 使用固定阈值
            dynamic_buy_threshold = params.get('buy_threshold', 55)
            dynamic_sell_threshold = params.get('sell_threshold', 42)
            print(f"  [固定阈值] 买入={dynamic_buy_threshold:.1f}, 卖出={dynamic_sell_threshold:.1f}")

        # ========== 第三步：分配信号 ==========
        require_uptrend = params.get('require_uptrend', False)

        for code, final_score in temp_scores.items():
            data = temp_data[code]
            uptrend = data.get('uptrend', True)

            # Plan D: 趋势确认过滤
            if require_uptrend and not uptrend:
                # 下降趋势中不买入,但可以持有或卖出
                if final_score < dynamic_sell_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
            else:
                # 正常信号分配
                if final_score > dynamic_buy_threshold:
                    signal = 'BUY'
                elif final_score < dynamic_sell_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'

            signals[code] = {
                'signal': signal,
                'score': final_score,
                'buy_threshold': dynamic_buy_threshold,
                'sell_threshold': dynamic_sell_threshold,
                'ml_pred': (data['ml_score'] - 50) / 100,
                'xgb_score': data['xgb_score'],
                'lstm_score': data['lstm_score'],
                'rule_score': data['rule_score'],
                'ensemble': data['ensemble'],
                'trend_score': data['trend_score'],
                'mean_rev': data['mean_rev'],
                'momentum': data['momentum'],
                'price': data['price'],
                'news_boost': data['news_boost'],
                'news_reason': data['news_reason'],
                'cffex_boost': data.get('cffex_boost', 0),
                'cffex_description': data.get('cffex_description', ''),
                'margin_boost': data.get('margin_boost', 0),
                'margin_description': data.get('margin_description', ''),
                'sector_boost': data['sector_boost'],
                'industry': data['industry']
            }

        return signals

    def _calculate_legacy_signals(self, data_dict: Dict[str, pd.DataFrame], params: dict) -> Dict[str, dict]:
        """原始信号计算 (回退方案)"""
        ml_preds = self.model.predict_all(data_dict)

        signals = {}
        for code, df in data_dict.items():
            if len(df) < 60:
                continue

            close = df['close'].values

            # ========== 策略1: 趋势追踪 ==========
            ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
            ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
            rsi = self._calc_rsi(close)
            uptrend = ma20 > ma60

            trend_score = 50
            if uptrend:
                trend_score += 30
            if rsi < 35:
                trend_score += 20
            elif rsi > 70:
                trend_score -= 20

            # ========== 策略2: 均值回归 ==========
            ret_20d = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
            mean_rev_score = 50
            if ret_20d < -0.08:
                mean_rev_score += 30
            elif ret_20d > 0.08:
                mean_rev_score -= 20

            # ========== 策略3: 动量 ==========
            ret_60d = (close[-1] - close[-60]) / close[-60] if len(close) >= 60 else 0
            momentum_score = 50
            if ret_60d > 0.10:
                momentum_score += 25
            elif ret_60d < -0.10:
                momentum_score -= 25

            # ========== ML预测 ==========
            ml_pred = ml_preds.get(code, 0)
            ml_score = 50 + ml_pred * 300

            # ========== 综合 ==========
            combined = (
                ml_score * params['ml_weight'] +
                trend_score * params['trend_weight'] +
                mean_rev_score * params['mean_rev_weight'] +
                momentum_score * params['momentum_weight']
            )

            # 信号
            if combined > params['buy_threshold'] and ml_pred > -0.02:
                signal = 'BUY'
            elif combined < params['sell_threshold'] or ml_pred < -0.05:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            signals[code] = {
                'signal': signal,
                'score': combined,
                'ml_pred': ml_pred,
                'trend_score': trend_score,
                'mean_rev': mean_rev_score,
                'momentum': momentum_score,
                'price': float(close[-1])
            }

        return signals

    def _load_live_signals(self) -> Dict[str, dict]:
        """从实时信号缓存加载信号"""
        signals_file = f'{CACHE_DIR}/live_signals.json'
        if not os.path.exists(signals_file):
            return None

        try:
            file_time = os.path.getmtime(signals_file)
            age = datetime.now().timestamp() - file_time
            if age > 3600:
                return None

            with open(signals_file, 'r') as f:
                data = json.load(f)

            signals = {}
            for item in data.get('signals', []):
                code = item.get('code')
                if not code:
                    continue
                signals[code] = {
                    'signal': item.get('signal', 'HOLD'),
                    'score': item.get('score', 50),
                    'rsi': item.get('rsi', 50),
                    'ml_pred': 0,
                    'trend_score': item.get('score', 50),
                    'mean_rev': 50,
                    'momentum': 50,
                    'price': item.get('close', 0),
                    'uptrend': item.get('uptrend', False)
                }
            return signals if signals else None
        except Exception as e:
            print(f"加载实时信号失败: {e}")
            return None

    def _calc_rsi(self, prices, period=14):
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(period).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 50
        return 100 - (100 / (1 + rs)) if not pd.isna(rs) else 50


class StrategyOptimizer:
    """策略优化器"""

    def __init__(self):
        self.current_round = 0
        self.best_score = 0
        self.best_params = DEFAULT_PARAMS.copy()
        self.experiments = []

    def load_state(self):
        if os.path.exists(EVOLUTION_LOG):
            with open(EVOLUTION_LOG, 'r') as f:
                data = json.load(f)
                self.current_round = data.get('current_round', 0)
                self.best_score = data.get('best_score', 0)
                loaded = data.get('best_params', {})
                for k, v in DEFAULT_PARAMS.items():
                    if k not in loaded:
                        loaded[k] = v
                self.best_params = loaded
        return self.best_params

    def save_state(self):
        data = {
            'current_round': self.current_round,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'experiments': self.experiments[-30:]
        }
        with open(EVOLUTION_LOG, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_params(self, base: dict = None) -> dict:
        if base is None:
            base = DEFAULT_PARAMS.copy()

        params = base.copy()
        if random.random() < 0.4:
            key = random.choice(list(DEFAULT_PARAMS.keys()))
            if key in ['top_n']:
                params[key] = random.choice([5, 8, 10, 12, 15])
            elif key in ['buy_threshold']:
                params[key] = random.uniform(55, 75)
            elif key in ['sell_threshold']:
                params[key] = random.uniform(30, 45)
            elif key in ['stop_loss', 'take_profit', 'position_size']:
                delta = params[key] * random.uniform(-0.3, 0.3)
                params[key] = max(0.01, min(0.5, params[key] + delta))
            elif key in ['ml_weight', 'trend_weight', 'mean_rev_weight', 'momentum_weight']:
                params[key] = random.uniform(0.1, 0.6)
        return params

    def evaluate_params(self, params: dict, data_dict: Dict[str, pd.DataFrame], signal_gen: MultiStrategySignal) -> float:
        try:
            signals_dict = {}
            for code, df in data_dict.items():
                if len(df) < 60:
                    continue

                close = df['close'].values
                dates = df['date'].values

                # 用不同参数计算评分
                signals = []
                scores = []

                for i in range(60, len(df)):
                    # 简化的窗口内评估
                    window_df = df.iloc[:i+1]
                    window_close = window_df['close'].values

                    ma20 = pd.Series(window_close).rolling(20).mean().iloc[-1]
                    ma60 = pd.Series(window_close).rolling(60).mean().iloc[-1]
                    rsi = self._calc_rsi(window_close)
                    uptrend = ma20 > ma60

                    trend_score = 50 + (30 if uptrend else 0) + (20 if rsi < 35 else (-20 if rsi > 70 else 0))

                    ret_20d = (window_close[-1] - window_close[-20]) / window_close[-20] if len(window_close) >= 20 else 0
                    mean_rev_score = 50 + (30 if ret_20d < -0.08 else (-20 if ret_20d > 0.08 else 0))

                    ret_60d = (window_close[-1] - window_close[-60]) / window_close[-60] if len(window_close) >= 60 else 0
                    momentum_score = 50 + (25 if ret_60d > 0.10 else (-25 if ret_60d < -0.10 else 0))

                    ml_pred = 0  # 简化
                    ml_score = 50 + ml_pred * 300

                    combined = (
                        ml_score * params['ml_weight'] +
                        trend_score * params['trend_weight'] +
                        mean_rev_score * params['mean_rev_weight'] +
                        momentum_score * params['momentum_weight']
                    )

                    if combined > params['buy_threshold']:
                        signals.append('BUY')
                        scores.append(combined)
                    elif combined < params['sell_threshold']:
                        signals.append('SELL')
                        scores.append(combined)
                    else:
                        signals.append('HOLD')
                        scores.append(combined)

                sig_df = pd.DataFrame({
                    'date': dates[60:],
                    'signal': signals,
                    'score': scores
                })
                signals_dict[code] = sig_df

            if not signals_dict:
                return 0

            from src.backtest.light_backtest import LightBacktest
            bt = LightBacktest(
                initial_capital=1000000,
                commission=0.0003,
                slippage=0.0001,
                position_size=params['position_size'],
                max_positions=int(params['top_n']),
                stop_loss=params['stop_loss'],
                take_profit=params['take_profit']
            )

            eval_data = {k: v.tail(200).reset_index(drop=True) for k, v in data_dict.items() if len(v) > 200}

            result = bt.run(eval_data, signals_dict, start_date='2025-07-01', end_date='2026-07-20')

            if result is None or result.total_trades < 5:
                return 0

            score = 0
            if result.annual_return > 0:
                score += min(result.annual_return * 25, 45)
            else:
                score += max(result.annual_return * 15, -30)

            score += min(max(result.sharpe_ratio * 8, 0), 25)
            if result.max_drawdown < 0:
                score += min(-result.max_drawdown * 20, 20)
            score += min(result.win_rate * 10, 10)

            return max(score, 0)

        except Exception as e:
            print(f"评估失败: {e}")
            return 0

    def _calc_rsi(self, prices, period=14):
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(period).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 50
        return 100 - (100 / (1 + rs)) if not pd.isna(rs) else 50

    def evolve(self, data_dict: Dict[str, pd.DataFrame], signal_gen: MultiStrategySignal, rounds: int = 30):
        print(f"\n{'='*60}")
        print("策略进化")
        print(f"{'='*60}")

        self.load_state()

        best = self.best_params.copy()
        best_score = self.best_score
        no_improve = 0

        for i in range(rounds):
            self.current_round += 1

            if random.random() < 0.2:
                params = self.generate_params()
            else:
                params = self.generate_params(best)

            print(f"\n轮次 {self.current_round}: 评估中...")
            score = self.evaluate_params(params, data_dict, signal_gen)

            print(f"  评分: {score:.1f}")

            self.experiments.append({
                'round': self.current_round,
                'score': score,
                'params': {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}
            })

            if score > best_score:
                best_score = score
                best = params.copy()
                self.best_score = best_score
                self.best_params = best.copy()
                no_improve = 0
                print(f"  ✓ 新最佳! score={score:.1f}")
            else:
                no_improve += 1
                if no_improve >= 8:
                    print("  重新随机搜索")
                    best = self.generate_params()
                    no_improve = 0

            if i > 10 and no_improve >= 10:
                print("早停")
                break

        self.best_score = best_score
        self.best_params = best.copy()
        self.save_state()

        print(f"\n最佳评分: {self.best_score:.1f}")
        return self.best_params


class EvolvingPortfolio:
    """进化型模拟盘"""

    def __init__(self):
        self.model = EnhancedModel()
        self.signal_generator = None
        self.optimizer = StrategyOptimizer()
        self.portfolio = load_portfolio()
        self.positions = load_positions()

    def initialize(self):
        if self.model.load():
            self.signal_generator = MultiStrategySignal(self.model)

    def get_current_params(self) -> dict:
        return self.optimizer.load_state()

    def run_daily(self):
        print("\n" + "=" * 60)
        print("模拟盘每日运行 (增强模型)")
        print("=" * 60)

        params = self.get_current_params()
        print(f"参数: ml_w={params['ml_weight']:.2f}, trend_w={params['trend_weight']:.2f}")

        # 尝试加载实时信号缓存
        signals = self._load_live_signals()
        if signals:
            print(f"使用实时信号: {len(signals)} 只股票")
        else:
            # 加载数据并生成信号
            pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
            if os.path.exists(pool_path):
                with open(pool_path, 'r') as f:
                    pool = json.load(f)
                    codes = pool.get('stocks', [])[:50]
            else:
                codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

            data_dict = load_stock_data(codes)

            if self.signal_generator is None:
                print("模型未加载，使用基础信号")
                signals = self.generate_basic_signals(data_dict, params)
            else:
                signals = self.signal_generator.calculate_signals(data_dict, params)

        buy_signals = {k: v for k, v in signals.items() if v['signal'] == 'BUY'}
        sell_signals = {k: v for k, v in signals.items() if v['signal'] == 'SELL'}

        print(f"买入={len(buy_signals)}, 卖出={len(sell_signals)}, 持仓={len(self.positions)}")

        self.execute_trades(buy_signals, sell_signals, params)
        self.update_portfolio()
        self.push_report(buy_signals, sell_signals, signals)

    def generate_basic_signals(self, data_dict: Dict[str, pd.DataFrame], params: dict) -> Dict[str, dict]:
        """基础信号（无ML模型时）"""
        signals = {}
        for code, df in data_dict.items():
            if len(df) < 60:
                continue

            close = df['close'].values
            ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
            ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
            rsi = self._calc_rsi(close)

            uptrend = ma20 > ma60
            score = 50 + (30 if uptrend else 0) + (20 if rsi < 35 else (-20 if rsi > 70 else 0))

            if score > params['buy_threshold']:
                signal = 'BUY'
            elif score < params['sell_threshold']:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            signals[code] = {
                'signal': signal,
                'score': score,
                'ml_pred': 0,
                'trend_score': score,
                'mean_rev': 50,
                'momentum': 50,
                'price': float(close[-1])
            }

        return signals

    def _load_live_signals(self) -> Dict[str, dict]:
        """从实时信号缓存加载信号"""
        signals_file = f'{CACHE_DIR}/live_signals.json'
        if not os.path.exists(signals_file):
            return None

        try:
            file_time = os.path.getmtime(signals_file)
            age = datetime.now().timestamp() - file_time
            if age > 3600:
                return None

            with open(signals_file, 'r') as f:
                data = json.load(f)

            signals = {}
            for item in data.get('signals', []):
                code = item.get('code')
                if not code:
                    continue
                signals[code] = {
                    'signal': item.get('signal', 'HOLD'),
                    'score': item.get('score', 50),
                    'rsi': item.get('rsi', 50),
                    'ml_pred': 0,
                    'trend_score': item.get('score', 50),
                    'mean_rev': 50,
                    'momentum': 50,
                    'price': item.get('close', 0),
                    'uptrend': item.get('uptrend', False)
                }
            return signals if signals else None
        except Exception as e:
            print(f"加载实时信号失败: {e}")
            return None

    def _calc_rsi(self, prices, period=14):
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(period).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 50
        return 100 - (100 / (1 + rs)) if not pd.isna(rs) else 50

    def execute_trades(self, buy_signals: dict, sell_signals: dict, params: dict):
        TRADE_FILE = f'{CACHE_DIR}/trade_history.json'

        # Load existing trade history
        trades = []
        if os.path.exists(TRADE_FILE):
            with open(TRADE_FILE, 'r') as f:
                trades = json.load(f)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for pos in self.positions[:]:
            if pos.code in sell_signals:
                sig = sell_signals[pos.code]
                if sig['price'] > 0:
                    pnl_pct = (sig['price'] - pos.entry_price) / pos.entry_price
                    pnl_amount = pos.shares * (sig['price'] - pos.entry_price) * 0.999
                    revenue = pos.shares * sig['price'] * 0.999
                    self.portfolio.cash += revenue

                    # Record sell trade
                    trades.append({
                        'date': now,
                        'action': 'SELL',
                        'code': pos.code,
                        'name': pos.name,
                        'price': sig['price'],
                        'shares': pos.shares,
                        'entry_price': pos.entry_price,
                        'entry_date': pos.entry_date,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'reason': sig.get('signal', 'SIGNAL')
                    })

                    print(f"卖出 {pos.name} @ {sig['price']:.2f} (盈亏: {pnl_pct*100:+.1f}%)")
                    self.positions.remove(pos)

        if buy_signals and len(self.positions) < int(params['top_n']):
            sorted_buys = sorted(buy_signals.items(), key=lambda x: x[1]['score'], reverse=True)
            for code, sig in sorted_buys:
                if sig['price'] <= 0:
                    continue
                if len(self.positions) >= int(params['top_n']):
                    break

                # 根据仓位比例计算买入股数
                target_value = self.portfolio.total_value * params.get('position_size', 0.15)
                shares = max(100, int(target_value / sig['price'] / 100) * 100)  # 至少100股，按手计算
                cost = shares * sig['price'] * 1.0004

                if cost <= self.portfolio.cash:
                    self.portfolio.cash -= cost
                    entry_price = sig['price']
                    entry_date = datetime.now().strftime('%Y-%m-%d')
                    self.positions.append(Position(
                        code=code,
                        name=get_stock_name(code),
                        shares=shares,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        current_price=entry_price,
                        pnl_pct=0
                    ))

                    # Record buy trade
                    trades.append({
                        'date': now,
                        'action': 'BUY',
                        'code': code,
                        'name': get_stock_name(code),
                        'price': entry_price,
                        'shares': shares,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'pnl_pct': 0,
                        'pnl_amount': 0,
                        'reason': sig.get('signal', 'SIGNAL')
                    })

                    print(f"买入 {get_stock_name(code)} @ {entry_price:.2f}")

        # Save trade history
        with open(TRADE_FILE, 'w') as f:
            json.dump(trades, f, ensure_ascii=False, indent=2)

    def update_portfolio(self):
        positions_value = 0
        for pos in self.positions:
            code_fmt = pos.code.replace('.', '_')
            filepath = f'{DATA_DIR}/raw/kline_{code_fmt}.csv'
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    pos.current_price = float(df['close'].iloc[-1])
                    pos.pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
                    positions_value += pos.shares * pos.current_price

        self.portfolio.total_value = self.portfolio.cash + positions_value

        self.portfolio.equity_curve.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions_value': positions_value
        })

        if len(self.portfolio.equity_curve) > 1000:
            self.portfolio.equity_curve = self.portfolio.equity_curve[-1000:]

        save_positions(self.positions)
        save_portfolio(self.portfolio)

        initial = 2000000
        ret_pct = (self.portfolio.total_value - initial) / initial

        print(f"\n账户: {self.portfolio.total_value:,.2f} ({ret_pct:+.2%})")

    def push_report(self, buy_signals: dict, sell_signals: dict, all_signals: dict):
        try:
            from src.notification.lark_notify import LarkNotifier
            from src.notification.wechat_notify import load_config

            config = load_config()
            webhook = config.get('lark_webhook', '')
            if not webhook:
                return

            notifier = LarkNotifier(webhook_url=webhook)

            initial = 2000000
            ret_pct = (self.portfolio.total_value - initial) / initial

            content = f"**量化系统每日报告**\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            content += f"**账户:** {self.portfolio.total_value:,.0f} ({ret_pct:+.1%})\n"
            content += f"**持仓:** {len(self.positions)} 只\n"
            content += f"**策略:** 增强模型+多策略\n\n"
            content += f"**信号:** 买{len(buy_signals)} 卖{len(sell_signals)}\n"

            if self.positions:
                content += f"\n**持仓:**\n"
                for pos in self.positions[:5]:
                    pnl = pos.pnl_pct * 100
                    content += f"• {pos.name}: {pnl:+.1f}%\n"

            notifier.send_card(content)
            print("\n飞书已推送")
        except Exception as e:
            print(f"推送失败: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--evolve', action='store_true')
    parser.add_argument('--rounds', type=int, default=30)
    args = parser.parse_args()

    portfolio = EvolvingPortfolio()
    portfolio.initialize()

    if args.evolve:
        pool_path = f'{DATA_DIR}/expanded_stock_pool.json'
        if os.path.exists(pool_path):
            with open(pool_path, 'r') as f:
                pool = json.load(f)
                codes = pool.get('stocks', [])[:30]
        else:
            codes = ['sh.600519', 'sz.000858', 'sh.600036', 'sh.601318']

        data_dict = load_stock_data(codes)
        print(f"加载 {len(data_dict)} 只股票")

        params = portfolio.optimizer.evolve(data_dict, portfolio.signal_generator, rounds=args.rounds)
        print(f"\n最佳参数已保存")
        print(f"参数: {json.dumps(params, indent=2)}")

    else:
        portfolio.run_daily()


if __name__ == '__main__':
    main()