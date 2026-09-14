# Skill: data-quality
# 数据质量检查技能

## 触发条件
当处理K线数据、计算因子、加载股票池时自动触发

## 检查规则

### 1. 停牌数据检测
```python
# 停牌判断条件
is_suspended = (volume == 0) or (all_prices_equal)
```
- 停牌日期需跳过，不参与因子计算
- 停牌期间不产生交易信号

### 2. 涨跌停标记
```python
# 涨停：close == high and close > prev_close * 1.099
# 跌停：close == low and close < prev_close * 0.901
```
- 涨停日不产生买入信号（无法买入）
- 跌停日不产生卖出信号（无法卖出）

### 3. 数据完整性
- 缺失值处理：使用前向填充(ffill)
- 异常值检测：收盘价变化超过20%需人工确认
- 日期连续性：检查是否有交易日缺失

### 4. 复权一致性
- 全部使用前复权数据
- 不混用复权方式
- 成交量的复权处理

## 输出格式
```json
{
  "quality_score": 0-100,
  "issues": [
    {"type": "suspended", "date": "2026-01-01", "code": "sh.600519"},
    {"type": "limit_up", "date": "2026-01-01", "code": "sh.600519"}
  ],
  "recommendation": "PASS / CLEAN / REJECT"
}
```

## 执行函数
```python
from src.data.data_cleaner import DataCleaner
cleaner = DataCleaner()
result = cleaner.check_quality(df, code)
```
