# Skill: backtest-validation
# 回测验证技能

## 触发条件
当运行回测、验证策略、计算绩效指标时自动触发

## 检查规则

### 1. 未来函数检测
```python
# 禁止模式：使用当日收盘后数据
# 正确：使用截至当日的数据计算因子
# 错误：因子计算中使用了当日收盘价计算的收益
```
- 回测必须使用"事后诸葛亮"之前的因子数据
- 信号生成必须在收盘后

### 2. 滑点成本
```python
# 买入：成交价 * (1 + 0.001)  # 0.1%滑点
# 卖出：成交价 * (1 - 0.001)
```
- 每次交易考虑0.1%滑点
- 印花税0.1%（卖出）
- 佣金0.03%

### 3. 绩效指标计算
```python
# 年化收益 = (总收益 + 1) ^ (252/交易天数) - 1
# 夏普比率 = (策略收益 - 无风险收益) / 策略收益标准差
# 最大回撤 = max(peak - trough) / peak
```

### 4. 基准对比
- 必须有基准对比（沪深300或同等指数）
- 阿尔法 = 策略收益 - 基准收益
- 信息比率 = 阿尔法 / 跟踪误差

## 输出格式
```json
{
  "annual_return": 0.15,
  "sharpe_ratio": 1.5,
  "max_drawdown": -0.12,
  "win_rate": 0.55,
  "profit_loss_ratio": 1.8,
  "issues": [
    {"type": "future_leak", "severity": "HIGH"}
  ],
  "recommendation": "PASS / FAIL"
}
```

## 执行函数
```python
from src.backtest.light_backtest import LightBacktest
bt = LightBacktest()
result = bt.run(strategy, data, params)
```
