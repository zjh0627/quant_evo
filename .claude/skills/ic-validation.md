# Skill: ic-validation
# IC验证技能

## 触发条件
当添加新因子、调整因子权重、运行因子研究时自动触发

## IC验证规则

### 1. IC计算
```python
# IC (Information Coefficient) = Spearman相关系数
# IC = spearmanr(scores, future_returns).correlation

# 滚动IC计算
ic_series = []
for date in dates:
    scores = calculate_scores(all_stocks, date)
    future_returns = get_future_returns(all_stocks, date, horizon=5)
    ic = spearmanr(scores, future_returns).correlation
    ic_series.append(ic)
```

### 2. 有效因子标准
```python
# |IC| > 0.03: 有预测能力
# ICIR > 0.5: 稳定性达标
# IC > 55%: 方向一致性

def is_effective_factor(ic_mean, ic_std):
    icir = ic_mean / ic_std if ic_std > 0 else 0
    return {
        'has_predictability': abs(ic_mean) > 0.03,
        'is_stable': icir > 0.5,
        'directional': ic_mean > 0  # 正相关
    }
```

### 3. ICIR动态权重
```python
# ICIR > 1.0: 高权重 (权重 × 1.5, 上限0.5)
# ICIR 0.5-1.0: 正常权重
# ICIR < 0.3: 低权重 (权重 × 0.5, 下限0.05)

def adjust_weight(base_weight, icir):
    if icir > 1.0:
        return min(base_weight * 1.5, 0.5)
    elif icir < 0.3:
        return max(base_weight * 0.5, 0.05)
    return base_weight
```

### 4. 因子淘汰
```python
# 每周剔除ICIR < 0.3的因子
# 保留ICIR >= 0.3的因子
ineffective_factors = [f for f, icir in factor_icirs.items() if icir < 0.3]
```

## 已知IC值（2026-08验证）
| 因子 | IC值 | ICIR | 权重 |
|------|------|------|------|
| 均值回归 | 0.46 | 高 | 30% |
| 趋势 | 0.23 | 中 | 20% |
| 动量 | 0.13 | 低 | 25% |
| 新因子 | - | - | 25% |

## 输出格式
```json
{
  "ic_mean": 0.15,
  "ic_std": 0.08,
  "icir": 1.875,
  "ic_positive_rate": 0.65,
  "effective": true,
  "recommendation": "KEEP / UPGRADE / REMOVE"
}
```

## 执行函数
```python
from src.ml.icir_weight_manager import ICIRDynamicWeightManager
icir_mgr = ICIRDynamicWeightManager(factors=['trend', 'mean_rev', 'momentum', 'new_factor'])
icir_mgr.add_ic('2026-09-14', 'trend', 0.23)
dynamic_weights = icir_mgr.get_dynamic_weights()
```
