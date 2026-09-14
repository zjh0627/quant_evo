# Skill: risk-management
# 风险管理技能

## 触发条件
当执行交易、调整仓位、触发止损止盈时自动触发

## 风险规则

### 1. 仓位控制
```python
# 单只股票仓位上限
max_position_per_stock = 0.20  # 20%

# 总仓位上限
max_total_position = 0.80  # 80%

# 最小持仓数
min_positions = 1
max_positions = 8  # 分散化
```

### 2. 止损止盈
```python
# 止损规则
if pnl_pct <= -0.10:  # 亏损10%
    trigger_stop_loss()

# 止盈规则
if pnl_pct >= 0.20:  # 盈利20%
    trigger_take_profit()

# 移动止损（可选）
if current_price < peak_price * (1 - trailing_stop_pct):
    trigger_trailing_stop()
```

### 3. 行业分散
```python
# 单一行业最大持仓
max_sector_exposure = 0.30  # 30%

# 强相关行业互斥
# 已持有"煤炭"，则不买"石油石化"、"钢铁"
```

### 4. 防回转机制
```python
# 卖出后冷却期
sell_cooldown_days = 5  # 5个交易日

# 止损后观察期
stop_loss_cooldown_days = 10  # 10个交易日

# 最小持仓天数
min_hold_days = 3
```

## 风控检查流程
```python
def check_risk(position, portfolio, params):
    checks = {
        'position_size': position_value / total_value <= params['max_position_per_stock'],
        'total_position': total_position_value / total_value <= params['max_total_position'],
        'sector_limit': sector_exposure <= params['max_sector_exposure'],
        'not_in_cooldown': code not in cooldown_list,
        'min_hold_days': hold_days >= params['min_hold_days']
    }
    return all(checks.values())
```

## 输出格式
```json
{
  "can_buy": true,
  "can_sell": true,
  "warnings": [],
  "blocked": false,
  "block_reason": null
}
```
