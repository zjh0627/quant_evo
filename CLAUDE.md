# quant_evo - AI原生量化交易系统

## 项目概述

quant_evo 是一个基于机器学习的A股量化交易系统，支持多因子选股、策略回测和模拟交易。

**核心定位**: AI Agent 自主研究、发现因子、优化策略，输出交易信号（用户自主决策，非自动交易）

## 技术栈

- **数据源**: baostock（A股日线）
- **数据库**: DolphinDB（备用）
- **AI**: MiniMax API
- **Web**: Flask (端口5124)
- **语言**: Python 3.9

## 项目结构

```
quant_evo/
├── src/
│   ├── ml/              # 机器学习模型 (LSTM, XGBoost, 混合预测)
│   ├── signals/         # 信号生成模块
│   ├── factors/         # 因子计算
│   ├── backtest/        # 回测引擎
│   ├── simulation/      # 模拟交易
│   ├── strategies/      # 交易策略
│   ├── analysis/        # 分析服务
│   ├── data/            # 数据管道
│   └── notification/    # 通知模块
├── scripts/             # 工具脚本
├── data/                # 数据目录
│   ├── cache/           # 缓存（持仓、信号）
│   ├── models/          # 模型文件
│   └── expanded_stock_pool.json  # 股票池
├── templates/           # Web UI模板
└── tests/               # 测试用例
```

## 代码规范

### Python代码规范
- 使用类型提示（type hints）
- docstring格式：Google style
- 导入顺序：标准库 → 第三方库 → 本地模块

### 命名规范
- 类名：CapWords（例：`MultiStrategySignal`）
- 函数名：snake_case（例：`calculate_enhanced_features`）
- 常量：UPPER_SNAKE_CASE（例：`INITIAL_CASH`）

### 关键常量
```python
INITIAL_CASH = 2000000.0  # 初始资金200万
DEFAULT_STOP_LOSS = 0.10  # 止损10%
DEFAULT_TAKE_PROFIT = 0.20  # 止盈20%
```

## 数据规范

### 股票代码格式
- 上交所：`sh.600519`
- 深交所：`sz.000858`

### K线数据字段
```
date, code, open, high, low, close, volume, amount, turn, pctChg
```

### 行业分类
- 使用申万一级行业（31个行业）
- 行业映射：`src/simulation/industry_mapping.py`

## 因子规范

### 因子分类
1. **技术因子**: RSI, MACD, KDJ, CCI, ATR, ADX, 布林带
2. **趋势因子**: MA(5/10/20/60), 均线多头排列
3. **动量因子**: 1/5/10/20日动量
4. **资金流因子**: 主力净流入, 散户恐慌指数
5. **情绪因子**: 涨停/跌停次数, 换手率异动

### 评分体系
- 综合评分范围：0-100
- 买入阈值：55（可动态调整）
- 卖出阈值：42（可动态调整）
- IC验证阈值：|IC| > 0.03, ICIR > 0.5

## 策略参数

### 当前最优参数
```python
top_n = 8                # 持仓分散化
buy_threshold = 55       # 买入阈值
sell_threshold = 42      # 卖出阈值
stop_loss = 0.10         # 止损10%
take_profit = 0.20       # 止盈20%
position_size = 0.10     # 单只仓位10%
```

### 因子权重（ICIR验证后）
- 均值回归：30%（IC=0.46）
- 趋势：20%（IC=0.23）
- 动量：25%（IC=0.13）
- 新因子：25%

## 常见错误与陷阱

### 1. 回测避坑
- **不使用未来数据**：计算因子时只用当日及之前的数据
- **复权一致**：统一使用前复权数据
- **滑点成本**：每次交易考虑0.1%滑点

### 2. 数据处理
- **停牌过滤**：`volume == 0` 的日期需跳过
- **涨跌停标记**：涨跌停日不能买入
- **缺失值处理**：使用前向填充

### 3. 模拟交易
- **持仓上限**：单一股票不超过总仓位20%
- **总仓位上限**：不超过80%
- **止损优先**：触发止损时优先于其他信号

### 4. 行业轮动
- **强相关行业**：已持有某行业则不买其强相关行业
- **强势行业加成**：动量前5行业+5分
- **非强势过滤**：不在前5行业评分-10

## API端点规范

### Web UI（端口5124）
- `GET /` - 主页
- `GET /portfolio` - 模拟盘Dashboard
- `GET /api/portfolio` - 持仓状态
- `GET /api/signals` - 交易信号
- `GET /api/signals/detailed` - 详细信号
- `GET /api/trade_history` - 交易历史
- `POST /api/reset` - 重置账户
- `GET /api/market/indices` - 大盘指数
- `GET /api/market/sectors` - 板块热点

## 文件路径约定

| 类型 | 路径 |
|------|------|
| 缓存数据 | `data/cache/` |
| 模型文件 | `data/models/` |
| K线数据 | `data/cache/kline_{code}.parquet` |
| 股票池 | `data/expanded_stock_pool.json` |
| 持仓 | `data/cache/positions.json` |
| 交易历史 | `data/cache/real_time_trades.json` |

## 开发流程

### 1. Plan阶段
- 创建 `intent.md` 描述需求
- 分析可行性和风险
- 设定验收标准

### 2. Design阶段
- 创建 `spec.md` 详细设计
- 编写技术方案
- 定义接口和数据结构

### 3. Build阶段
- 使用 Plan Mode 先计划后实现
- 遵循代码规范
- 同时编写测试用例

### 4. Test阶段
- 运行回测验证
- 检查IC验证
- 对比基准收益

### 5. Deploy阶段
- 创建PR并评审
- 更新todo.md状态
- 提交到git

### 6. Maintain阶段
- 监控模拟盘运行
- 记录事故和异常
- 触发新的intent.md

## 与Claude Code协作

### 常用命令
```bash
# 启动模拟盘
python -m src.simulation.evolving_portfolio

# 运行回测
python scripts/run_backtest.py

# 重置账户
python scripts/reset_portfolio.py
```

### 任务执行流程
1. 用户提出需求 → 我思考方案
2. 拆分任务 → 记录到 todo.md
3. 执行任务 → 更新文档
4. 测试验证 → 功能通过
5. 提交代码 → git push
6. 更新状态 → todo.md标记完成
7. 继续下一个任务

### 完成定义
单个任务完成的判断标准：
1. 功能开发完成
2. 测试通过
3. 代码提交到git (origin/main)
4. todo.md状态更新
