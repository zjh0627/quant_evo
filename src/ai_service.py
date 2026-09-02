#!/usr/bin/env python3
"""
AI 服务 - 使用 MiniMax API (Anthropic兼容格式)
"""

import os
import anthropic
from datetime import datetime

# MiniMax API配置
MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
MINIMAX_BASE_URL = 'https://api.minimaxi.com/anthropic'
MINIMAX_MODEL = 'MiniMax-M2.7'

# 默认API Key（从app-state.json获取）
DEFAULT_API_KEY = 'sk-cp-7XuPTDGGiyYqxNwdsBWvJraPHlzrzgw9voBrttnhZbtmgO5fzofZJhHYRGGWZAFGOnj4k6mdWRAPCzLS-k0vfdbnjrZmft1kYqN_kWT8kmpuB3W3aDLllrk'


class AIService:
    """AI服务 - MiniMax"""

    def __init__(self, api_key=None):
        self.api_key = api_key or MINIMAX_API_KEY or DEFAULT_API_KEY
        self.client = None
        self.cache = {}
        if self.api_key:
            self.client = anthropic.Anthropic(
                base_url=MINIMAX_BASE_URL,
                api_key=self.api_key
            )

    def chat(self, prompt, system_prompt='', max_tokens=2000, thinking_disabled=True):
        """发送聊天请求"""
        if not self.client:
            return None, "未配置API Key"

        try:
            kwargs = {
                'model': MINIMAX_MODEL,
                'max_tokens': max_tokens,
                'system': system_prompt,
                'messages': [{"role": "user", "content": prompt}]
            }

            message = self.client.messages.create(**kwargs)
            # 处理返回内容
            text_result = []
            for block in message.content:
                if block.type == "text":
                    text_result.append(block.text.strip())
                elif block.type == "thinking":
                    pass
            return '\n'.join(text_result) if text_result else '', None
        except Exception as e:
            return None, str(e)

    def analyze_strategy(self, backtest_result: dict, current_params: dict) -> dict:
        """
        分析回测结果，生成参数修改建议
        """
        system_prompt = """你是一个量化策略优化专家。你的任务是分析回测结果，提出策略参数修改建议。

可调整的参数范围:
- min_score: 入场评分阈值 (40-80)
- rebalance_days: 调仓周期 (3-20天)
- max_positions: 最大持仓数 (3-8)
- stop_loss_pct: 止损比例 (0.03-0.15)
- take_profit_pct: 止盈比例 (0.10-0.30)

分析策略:
1. 如果年化收益低: 考虑调整入场条件或增加持仓
2. 如果回撤过大: 降低仓位或收紧止损
3. 如果夏普比率低: 可能是收益不稳定，考虑优化出场时机
4. 如果胜率低: 考虑调整选股条件或止损策略

请分析并给出JSON格式的参数修改建议:
{"param_changes": {"参数名": 新值}, "reason": "修改理由"}"""

        prompt = f"""当前策略参数: {current_params}

回测结果:
- 年化收益: {backtest_result.get('annual_return', 0):.2f}%
- 最大回撤: {backtest_result.get('max_drawdown', 0):.2f}%
- 夏普比率: {backtest_result.get('sharpe_ratio', 0):.3f}
- 胜率: {backtest_result.get('win_rate', 0):.1f}%

请分析并给出参数修改建议（只改1-2个参数）。"""

        response, error = self.chat(prompt, system_prompt, max_tokens=1000)

        if error:
            return {"param_changes": {}, "reason": f"API错误: {error}"}

        # 解析JSON响应 - 支持嵌套结构
        import re
        try:
            # 尝试匹配整个JSON对象
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = eval(json_match.group())
                if isinstance(result, dict) and 'param_changes' in result:
                    return result
        except Exception as e:
            pass

        # 尝试直接从文本提取
        changes = {}
        for param in ['min_score', 'rebalance_days', 'max_positions', 'stop_loss_pct', 'take_profit_pct']:
            pattern = rf'{param}[^0-9]*([0-9.]+)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = match.group(1)
                changes[param] = float(val) if '.' in val else int(val)

        if changes:
            return {"param_changes": changes, "reason": "从响应解析"}

        return {"param_changes": {}, "reason": "未获得有效建议"}

    def discover_factors(self, market_data: dict) -> list:
        """
        发现新因子
        """
        system_prompt = """你是一个量化因子研究专家。你的任务是从市场数据中发现有效的预测因子。

发现的因子应该:
1. 具有经济意义
2. 在历史上有效
3. 尽量与现有因子低相关

请以JSON格式输出发现的因子:
[{"name": "因子名称", "formula": "计算公式", "category": "类别", "description": "描述"}]"""

        prompt = f"""市场数据摘要:
- 股票数量: {len(market_data.get('stocks', []))}
- 时间范围: {market_data.get('date_range', 'N/A')}

请发现新的有效因子。"""

        response, error = self.chat(prompt, system_prompt, max_tokens=2000)

        if error:
            return []

        # 解析JSON响应
        import re
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return eval(json_match.group())
        except:
            pass

        return []

    def generate_trading_signal(self, stock_data: dict, factors: dict) -> dict:
        """
        生成交易信号
        """
        system_prompt = """你是一个量化交易专家。根据股票因子数据，给出交易建议。

输出格式:
{"signal": "BUY/SELL/HOLD", "confidence": 0-100, "reason": "理由", "target_price": 价格, "stop_loss": 价格}"""

        prompt = f"""股票: {stock_data.get('code')}
当前价格: {stock_data.get('close', 0)}
因子数据: {factors}

请给出交易建议。"""

        response, error = self.chat(prompt, system_prompt, max_tokens=500)

        if error:
            return {"signal": "HOLD", "confidence": 0, "reason": f"API错误: {error}"}

        import re
        try:
            json_match = re.search(r'\{[^{}]*"signal"[^{}]*\}', response, re.DOTALL)
            if json_match:
                return eval(json_match.group())
        except:
            pass

        return {"signal": "HOLD", "confidence": 0, "reason": "解析失败"}


if __name__ == '__main__':
    # 测试
    service = AIService()

    # 测试聊天
    response, error = service.chat("你好，请介绍一下自己", max_tokens=100)
    if error:
        print(f"错误: {error}")
    else:
        print(f"响应: {response[:200]}")

    # 测试策略分析
    result = service.analyze_strategy(
        {'annual_return': 15.5, 'max_drawdown': 8.2, 'sharpe_ratio': 1.2, 'win_rate': 55},
        {'min_score': 55, 'rebalance_days': 5}
    )
    print(f"\n策略分析: {result}")
