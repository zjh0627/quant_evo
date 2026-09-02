#!/usr/bin/env python3
"""
Orchestrator - 多Agent系统协调器
管理Agent间通信、状态维护、决策循环
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.agents.base import Agent, Message

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
STATE_FILE = f'{PROJECT_ROOT}/data/cache/system_state.json'


class Orchestrator:
    """协调器 - 管理整个Agent系统"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.global_state: Dict[str, Any] = {
            'status': 'idle',
            'current_round': 0,
            'best_score': 0,
            'best_params': {},
            'research_results': [],
            'last_update': None
        }
        self.message_log: List[Message] = []
        self.load_state()

    def register_agent(self, agent: Agent):
        """注册Agent"""
        self.agents[agent.name] = agent
        print(f"✓ Agent注册: {agent.name}")

    def unregister_agent(self, name: str):
        """注销Agent"""
        if name in self.agents:
            del self.agents[name]
            print(f"✗ Agent注销: {name}")

    def send_message(self, from_agent: str, to_agent: str,
                    content: Dict[str, Any], msg_type: str):
        """发送消息"""
        if from_agent in self.agents:
            msg = self.agents[from_agent].send(to_agent, content, msg_type)
            self.message_log.append(msg)

            # 如果目标Agent存在，直接传递
            if to_agent in self.agents:
                self.agents[to_agent].receive(msg)

    def broadcast(self, from_agent: str, content: Dict[str, Any], msg_type: str):
        """广播消息给所有Agent"""
        for agent_name in self.agents:
            if agent_name != from_agent:
                self.send_message(from_agent, agent_name, content, msg_type)

    def get_agent_state(self, agent_name: str) -> Optional[Dict]:
        """获取Agent状态"""
        if agent_name in self.agents:
            return self.agents[agent_name].to_dict()
        return None

    def get_all_states(self) -> Dict[str, Any]:
        """获取所有状态"""
        return {
            'orchestrator': self.global_state,
            'agents': {name: agent.to_dict() for name, agent in self.agents.items()},
            'message_count': len(self.message_log)
        }

    def update_global_state(self, updates: Dict[str, Any]):
        """更新全局状态"""
        self.global_state.update(updates)
        self.global_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_state()

    def save_state(self):
        """保存状态到文件"""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.global_state, f, indent=2, ensure_ascii=False)

    def load_state(self):
        """加载状态"""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                self.global_state = json.load(f)

    def run_research_cycle(self) -> Dict[str, Any]:
        """运行一轮研究循环"""
        print(f"\n{'='*60}")
        print(f"研究循环 Round {self.global_state['current_round'] + 1}")
        print(f"{'='*60}")

        self.update_global_state({
            'status': 'running',
            'current_round': self.global_state['current_round'] + 1
        })

        # 1. 新闻分析 Agent 收集舆情
        if 'NewsAnalysis' in self.agents:
            self.send_message('Orchestrator', 'NewsAnalysis',
                            {'action': 'analyze'}, 'REQUEST')
            self.agents['NewsAnalysis'].process()

        # 2. 因子挖掘 Agent 发现因子
        if 'FactorDiscovery' in self.agents:
            self.send_message('Orchestrator', 'FactorDiscovery',
                            {'action': 'discover'}, 'REQUEST')
            self.agents['FactorDiscovery'].process()

        # 3. 策略组合 Agent 优化策略
        if 'StrategyComposer' in self.agents:
            self.send_message('Orchestrator', 'StrategyComposer',
                            {'action': 'compose'}, 'REQUEST')
            self.agents['StrategyComposer'].process()

        # 4. 风控 Agent 评估风险
        if 'RiskControl' in self.agents:
            self.send_message('Orchestrator', 'RiskControl',
                            {'action': 'evaluate'}, 'REQUEST')
            self.agents['RiskControl'].process()

        # 5. 模拟 Agent 运行回测
        if 'Simulation' in self.agents:
            self.send_message('Orchestrator', 'Simulation',
                            {'action': 'backtest'}, 'REQUEST')
            result = self.agents['Simulation'].process()

            if result:
                score = result.content.get('score', 0)
                if score > self.global_state.get('best_score', 0):
                    self.update_global_state({
                        'best_score': score,
                        'best_params': result.content.get('params', {})
                    })
                    print(f"✓ 新最优评分: {score}")

        self.update_global_state({'status': 'idle'})
        return self.global_state

    def print_status(self):
        """打印系统状态"""
        print(f"\n{'='*60}")
        print("系统状态")
        print(f"{'='*60}")
        print(f"状态: {self.global_state.get('status', 'unknown')}")
        print(f"当前轮次: {self.global_state.get('current_round', 0)}")
        print(f"最优评分: {self.global_state.get('best_score', 0)}")
        print(f"\n已注册Agent:")
        for name in self.agents:
            print(f"  - {name}")
        print(f"\n消息数: {len(self.message_log)}")
        print(f"最后更新: {self.global_state.get('last_update', 'N/A')}")
        print(f"{'='*60}")


class SimpleOrchestrator:
    """简化版协调器 - 用于standalone模式"""

    def __init__(self):
        self.state = {
            'research_round': 0,
            'best_score': 0,
            'best_params': {},
            'system_status': 'ready'
        }

    def start_research(self, rounds: int = 10):
        """启动研究"""
        self.state['system_status'] = 'running'

        for i in range(rounds):
            self.state['research_round'] = i + 1
            print(f"\n--- Round {i+1}/{rounds} ---")

            # 模拟各Agent工作
            self._run_news_analysis()
            self._run_factor_discovery()
            self._run_strategy_composer()
            self._run_risk_control()
            result = self._run_simulation()

            if result and result.get('score', 0) > self.state['best_score']:
                self.state['best_score'] = result['score']
                self.state['best_params'] = result.get('params', {})

        self.state['system_status'] = 'completed'
        return self.state

    def _run_news_analysis(self):
        """模拟新闻分析"""
        print("  [NewsAnalysis] 分析舆情...")

    def _run_factor_discovery(self):
        """模拟因子发现"""
        print("  [FactorDiscovery] 发现因子...")

    def _run_strategy_composer(self):
        """模拟策略组合"""
        print("  [StrategyComposer] 组合策略...")

    def _run_risk_control(self):
        """模拟风控"""
        print("  [RiskControl] 评估风控...")

    def _run_simulation(self) -> Optional[Dict]:
        """模拟回测"""
        print("  [Simulation] 运行回测...")
        # 这里会调用实际的回测模块
        return None

    def get_status(self) -> Dict:
        """获取状态"""
        return self.state


if __name__ == '__main__':
    # 测试协调器
    orch = SimpleOrchestrator()
    orch.state['best_score'] = 87.7
    orch.state['best_params'] = {
        'min_score': 55,
        'rebalance_days': 3,
        'max_positions': 5
    }
    print(orch.get_status())
