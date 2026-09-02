#!/usr/bin/env python3
"""
quant_evo Multi-Agent System
"""

from src.agents.base import Agent, Message
from src.agents.orchestrator import Orchestrator, SimpleOrchestrator
from src.agents.factor_discovery import FactorDiscoveryAgent
from src.agents.strategy_composer import StrategyComposerAgent
from src.agents.risk_control import RiskControlAgent
from src.agents.simulation import SimulationAgent
from src.agents.news_analysis import NewsAnalysisAgent

__all__ = [
    'Agent',
    'Message',
    'Orchestrator',
    'SimpleOrchestrator',
    'FactorDiscoveryAgent',
    'StrategyComposerAgent',
    'RiskControlAgent',
    'SimulationAgent',
    'NewsAnalysisAgent',
]
