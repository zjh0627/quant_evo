#!/usr/bin/env python3
"""
Agent 基类
所有 Agent 的父类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Message:
    """Agent 间消息"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    msg_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_dict(self) -> Dict:
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'msg_type': self.msg_type,
            'timestamp': self.timestamp
        }


class Agent(ABC):
    """Agent 基类"""

    def __init__(self, name: str):
        self.name = name
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def process(self, message: Optional[Message] = None) -> Optional[Message]:
        """
        处理消息，返回响应消息（如果有）
        """
        pass

    def receive(self, message: Message):
        """接收消息"""
        self.inbox.append(message)

    def send(self, receiver: str, content: Dict[str, Any], msg_type: str):
        """发送消息"""
        msg = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type
        )
        self.outbox.append(msg)
        return msg

    def clear_inbox(self):
        """清空收件箱"""
        self.inbox.clear()

    def clear_outbox(self):
        """清空发件箱"""
        self.outbox.clear()

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any):
        """设置状态"""
        self.state[key] = value

    def update_state(self, updates: Dict[str, Any]):
        """批量更新状态"""
        self.state.update(updates)

    def to_dict(self) -> Dict:
        """导出 Agent 信息"""
        return {
            'name': self.name,
            'state': self.state,
            'inbox_count': len(self.inbox),
            'outbox_count': len(self.outbox)
        }

    def __repr__(self) -> str:
        return f"<Agent: {self.name}>"
