#!/usr/bin/env python3
"""
微信通知模块
使用企业微信 Webhook 推送交易信号
"""

import requests
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

PROJECT_ROOT = '/Users/keira/project/claude/quant_evo'
CONFIG_FILE = f'{PROJECT_ROOT}/data/cache/notify_config.json'


def load_config() -> Dict:
    """加载配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: Dict):
    """保存配置"""
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class WeChatNotifier:
    """企业微信通知器"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
        if not self.webhook_url:
            config = load_config()
            self.webhook_url = config.get('wechat_webhook', '')

    def send_text(self, content: str, mentioned_list: List[str] = None) -> bool:
        """发送文本消息"""
        if not self.webhook_url:
            print("未配置 Webhook URL")
            return False

        payload = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list or []
            }
        }

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            result = resp.json()
            return result.get('errcode', 1) == 0
        except Exception as e:
            print(f"发送失败: {e}")
            return False

    def send_signal_alert(self, signals: List[Dict], signal_type: str = 'BUY') -> bool:
        """发送信号预警

        Args:
            signals: 信号列表
            signal_type: 信号类型 BUY/SELL
        """
        if not signals:
            return False

        # 按评分排序取前5
        top_signals = sorted(signals, key=lambda x: x.get('score', 0), reverse=True)[:5]

        content = f"📊 {signal_type} 信号预警 ({datetime.now().strftime('%H:%M:%S')})\n\n"

        for s in top_signals:
            name = s.get('name', s.get('code', ''))
            code = s.get('code', '')
            price = s.get('close', 0)
            score = s.get('score', 0)
            reason = s.get('reason', '')
            content += f"▶ {name} ({code})\n"
            content += f"   价格: {price:.2f} | 评分: {score:.1f}\n"
            content += f"   原因: {reason}\n\n"

        content += f"共 {len(signals)} 只股票发出 {signal_type} 信号"

        return self.send_text(content)

    def send_trade_signal_report(self, buy_count: int, sell_count: int, hold_count: int, top_buys: List[Dict] = None, top_sells: List[Dict] = None) -> bool:
        """发送交易信号报告"""
        content = f"📈 量化系统信号报告\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        content += f"买入信号: {buy_count} 只\n"
        content += f"卖出信号: {sell_count} 只\n"
        content += f"观望信号: {hold_count} 只\n\n"

        if top_buys:
            content += "🔥 TOP 5 买入信号:\n"
            for i, s in enumerate(top_buys[:5], 1):
                content += f"{i}. {s.get('name', s.get('code', ''))} ({s.get('code', '')}) {s.get('close', 0):.2f}\n"
            content += "\n"

        if top_sells:
            content += "⚠️ TOP 5 卖出信号:\n"
            for i, s in enumerate(top_sells[:5], 1):
                content += f"{i}. {s.get('name', s.get('code', ''))} ({s.get('code', '')}) {s.get('close', 0):.2f}\n"

        return self.send_text(content)


def main():
    """测试推送"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--webhook', type=str, help='企业微信 webhook URL')
    parser.add_argument('--test', action='store_true', help='发送测试消息')
    args = parser.parse_args()

    notifier = WeChatNotifier(webhook_url=args.webhook)

    if args.test:
        notifier.send_text("🧪 量化系统通知测试\n连接正常！")


if __name__ == '__main__':
    main()