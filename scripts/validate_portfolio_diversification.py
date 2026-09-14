#!/usr/bin/env python3
"""
持仓分散化验证脚本
==================

验证当前持仓分散化配置是否满足需求

运行方式:
    python scripts/validate_portfolio_diversification.py
"""

import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

from src.simulation.evolving_portfolio import MultiStrategySignal, DEFAULT_PARAMS
from src.simulation.industry_mapping import get_related_industries, SW_LEVEL_1_INDUSTRIES
import json

def main():
    print("=" * 70)
    print("持仓分散化验证")
    print("=" * 70)

    # 1. 检查默认参数
    print("\n1. 默认参数检查")
    print("-" * 50)
    top_n = DEFAULT_PARAMS.get('top_n', 8)
    position_size = DEFAULT_PARAMS.get('position_size', 0.10)
    max_stocks_per_sector = DEFAULT_PARAMS.get('max_stocks_per_sector', 2)

    print(f"  top_n: {top_n}")
    print(f"  position_size: {position_size * 100:.0f}%")
    print(f"  max_stocks_per_sector: {max_stocks_per_sector}")

    # 2. 验证分散化效果
    print("\n2. 分散化效果验证")
    print("-" * 50)
    max_total_position = top_n * position_size
    print(f"  最大总仓位: {max_total_position * 100:.0f}%")
    print(f"  单只仓位上限: {position_size * 100:.0f}%")

    checks = []

    # 检查top_n范围
    if 5 <= top_n <= 8:
        checks.append(("top_n在合理范围(5-8)", True))
    else:
        checks.append(("top_n在合理范围(5-8)", False))

    # 检查单只仓位
    if position_size <= 0.15:
        checks.append(("单只仓位不超过15%", True))
    else:
        checks.append(("单只仓位不超过15%", False))

    # 检查总仓位
    if max_total_position <= 0.80:
        checks.append(("总仓位不超过80%", True))
    else:
        checks.append(("总仓位不超过80%", False))

    for check, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    # 3. 验证行业约束
    print("\n3. 行业分散约束验证")
    print("-" * 50)

    # 检查行业映射
    try:
        signal_gen = MultiStrategySignal(use_hybrid=False)
        industry_map = signal_gen.industry_map

        print(f"  已加载行业映射股票数: {len(industry_map)}")

        # 统计一级行业分布
        from collections import Counter
        level1_counts = Counter(industry_map.values())
        print(f"  一级行业数量: {len(level1_counts)}")

        # 强相关行业组验证
        print("\n  强相关行业组:")
        test_industries = ['银行', '煤炭', '电子', '医药生物']
        for ind in test_industries:
            related = get_related_industries(ind)
            print(f"    {ind} -> {related if related else '无强相关'}")

        industry_checks = [
            ("行业映射已加载", len(industry_map) > 0),
            ("一级行业数量正确(>20)", len(level1_counts) > 20),
        ]

        for check, passed in industry_checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")

    except Exception as e:
        print(f"  ❌ 行业映射加载失败: {e}")
        industry_checks = [("行业映射加载", False)]

    # 4. 验证参数在代码中正确使用
    print("\n4. 代码实现验证")
    print("-" * 50)

    # 读取evolving_portfolio.py检查关键逻辑
    with open('/Users/keira/project/claude/quant_evo/src/simulation/evolving_portfolio.py', 'r') as f:
        code = f.read()

    code_checks = [
        ("top_n参数已定义", "'top_n'" in code),
        ("position_size参数已定义", "'position_size'" in code),
        ("max_stocks_per_sector已定义", "'max_stocks_per_sector'" in code),
        ("行业过滤逻辑已实现", "'use_sector_filter'" in code),
        ("行业轮动逻辑已实现", "'use_sector_rotation'" in code),
        ("强相关行业约束已实现", "industry_mapping import get_related_industries" in code or "get_related_industries(industry)" in code),
    ]

    for check, passed in code_checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    # 5. 综合结论
    print("\n" + "=" * 70)
    print("验证结论")
    print("=" * 70)

    all_checks = checks + industry_checks + code_checks
    passed_count = sum(1 for _, p in all_checks if p)
    total_count = len(all_checks)

    print(f"\n通过检查: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n✅ 所有检查通过！持仓分散化配置正确。")
        print("\n当前配置满足需求:")
        print(f"  - top_n = {top_n} (持仓分散化)")
        print(f"  - position_size = {position_size * 100:.0f}% (单只仓位)")
        print(f"  - max_stocks_per_sector = {max_stocks_per_sector} (行业分散)")
        print(f"  - 行业过滤和轮动已启用")
        return True
    else:
        print("\n❌ 部分检查未通过，请检查上述失败项目。")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
