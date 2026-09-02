#!/usr/bin/env python3
"""获取中信期货持仓数据"""
import sys
sys.path.insert(0, '/Users/keira/project/claude/quant_evo')

import akshare as ak
import warnings
warnings.filterwarnings('ignore')

def get_citic_position(date, contract):
    """获取指定日期和合约的中信期货净持仓"""
    try:
        df = ak.get_cffex_rank_table(date=date, vars_list=[contract])
        if contract not in df or not df[contract]:
            return None

        symbol = list(df[contract].keys())[0]
        d = df[contract][symbol]

        cl = d[d['long_party_name'].str.contains('中信', na=False)]['long_open_interest'].sum()
        cs = d[d['short_party_name'].str.contains('中信', na=False)]['short_open_interest'].sum()
        clc = d[d['long_party_name'].str.contains('中信', na=False)]['long_open_interest_chg'].sum()
        csc = d[d['short_party_name'].str.contains('中信', na=False)]['short_open_interest_chg'].sum()

        net = cs - cl
        return {
            'contract': contract,
            'symbol': symbol,
            'long': cl,
            'short': cs,
            'long_chg': clc,
            'short_chg': csc,
            'net': net,
            'direction': '净空' if net > 0 else '净多'
        }
    except Exception as e:
        print(f"  错误: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else '2026-08-31'

    print(f"=" * 60)
    print(f"中信期货净持仓 {date}")
    print(f"=" * 60)

    contracts = [
        ('IF', '沪深300'),
        ('IH', '上证50'),
        ('IC', '中证500'),
        ('IM', '中证1000')
    ]

    for contract, name in contracts:
        print(f"\n{contract} {name}:")
        result = get_citic_position(date, contract)
        if result:
            print(f"  主力合约: {result['symbol']}")
            print(f"  多单: {result['long']:,} 手 (较前日 {result['long_chg']:+,} 手)")
            print(f"  空单: {result['short']:,} 手 (较前日 {result['short_chg']:+,} 手)")
            print(f"  净持仓: {abs(result['net']):,} 手 {result['direction']}")
        else:
            print(f"  无数据")