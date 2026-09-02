#!/usr/bin/env python3
"""
CSV to Parquet 格式转换脚本
===========================

将 data/cache 和 data/raw 中的 CSV 文件转换为 Parquet 格式

使用方法:
    python scripts/convert_to_parquet.py --convert     # 转换
    python scripts/convert_to_parquet.py --verify      # 验证
    python scripts/convert_to_parquet.py --cleanup     # 转换后删除 CSV
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
BACKUP_DIR = PROJECT_ROOT / 'data' / 'csv_backup'

sys.path.insert(0, str(PROJECT_ROOT))


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> bool:
    """转换单个 CSV 文件为 Parquet"""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df.to_parquet(parquet_path, index=False, compression='snappy')
        return True
    except Exception as e:
        print(f"  转换失败: {e}")
        return False


def convert_directory(dir_path: Path, dry_run: bool = False) -> dict:
    """转换目录中的所有 CSV 文件"""
    import pandas as pd

    csv_files = list(dir_path.glob('*.csv'))
    if not csv_files:
        print(f"  目录为空: {dir_path}")
        return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    results = {'total': len(csv_files), 'success': 0, 'failed': 0, 'skipped': 0}
    total_size_csv = 0
    total_size_parquet = 0

    for csv_file in sorted(csv_files):
        parquet_file = csv_file.with_suffix('.parquet')

        # 计算大小
        csv_size = csv_file.stat().st_size
        total_size_csv += csv_size

        # 检查是否已转换
        if parquet_file.exists():
            print(f"  跳过 (已存在): {csv_file.name}")
            results['skipped'] += 1
            total_size_parquet += parquet_file.stat().st_size
            continue

        if dry_run:
            print(f"  [dry-run] 将转换: {csv_file.name}")
            continue

        # 转换
        print(f"  转换: {csv_file.name} ...", end=' ', flush=True)
        if csv_to_parquet(csv_file, parquet_file):
            parquet_size = parquet_file.stat().st_size
            total_size_parquet += parquet_size
            ratio = parquet_size / csv_size if csv_size > 0 else 0
            print(f"✓ ({csv_size/1024:.1f}KB → {parquet_size/1024:.1f}KB, 压缩比 {ratio:.1%})")
            results['success'] += 1
        else:
            results['failed'] += 1

    # 统计
    print(f"\n  总计: {results['success']} 成功, {results['skipped']} 跳过, {results['failed']} 失败")
    print(f"  大小: {total_size_csv/1024/1024:.2f}MB (CSV) → {total_size_parquet/1024/1024:.2f}MB (Parquet)")

    return results


def verify_parquet(dir_path: Path) -> dict:
    """验证 Parquet 文件"""
    import pandas as pd

    parquet_files = list(dir_path.glob('*.parquet'))
    csv_files = {f.with_suffix('.csv').name for f in parquet_files}

    results = {'total': len(parquet_files), 'valid': 0, 'invalid': 0, 'missing_csv': 0}

    for pq_file in sorted(parquet_files):
        csv_file = pq_file.with_suffix('.csv')

        try:
            df = pd.read_parquet(pq_file)

            # 基本验证
            if df.empty:
                print(f"  空文件: {pq_file.name}")
                results['invalid'] += 1
                continue

            # 对比 CSV
            if csv_file.exists():
                df_csv = pd.read_csv(csv_file, parse_dates=['date'])
                if len(df) != len(df_csv):
                    print(f"  行数不一致: {pq_file.name} (Parquet: {len(df)}, CSV: {len(df_csv)})")
                    results['invalid'] += 1
                else:
                    print(f"  ✓ 验证通过: {pq_file.name} ({len(df)} 行)")
                    results['valid'] += 1
            else:
                print(f"  ✓ 无对应 CSV: {pq_file.name} ({len(df)} 行)")
                results['valid'] += 1
                results['missing_csv'] += 1

        except Exception as e:
            print(f"  验证失败: {pq_file.name} - {e}")
            results['invalid'] += 1

    return results


def backup_csv(dir_path: Path) -> bool:
    """备份 CSV 文件"""
    import shutil

    csv_files = list(dir_path.glob('*.csv'))
    if not csv_files:
        print(f"  目录为空: {dir_path}")
        return False

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # 按目录名创建子目录
    backup_subdir = BACKUP_DIR / dir_path.name
    backup_subdir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        backup_file = backup_subdir / csv_file.name
        shutil.copy2(csv_file, backup_file)

    print(f"  已备份 {len(csv_files)} 个 CSV 文件到 {backup_subdir}")
    return True


def cleanup_csv(dir_path: Path, backup: bool = True) -> dict:
    """删除 CSV 文件（如果 Parquet 存在）"""
    csv_files = list(dir_path.glob('*.csv'))
    parquet_files = {f.with_suffix('.csv').name for f in dir_path.glob('*.parquet')}

    results = {'total': 0, 'deleted': 0, 'kept': 0}

    for csv_file in csv_files:
        results['total'] += 1

        if csv_file.name in parquet_files:
            if backup:
                # 先备份
                if backup_csv(dir_path):
                    csv_file.unlink()
                    results['deleted'] += 1
            else:
                csv_file.unlink()
                results['deleted'] += 1
        else:
            results['kept'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='CSV to Parquet 格式转换')
    parser.add_argument('--convert', action='store_true', help='执行转换')
    parser.add_argument('--verify', action='store_true', help='验证 Parquet 文件')
    parser.add_argument('--cleanup', action='store_true', help='转换后删除 CSV')
    parser.add_argument('--backup', action='store_true', default=True, help='删除前备份 CSV')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要进行的操作')
    parser.add_argument('--dir', choices=['cache', 'raw', 'both'], default='cache',
                        help='指定目录')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("CSV to Parquet 格式转换")
    print(f"{'='*60}")

    dirs = []
    if args.dir == 'cache':
        dirs = [CACHE_DIR]
    elif args.dir == 'raw':
        dirs = [RAW_DIR]
    else:
        dirs = [CACHE_DIR, RAW_DIR]

    if args.verify:
        for d in dirs:
            print(f"\n验证目录: {d}")
            results = verify_parquet(d)
            print(f"  结果: {results['valid']}/{results['total']} 有效")

    elif args.convert:
        for d in dirs:
            print(f"\n转换目录: {d}")
            results = convert_directory(d, dry_run=args.dry_run)

        if not args.dry_run:
            # 验证
            print(f"\n验证转换结果:")
            for d in dirs:
                print(f"\n  {d}:")
                results = verify_parquet(d)
                print(f"    {results['valid']}/{results['total']} 验证通过")

        if args.cleanup and not args.dry_run:
            print(f"\n清理 CSV 文件:")
            for d in dirs:
                print(f"\n  {d}:")
                results = cleanup_csv(d, backup=args.backup)
                print(f"    删除: {results['deleted']}, 保留: {results['kept']}")

    else:
        # 显示当前状态
        print("\n当前格式状态:")
        for d in dirs:
            csv_count = len(list(d.glob('*.csv')))
            pq_count = len(list(d.glob('*.parquet')))
            print(f"  {d}: {csv_count} CSV, {pq_count} Parquet")

        print(f"\n使用方法:")
        print(f"  python convert_to_parquet.py --convert    # 转换")
        print(f"  python convert_to_parquet.py --verify     # 验证")
        print(f"  python convert_to_parquet.py --cleanup    # 转换并清理 CSV")
        print(f"  python convert_to_parquet.py --dry-run    # 预览")


if __name__ == '__main__':
    main()
