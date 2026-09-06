"""回溯测试入口。

用法：
    python backtest.py --start 2025-01-01 --end 2025-12-31
    python backtest.py --start 2025-01-01 --end 2025-12-31 --holding 10
    python backtest.py --start 2025-01-01 --end 2025-12-31 --local   # 用本地样本库离线回溯
"""
from __future__ import annotations

import argparse
import sys

from core.backtester import Backtester
from core.data_fetcher import DataFetcher
from core.local_library import LocalDataLibrary
from core.strategy_loader import load_strategies, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="策略回溯测试（输出胜率与涨幅）")
    parser.add_argument("--start", required=True, help="回溯开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="回溯结束日期 YYYY-MM-DD")
    parser.add_argument("--holding", type=int, help="持仓周期（交易日），覆盖配置")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--strategies", default="config/strategies.yaml")
    parser.add_argument("--local", action="store_true", help="使用本地回溯样本库（Parquet），不联网")
    parser.add_argument("--library", default="data_library/zz500_daily.parquet",
                        help="本地样本库 Parquet 路径")
    args = parser.parse_args()

    settings = load_yaml(args.config)
    strategies = load_strategies(args.strategies)
    if not strategies:
        print("[error] 没有启用的策略")
        return 1

    if args.local:
        fetcher = LocalDataLibrary(args.library)
        print(f"[local] 样本库: {args.library}，范围 {fetcher.date_range}")
    else:
        fetcher = DataFetcher()

    try:
        fetcher.login()
        bt = Backtester(fetcher, settings)
        if args.holding is not None:
            bt.holding_days = args.holding

        print(f"回溯区间: {args.start} ~ {args.end}, 持仓 {bt.holding_days} 日")
        detail_path, summary_path = bt.run(strategies, args.start, args.end)

        print(f"逐笔明细: {detail_path}")
        print(f"汇总报告: {summary_path}")
        print("---------- 汇总 ----------")
        import pandas as pd
        summary = pd.read_csv(summary_path)
        print(summary.to_string(index=False))
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        fetcher.logout()


if __name__ == "__main__":
    raise SystemExit(main())
