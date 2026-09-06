"""定时运行调度器（无界面常驻进程）。

用法：
    python scheduler.py                 # 常驻，按配置时间定时选股（联网，全市场）
    python scheduler.py --once          # 立即执行一次后退出（供 cron/任务计划调用）
    python scheduler.py --once --local  # 用本地样本库离线选股（秒级）
    python scheduler.py --once --local --filter   # 离线选股 + 胜率模型二次过滤

调度时间与"仅交易日运行"在 config/settings.yaml 的 schedule 段配置。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule

from core.data_fetcher import DataFetcher
from core.local_library import LocalDataLibrary
from core.selector import Selector
from core.strategy_loader import load_strategies, load_yaml


def run_once(settings: dict, strategies: list, args) -> None:
    """执行一次盘后选股（可选二次过滤）。"""
    if args.local:
        fetcher = LocalDataLibrary(args.library)
    else:
        fetcher = DataFetcher()

    try:
        fetcher.login()
        if args.local:
            as_of_date = fetcher.date_range[1]
        else:
            as_of_date = fetcher.latest_trading_day()

        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 盘后选股: {as_of_date}")
        selector = Selector(fetcher, settings)
        out_path = selector.run(as_of_date, strategies)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 完成: {out_path}")

        # 模块4：胜率模型二次过滤
        if args.filter:
            from core.win_model import apply_filter

            filtered_path, n_before, n_after, filtered = apply_filter(
                out_path,
                library_path=args.library,
                model_path=args.model,
                threshold=args.filter_threshold,
            )
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 二次过滤: "
                  f"{n_before} 只 → 保留 {n_after} 只（阈值 {args.filter_threshold}）→ {filtered_path}")
            if not filtered.empty:
                print(filtered[["code", "code_name", "win_prob", "close"]].to_string(index=False))
    finally:
        fetcher.logout()


def main() -> int:
    parser = argparse.ArgumentParser(description="盘后选股定时调度")
    parser.add_argument("--once", action="store_true", help="立即执行一次后退出")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--strategies", default="config/strategies.yaml")
    parser.add_argument("--strategy", help="仅运行指定策略（逗号分隔的名称，可选）")
    parser.add_argument("--local", action="store_true", help="使用本地样本库（Parquet），不联网")
    parser.add_argument("--library", default="data_library/zz500_daily.parquet",
                        help="本地样本库 Parquet 路径（--local 或 --filter 时使用）")
    parser.add_argument("--filter", action="store_true", help="选股后用胜率模型二次过滤")
    parser.add_argument("--model", default="models/win_model.joblib",
                        help="胜率预测模型路径（--filter 时使用）")
    parser.add_argument("--filter-threshold", type=float, default=0.5,
                        help="二次过滤的胜率阈值（默认 0.5）")
    args = parser.parse_args()

    settings = load_yaml(args.config)
    strategies = load_strategies(args.strategies)
    if args.strategy:
        wanted = {x.strip() for x in args.strategy.split(",") if x.strip()}
        strategies = [(n, s) for n, s in strategies if n in wanted]
    if not strategies:
        print("[error] 没有启用的策略")
        return 1

    if args.filter and not Path(args.model).exists():
        print(f"[error] 模型不存在: {args.model}（请先运行 train_model.py）", file=sys.stderr)
        return 1

    if args.once:
        run_once(settings, strategies, args)
        return 0

    sched_cfg = settings.get("schedule", {})
    run_time = sched_cfg.get("time", "15:30")
    trading_day_only = bool(sched_cfg.get("trading_day_only", True))

    def job():
        try:
            if trading_day_only and datetime.now().weekday() >= 5:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 非交易日，跳过")
                return
            run_once(settings, strategies, args)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)

    schedule.every().day.at(run_time).do(job)
    mode = "离线样本库" if args.local else "联网全市场"
    extra = " + 胜率模型二次过滤" if args.filter else ""
    print(f"调度已启动，每日 {run_time} 运行（{mode}{extra}，仅交易日: {trading_day_only}）")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
