"""回溯测试入口（统一版，合并原 eval_strategy.py 的全部能力）。

特性：
- 选策略：`--strategy` 指定单个/多个策略（默认跑 strategies.yaml 里所有启用的策略）；
- 多持仓周期：`--holdings 3,5,10` 一次跑完，无需重跑；`--holding 5` 为单周期简写；
- 多数据源：`--local` 离线时用 `--libraries a.parquet,b.parquet` 一次合并多个样本库；
  也可用 `--library` 指定单个（向后兼容），或不加 `--local` 走 baostock 联网全市场；
- 向量化预筛：策略实现 `prefilter_mask()` 时自动跳过不可能的交易日，口径不变、速度更快；
  用 `--no-prefilter` 可关闭预筛做等价性校验；
- 完整指标：胜率 / 涨幅分布 / 累计收益 + 相对沪深300超额 + Sharpe / Sortino / 最大回撤。

用法：
    # 单策略、多持仓周期、合并样本库（推荐）
    python backtest.py --start 2025-11-12 --end 2026-09-30 --local \
        --library data_library/combined_daily.parquet \
        --strategy ma_cross_breakout --holdings 3,5,10

    # 全部启用策略、单持仓周期、单个本地库
    python backtest.py --start 2025-11-12 --end 2026-09-30 --local --holding 5 \
        --library data_library/combined_daily.parquet

    # 联网全市场（baostock）
    python backtest.py --start 2025-01-01 --end 2025-12-31 --holding 10

产出（backtest_output/）：
    backtest_detail_<start>_<end>.csv     逐笔明细（含基准/超额收益）
    backtest_summary_<start>_<end>.csv    策略 × 数据源 × 持仓周期 汇总
    backtest_combined_<start>_<end>.csv   策略 × 持仓周期（跨数据源合并）
    backtest_signals_<start>_<end>.csv    信号日志
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.backtester import Backtester, summarize, write_outputs
from core.data_fetcher import DataFetcher
from core.local_library import LocalDataLibrary
from core.market_context import build_context
from core.strategy_loader import load_strategies, load_yaml


def _parse_names(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_holdings(args: argparse.Namespace) -> list[int] | None:
    """--holding（单个，向后兼容）优先于 --holdings（多个）。"""
    if args.holding is not None:
        return [int(args.holding)]
    if args.holdings:
        return [int(h) for h in _parse_names(args.holdings)]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="策略回溯测试（多策略 / 多持仓周期 / 多数据源）")
    parser.add_argument("--start", required=True, help="回溯开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="回溯结束日期 YYYY-MM-DD")
    parser.add_argument("--strategy", help="仅运行指定策略（逗号分隔的名称，默认全部启用）")
    parser.add_argument("--holdings", help="持仓周期（交易日），逗号分隔，如 3,5,10")
    parser.add_argument("--holding", type=int, help="持仓周期（单个，向后兼容，优先于 --holdings）")
    parser.add_argument("--local", action="store_true", help="使用本地回溯样本库（Parquet），不联网")
    parser.add_argument("--library", default="data_library/combined_daily.parquet",
                        help="单个本地样本库 Parquet 路径（--local 且未给 --libraries 时生效）")
    parser.add_argument("--libraries", help="多个本地样本库，逗号分隔（优先于 --library）")
    parser.add_argument("--no-prefilter", action="store_true",
                        help="关闭策略向量化预筛（逐日全量计算，用于等价性校验）")
    parser.add_argument("--market-index", default="399317",
                        help="市场环境变量所用指数（默认 399317 国证A指）")
    parser.add_argument("--refresh-ctx", action="store_true",
                        help="强制刷新市场指数/证券元数据缓存（忽略本地缓存）")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--strategies", default="config/strategies.yaml")
    args = parser.parse_args()

    settings = load_yaml(args.config)
    all_strategies = load_strategies(args.strategies)

    # ---- 策略过滤 ----
    wanted = _parse_names(args.strategy)
    if wanted:
        by_name = dict(all_strategies)
        missing = [n for n in wanted if n not in by_name]
        if missing:
            print(f"[error] 未找到策略 {missing}（已启用：{list(by_name)}）", file=sys.stderr)
            return 1
        strategies = [(n, by_name[n]) for n in wanted]
    else:
        strategies = all_strategies
    if not strategies:
        print("[error] 没有可运行的策略", file=sys.stderr)
        return 1

    holdings = _parse_holdings(args)
    print(f"回溯区间: {args.start} ~ {args.end}，策略: {[n for n, _ in strategies]}，"
          f"持仓: {holdings or '配置默认'}"
          f"{'' if args.no_prefilter else '（启用预筛）'}")

    # ---- 策略上下文：市场指数行情 + 证券元数据（需要 bind_context 的策略用）----
    ctx = build_context(index_code=args.market_index, start=args.start, end=args.end,
                        refresh=args.refresh_ctx)
    for _, strat in strategies:
        strat.bind_context(ctx)

    # ---- 数据源 ----
    if args.local:
        libs = _parse_names(args.libraries) or [args.library]
        sources: list[tuple[str, object]] = []
        for p in libs:
            name = Path(p).stem.replace("_daily", "") or Path(p).stem
            sources.append((name, LocalDataLibrary(p)))
    else:
        sources = [("", DataFetcher())]

    all_trades: list[dict] = []
    all_signals: list[dict] = []

    for name, fetcher in sources:
        tag = name or "baostock"
        if args.local:
            print(f"[local] {tag}: {fetcher.describe()}")
        else:
            print(f"[net] {tag}: 联网获取全市场股票池")
        try:
            fetcher.login()
            bt = Backtester(fetcher, settings, library_name=name)
            bt.run(strategies, args.start, args.end,
                   holdings=holdings, use_prefilter=not args.no_prefilter)
            all_trades.extend(bt.trades)
            all_signals.extend(bt.signals)
            n_sig = sum(1 for s in bt.signals if s.get("library") == name)
            print(f"  {tag}: 交易 {len(bt.trades)} 笔，信号 {n_sig} 个")
        except Exception as exc:
            print(f"[error] {tag} 回溯失败: {exc}", file=sys.stderr)
            return 1
        finally:
            fetcher.logout()

    # ---- 汇总与落盘 ----
    detail, summary, combined, signals = summarize(all_trades, all_signals)
    out_dir = Path(settings.get("app", {}).get("backtest_output_dir", "backtest_output"))
    paths = write_outputs(detail, summary, combined, signals, out_dir, args.start, args.end)

    print(f"\n===== 信号总数：{len(signals)}，交易总数：{len(detail)} =====")
    print("\n===== 策略 × 数据源 × 持仓周期 =====")
    print(summary.to_string(index=False) if not summary.empty else "(无交易)")
    print("\n===== 策略 × 持仓周期（跨数据源合并）=====")
    print(combined.to_string(index=False) if not combined.empty else "(无交易)")
    print("\n输出文件:")
    for key in ("detail", "summary", "combined", "signals"):
        print(f"  {key:9s}: {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
