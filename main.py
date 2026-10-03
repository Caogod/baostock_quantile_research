"""盘后选股入口（单次运行）。

用法：
    python main.py                          # 对最近一个交易日选股（联网，全市场）
    python main.py --date 2026-09-04        # 指定交易日
    python main.py --local --date 2026-09-30            # 用本地合并库离线选股（默认 combined_daily）
    python main.py --local --strategy ma_cross_breakout # 仅用指定策略
    python main.py --local --strategy b1_oversold --filter   # 选股后二次过滤（按策略独立模型）
"""
from __future__ import annotations

import argparse
import sys

from core.data_fetcher import DataFetcher
from core.local_library import LocalDataLibrary
from core.market_context import build_context
from core.selector import Selector
from core.strategy_loader import load_strategies, load_yaml


def _configure_index_pool(fetcher: DataFetcher, settings: dict) -> None:
    """将配置的指数成分股合并为去重后的显式股票池。"""
    data_cfg = settings.setdefault("data", {})
    index_names = data_cfg.get("index_names", []) or []
    if not index_names:
        return

    codes = list(data_cfg.get("explicit_codes", []) or [])
    seen = set(codes)
    for index_name in index_names:
        constituents = fetcher.get_index_stocks(str(index_name))
        for code in constituents["code"].tolist():
            if code not in seen:
                codes.append(code)
                seen.add(code)
        print(f"股票池已加入 {index_name}: {len(constituents)} 只")

    data_cfg["explicit_codes"] = codes
    print(f"股票池合计: {len(codes)} 只（zz500 + hs300 去重后）")


def main() -> int:
    parser = argparse.ArgumentParser(description="盘后策略选股")
    parser.add_argument("--date", help="选股日期 YYYY-MM-DD，默认最近交易日")
    parser.add_argument("--config", default="config/settings.yaml", help="全局配置文件")
    parser.add_argument("--strategies", default="config/strategies.yaml", help="策略注册文件")
    parser.add_argument("--local", action="store_true", help="使用本地样本库（Parquet），不联网")
    parser.add_argument("--library", default="data_library/combined_daily.parquet",
                        help="本地样本库 Parquet 路径")
    parser.add_argument("--strategy", help="仅运行指定策略（逗号分隔的名称，可选）")
    parser.add_argument("--filter", action="store_true", help="选股后用胜率模型二次过滤")
    parser.add_argument("--market-index", default="399317",
                        help="市场环境变量所用指数（默认 399317 国证A指）")
    parser.add_argument("--model", default="models",
                        help="胜率模型目录（--filter 时按策略加载 win_model_<strategy>.joblib）")
    parser.add_argument("--filter-threshold", type=float, default=0.5,
                        help="二次过滤的胜率阈值（默认 0.5）")
    args = parser.parse_args()

    settings = load_yaml(args.config)
    strategies = load_strategies(args.strategies)
    if args.strategy:
        wanted = {x.strip() for x in args.strategy.split(",") if x.strip()}
        strategies = [(n, s) for n, s in strategies if n in wanted]
    if not strategies:
        print("[error] 没有启用的策略，请检查 %s" % args.strategies)
        return 1

    if args.local:
        fetcher = LocalDataLibrary(args.library)
        print(f"[local] 样本库: {args.library}，范围 {fetcher.date_range}")
        as_of_date = args.date or fetcher.date_range[1]
    else:
        fetcher = DataFetcher()

    try:
        fetcher.login()
        if not args.local:
            _configure_index_pool(fetcher, settings)
            as_of_date = args.date or fetcher.latest_trading_day()
        print(f"盘后选股交易日: {as_of_date}")
        print(f"启用策略: {', '.join(name for name, _ in strategies)}")

        # 策略上下文：市场指数行情 + 证券元数据（需要 bind_context 的策略用）
        ctx = build_context(index_code=args.market_index, start=as_of_date, end=as_of_date)
        for _, strat in strategies:
            strat.bind_context(ctx)

        selector = Selector(fetcher, settings)
        out_path = selector.run(as_of_date, strategies)
        print(f"选股完成，输出: {out_path}")

        # 模块4：胜率模型二次过滤
        if args.filter:
            from core.win_model import apply_filter

            filtered_path, n_before, n_after, filtered = apply_filter(
                out_path,
                library_path=args.library,
                model_path=args.model,
                threshold=args.filter_threshold,
            )
            print(f"二次过滤完成: 原始 {n_before} 只 → 保留 {n_after} 只"
                  f"（阈值 {args.filter_threshold}），输出 {filtered_path}")
            if not filtered.empty:
                print(filtered[["code", "code_name", "win_prob", "close"]].to_string(index=False))
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        fetcher.logout()


if __name__ == "__main__":
    raise SystemExit(main())

