"""完整回溯测试引擎。

对每只股票遍历历史交易日，在每个时点用截至当日的数据运行策略信号，
命中信号即模拟一笔"次日开盘买入 → 持有 N 日 → 卖出"的交易，
统计胜率（盈利交易占比）与涨幅（收益分布），输出逐笔明细与汇总 CSV。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .data_fetcher import DataFetcher
from .strategy_loader import PROJECT_ROOT


class Backtester:
    def __init__(self, fetcher: DataFetcher, settings: dict[str, Any]) -> None:
        self.fetcher = fetcher
        self.settings = settings

        data_cfg = settings.get("data", {})
        self.adjustflag = str(data_cfg.get("adjustflag", "2"))
        self.code_prefixes = data_cfg.get("code_prefixes", ["sh.60", "sh.68", "sz.00", "sz.30"])
        self.exclude_st = bool(data_cfg.get("exclude_st", True))
        self.explicit_codes = data_cfg.get("explicit_codes", []) or []

        # 请求限速
        self.fetcher.request_interval = float(data_cfg.get("request_interval", 0.05))

        bt_cfg = settings.get("backtest", {})
        self.holding_days = int(bt_cfg.get("holding_days", 5))
        self.lookback_days = int(bt_cfg.get("lookback_days", 200))
        self.commission = float(bt_cfg.get("commission", 0.001))  # 单边费率

        self.output_dir = PROJECT_ROOT / settings.get("app", {}).get("backtest_output_dir", "backtest_output")

    # ------------------------------------------------------------------ #
    def _build_pool(self, end_date: str) -> list[str]:
        if self.explicit_codes:
            return list(self.explicit_codes)

        all_stocks = self.fetcher.get_all_stocks(end_date)
        if all_stocks.empty:
            return []
        status_col = "tradeStatus" if "tradeStatus" in all_stocks.columns else "tradestatus"
        df = all_stocks[all_stocks[status_col].astype(str) == "1"]
        mask = pd.Series([False] * len(df), index=df.index)
        for prefix in self.code_prefixes:
            mask |= df["code"].str.startswith(prefix)
        df = df[mask]
        if self.exclude_st and "code_name" in df.columns:
            df = df[~df["code_name"].str.contains("ST|退", na=False)]
        return df["code"].tolist()

    # ------------------------------------------------------------------ #
    def run(
        self,
        strategies: list,
        start_date: str,
        end_date: str,
    ) -> tuple[Path, Path]:
        """运行回溯，返回 (逐笔明细CSV, 汇总CSV) 路径。"""
        pool = self._build_pool(end_date)
        fetch_start = (datetime.strptime(start_date, "%Y-%m-%d")
                       - timedelta(days=int(self.lookback_days * 1.8))).strftime("%Y-%m-%d")

        trades: list[dict] = []
        for code in pool:
            df = self.fetcher.get_history(code, fetch_start, end_date, self.adjustflag)
            if df.empty or len(df) < 2:
                continue
            dates = df["date"].astype(str).tolist()
            close = df["close"].astype(float).reset_index(drop=True)
            open_ = df["open"].astype(float).reset_index(drop=True)
            n = len(df)

            for strat_name, strat in strategies:
                for i in range(1, n):
                    if dates[i] < start_date or dates[i] > end_date:
                        continue
                    # 信号在 T 日收盘后生成（sub 含截至 T 日的全部数据）
                    sub = df.iloc[: i + 1]
                    try:
                        sig = strat.signal(sub)
                    except Exception:
                        continue
                    if not sig.get("buy"):
                        continue

                    # P0 修正：T 日收盘生成信号 → T+1 开盘价入场（可成交口径）
                    # 入场日 = i+1，持仓 holding_days 个交易日后于 i+1+holding 日收盘卖出
                    entry_idx = i + 1
                    exit_idx = i + 1 + self.holding_days
                    if entry_idx >= n or exit_idx >= n:
                        continue  # 入场或出场日超出样本，跳过

                    entry_date = dates[entry_idx]
                    entry_price = float(open_.iloc[entry_idx])  # 次日开盘
                    exit_date = dates[exit_idx]
                    exit_price = float(close.iloc[exit_idx])     # 持仓期末收盘

                    # 含双边佣金后的净收益
                    gross = exit_price / entry_price - 1
                    net = gross - 2 * self.commission

                    trades.append({
                        "strategy": strat_name,
                        "code": code,
                        "entry_date": entry_date,
                        "entry_price": round(entry_price, 4),
                        "exit_date": exit_date,
                        "exit_price": round(exit_price, 4),
                        "holding_days": self.holding_days,
                        "return_pct": round(net * 100, 4),
                        "win": 1 if net > 0 else 0,
                    })
            self.fetcher._throttle()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        detail_path, summary_path = self._save(trades, start_date, end_date)
        return detail_path, summary_path

    # ------------------------------------------------------------------ #
    def _save(self, trades: list[dict], start_date: str, end_date: str) -> tuple[Path, Path]:
        detail = pd.DataFrame(trades)
        detail_path = self.output_dir / f"backtest_detail_{start_date}_{end_date}.csv"
        if detail.empty:
            detail = pd.DataFrame(columns=[
                "strategy", "code", "entry_date", "entry_price",
                "exit_date", "exit_price", "holding_days", "return_pct", "win",
                "bench_return_pct", "excess_return_pct", "excess_win",
            ])
        else:
            # 基准同期收益（沪深300 entry→exit 收益率，百分数口径对齐 return_pct）
            from .market import benchmark_returns_for_trades
            bench = benchmark_returns_for_trades(detail) * 100.0
            detail["bench_return_pct"] = bench.round(4)
            detail["excess_return_pct"] = (detail["return_pct"].astype(float)
                                           - detail["bench_return_pct"]).round(4)
            detail["excess_win"] = (detail["excess_return_pct"] > 0).astype(int)
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

        # 年化因子：交易离散，按"每笔 ≈ holding_days 个交易日"近似年化
        # Sharpe/Sortino 年化 = mean/std * sqrt(252 / holding_days)
        ann_factor = (252.0 / self.holding_days) ** 0.5 if self.holding_days > 0 else 1.0

        summary_rows = []
        for strat_name, grp in detail.groupby("strategy"):
            total = len(grp)
            wins = int(grp["win"].sum())
            ret = grp["return_pct"].astype(float)
            bench = grp["bench_return_pct"].astype(float) if "bench_return_pct" in grp else pd.Series([0.0]*total)
            excess = grp["excess_return_pct"].astype(float) if "excess_return_pct" in grp else ret
            excess_wins = int((excess > 0).sum()) if total else 0

            # 风险指标（基于超额收益序列，百分数→小数）
            ex = excess / 100.0
            std = float(ex.std(ddof=1)) if total > 1 else 0.0
            # 下行标准差（仅取负超额）
            downside = ex[ex < 0]
            dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
            sharpe = float(ex.mean() / std * ann_factor) if std > 0 else 0.0
            sortino = float(ex.mean() / dstd * ann_factor) if dstd > 0 else 0.0
            # 最大回撤（按超额累计净值序列）
            nav = (1 + ex).cumprod()
            peak = nav.cummax()
            dd = (nav / peak - 1)
            max_dd = float(dd.min()) if not dd.empty else 0.0

            summary_rows.append({
                "strategy": strat_name,
                "total_trades": total,
                "win_count": wins,
                "win_rate": round(wins / total, 4) if total else 0.0,
                "excess_win_count": excess_wins,
                "excess_win_rate": round(excess_wins / total, 4) if total else 0.0,
                "avg_return_pct": round(float(ret.mean()), 4) if total else 0.0,
                "avg_excess_pct": round(float(excess.mean()), 4) if total else 0.0,
                "avg_bench_pct": round(float(bench.mean()), 4) if total else 0.0,
                "median_return_pct": round(float(ret.median()), 4) if total else 0.0,
                "max_return_pct": round(float(ret.max()), 4) if total else 0.0,
                "min_return_pct": round(float(ret.min()), 4) if total else 0.0,
                "cum_return_pct": round(float((1 + ret / 100).prod() - 1) * 100, 4) if total else 0.0,
                "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4),
                "max_drawdown_pct": round(max_dd * 100, 4),
            })

        summary = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / f"backtest_summary_{start_date}_{end_date}.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return detail_path, summary_path
