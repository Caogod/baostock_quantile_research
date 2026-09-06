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
            n = len(df)

            for strat_name, strat in strategies:
                for i in range(1, n):
                    if dates[i] < start_date or dates[i] > end_date:
                        continue
                    sub = df.iloc[: i + 1]
                    try:
                        sig = strat.signal(sub)
                    except Exception:
                        continue
                    if not sig.get("buy"):
                        continue

                    entry_date = dates[i]
                    entry_price = float(close.iloc[i])
                    exit_idx = i + self.holding_days
                    if exit_idx >= n:
                        continue  # 持仓期末超出样本，跳过

                    exit_date = dates[exit_idx]
                    exit_price = float(close.iloc[exit_idx])

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
            ])
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

        summary_rows = []
        for strat_name, grp in detail.groupby("strategy"):
            total = len(grp)
            wins = int(grp["win"].sum())
            ret = grp["return_pct"].astype(float)
            summary_rows.append({
                "strategy": strat_name,
                "total_trades": total,
                "win_count": wins,
                "win_rate": round(wins / total, 4) if total else 0.0,
                "avg_return_pct": round(float(ret.mean()), 4) if total else 0.0,
                "median_return_pct": round(float(ret.median()), 4) if total else 0.0,
                "max_return_pct": round(float(ret.max()), 4) if total else 0.0,
                "min_return_pct": round(float(ret.min()), 4) if total else 0.0,
                "cum_return_pct": round(float((1 + ret / 100).prod() - 1) * 100, 4) if total else 0.0,
            })

        summary = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / f"backtest_summary_{start_date}_{end_date}.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return detail_path, summary_path
