"""盘后选股引擎：遍历股票池 → 应用策略 → 汇总信号 → 输出 CSV。

健壮性：支持断点续跑。每处理 N 只股票（checkpoint_interval）将已处理代码与
命中结果落盘；若因 baostock 底层异常进程退出，重跑时自动跳过已处理股票继续。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .data_fetcher import DataFetcher
from .strategy_loader import PROJECT_ROOT

_COLUMNS = [
    "date", "code", "code_name", "strategy", "score", "reason",
    "close", "pctChg", "turn", "volume", "amount",
]


class Selector:
    def __init__(self, fetcher: DataFetcher, settings: dict[str, Any]) -> None:
        self.fetcher = fetcher
        self.settings = settings

        data_cfg = settings.get("data", {})
        self.adjustflag = str(data_cfg.get("adjustflag", "2"))
        self.history_days = int(data_cfg.get("history_days", 120))
        self.code_prefixes = data_cfg.get("code_prefixes", ["sh.60", "sh.68", "sz.00", "sz.30"])
        self.exclude_st = bool(data_cfg.get("exclude_st", True))
        self.price_min = float(data_cfg.get("price_min", 0.0))
        self.price_max = float(data_cfg.get("price_max", 0.0))
        self.max_stocks = int(data_cfg.get("max_stocks", 0))
        self.explicit_codes = data_cfg.get("explicit_codes", []) or []
        self.checkpoint_interval = int(data_cfg.get("checkpoint_interval", 50))
        self.resume = bool(data_cfg.get("resume", True))
        # 请求限速（秒），降低对免费数据源的压力
        self.fetcher.request_interval = float(data_cfg.get("request_interval", 0.05))

        self.output_dir = PROJECT_ROOT / settings.get("app", {}).get("output_dir", "output")

    # ------------------------------------------------------------------ #
    def _build_pool(self, as_of_date: str) -> list[tuple[str, str]]:
        """返回 [(code, code_name), ...]。"""
        if self.explicit_codes:
            return [(c, "") for c in self.explicit_codes]

        all_stocks = self.fetcher.get_all_stocks(as_of_date)
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

        if self.max_stocks > 0:
            df = df.head(self.max_stocks)

        return list(zip(df["code"].tolist(), df["code_name"].tolist()))

    # ------------------------------------------------------------------ #
    def _pool_signature(self) -> str:
        """生成股票池配置签名，用于隔离不同股票池的断点续跑检查点。

        避免显式股票池 / max_stocks 调试运行与全市场运行的检查点互相污染。
        """
        key = json.dumps(
            {
                "prefixes": self.code_prefixes,
                "explicit": self.explicit_codes,
                "max_stocks": self.max_stocks,
                "exclude_st": self.exclude_st,
                "adjustflag": self.adjustflag,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]

    def _checkpoint_path(self, as_of_date: str) -> Path:
        return self.output_dir / f"checkpoint_{as_of_date}_{self._pool_signature()}.json"

    # ------------------------------------------------------------------ #
    def _in_price_range(self, df: pd.DataFrame) -> bool:
        """判断该股票最新收盘价是否落在配置的价格区间内。

        区间由 settings.yaml 的 data.price_min / price_max 控制；
        两者均为 0（默认）时不作过滤。
        """
        if not self.price_min and not self.price_max:
            return True
        try:
            close = float(df.iloc[-1]["close"])
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        if self.price_min > 0 and close < self.price_min:
            return False
        if self.price_max > 0 and close > self.price_max:
            return False
        return True

    # ------------------------------------------------------------------ #
    def run(self, as_of_date: str, strategies: list) -> Path:
        """执行一次盘后选股，返回输出 CSV 路径。支持断点续跑。"""
        pool = self._build_pool(as_of_date)
        if not pool:
            raise RuntimeError(f"交易日 {as_of_date} 未获取到可用股票池")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"selection_{as_of_date}.csv"
        checkpoint_path = self._checkpoint_path(as_of_date)

        done: set[str] = set()
        rows: list[dict] = []
        if self.resume and checkpoint_path.exists():
            try:
                state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                done = set(state.get("done", []))
            except Exception:
                done = set()
            if out_path.exists():
                try:
                    rows = pd.read_csv(out_path).to_dict("records")
                except Exception:
                    rows = []
            print(f"断点续跑: 已跳过 {len(done)} 只，已有 {len(rows)} 条命中")

        start = (datetime.strptime(as_of_date, "%Y-%m-%d")
                 - timedelta(days=int(self.history_days * 1.8))).strftime("%Y-%m-%d")

        total = len(pool)
        for idx, (code, code_name) in enumerate(pool, 1):
            if code in done:
                continue
            df = self.fetcher.get_history(code, start, as_of_date, self.adjustflag)
            if not df.empty and self._in_price_range(df):
                for strat_name, strat in strategies:
                    try:
                        sig = strat.signal(df)
                    except Exception as exc:  # 单只股票异常不影响整体
                        print(f"  [warn] {code} {strat_name} 信号异常: {exc}")
                        continue
                    if sig.get("buy"):
                        last = df.iloc[-1]
                        rows.append({
                            "date": as_of_date,
                            "code": code,
                            "code_name": code_name,
                            "strategy": strat_name,
                            "score": sig.get("score", 0.0),
                            "reason": sig.get("reason", ""),
                            "close": last.get("close"),
                            "pctChg": last.get("pctChg"),
                            "turn": last.get("turn"),
                            "volume": last.get("volume"),
                            "amount": last.get("amount"),
                        })
            done.add(code)
            self.fetcher._throttle()

            if idx % self.checkpoint_interval == 0 or idx == total:
                self._save_progress(done, rows, checkpoint_path, out_path)
                print(f"  进度 {idx}/{total}（已跳过 {len(done)}），命中 {len(rows)} 条")

        # 完成：最终落盘并清理检查点
        result = self._sort_rows(rows)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass  # 某些环境（沙箱安全删除）不允许 unlink，残留检查点无害
        return out_path

    # ------------------------------------------------------------------ #
    def _sort_rows(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=_COLUMNS) if rows else pd.DataFrame(columns=_COLUMNS)
        if not df.empty:
            df = df.sort_values(["strategy", "score"], ascending=[True, False])
        return df

    def _save_progress(self, done: set[str], rows: list[dict],
                       checkpoint_path: Path, out_path: Path) -> None:
        checkpoint_path.write_text(
            json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8"
        )
        self._sort_rows(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
