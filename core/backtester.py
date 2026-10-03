"""完整回溯测试引擎（统一版）。

能力（合并原 eval_strategy.py）：
- 多策略：一次运行可跑若干策略（入口按 --strategy 过滤）；
- 多持仓周期：一次遍历同时展开多个持有天数（3/5/10 日），无需重跑；
- 多数据源：每个 Backtester 实例绑定一个数据源，用 library_name 标注来源，
  便于多本地样本库（zz500 / hs300）合并统计；
- 向量化预筛：策略可选实现 `prefilter_mask(close)` 声明"必要条件候选日"，
  引擎只对候选日调用 signal()（口径不变，速度提升一个量级）；
- 完整指标：胜率 / 涨幅分布 / 累计收益，以及相对沪深300的超额收益与
  Sharpe / Sortino / 最大回撤。
- 交易冷却期：同一股票同一策略，入场后冷却 holding 个交易日（= 持有期），
  平仓前不再产生新信号，避免同一标的在持有期内被重复计入。

成交口径（贯穿始终，不做任何简化）：
    T 日收盘生成信号 → T+1 开盘价入场 → 持有 H 个交易日后收盘卖出，
    net = 卖出收盘 / 买入开盘 - 1 - 2 × commission。

净值口径（cum_return_pct / max_drawdown_pct / Sharpe / Sortino）：
    按日历日把每笔交易收益摊到 [entry_date, exit_date]，同一天多笔交易按
    等权平均合并成组合日收益，再连乘得到去重叠的逐日净值曲线，据此计算
    累计收益、最大回撤与年化风险指标（而非把重叠交易直接连乘）。

用法（入口见 backtest.py）：
    bt = Backtester(LocalDataLibrary(path), settings, library_name="zz500")
    bt.run(strategies, "2025-11-12", "2026-09-30", holdings=[3, 5, 10])
    detail, summary, combined, signals = summarize(bt.trades, holdings)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .data_fetcher import DataFetcher
from .market import benchmark_returns_for_trades
from .strategy_loader import PROJECT_ROOT

# 逐笔明细列（保持历史列名以兼容 train_model.py 等下游）
DETAIL_COLUMNS = [
    "strategy", "library", "code", "signal_date",
    "entry_date", "entry_price", "exit_date", "exit_price",
    "holding_days", "return_pct", "win", "score",
    "bench_return_pct", "excess_return_pct", "excess_win",
]
SIGNAL_COLUMNS = ["strategy", "library", "code", "signal_date", "score", "reason"]


class Backtester:
    """单个数据源上的回溯执行器。"""

    def __init__(self, fetcher: Any, settings: dict[str, Any], library_name: str = "") -> None:
        self.fetcher = fetcher
        self.settings = settings
        self.library_name = library_name or ""

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

        # 运行期累积
        self.trades: list[dict] = []
        self.signals: list[dict] = []

    # ------------------------------------------------------------------ #
    # 股票池
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
    # 预筛
    # ------------------------------------------------------------------ #
    @staticmethod
    def _candidate_indices(strat: Any, n: int, close: np.ndarray,
                           df: pd.DataFrame | None = None) -> np.ndarray | range:
        """返回需要调用 signal() 的候选日索引。

        优先用 `prefilter_mask_df(df)`（可拿到完整日 K，支持 high/low 等字段），
        其次 `prefilter_mask(close)`；掩码非法（长度不符 / 抛异常）时安全回退为
        逐日全量扫描，绝不因预筛而漏信号。
        """
        if df is not None:
            pre_df = getattr(strat, "prefilter_mask_df", None)
            if callable(pre_df):
                try:
                    mask = pre_df(df)
                except Exception:  # noqa: BLE001
                    mask = None
                if mask is not None:
                    mask = np.asarray(mask, dtype=bool)
                    if mask.shape == (n,):
                        return np.nonzero(mask)[0]

        pre = getattr(strat, "prefilter_mask", None)
        if callable(pre):
            try:
                mask = pre(close)
            except Exception:
                mask = None
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape == (n,):
                    return np.nonzero(mask)[0]
        return range(1, n)

    # ------------------------------------------------------------------ #
    def run(
        self,
        strategies: Iterable[tuple[str, Any]],
        start_date: str,
        end_date: str,
        holdings: Sequence[int] | None = None,
        use_prefilter: bool = True,
    ) -> None:
        """执行回溯，结果累积在 self.trades / self.signals。"""
        strategies = list(strategies)
        hs = sorted({int(h) for h in (holdings or [self.holding_days]) if int(h) > 0})
        hs = hs or [self.holding_days]

        pool = self._build_pool(end_date)
        fetch_start = (datetime.strptime(start_date, "%Y-%m-%d")
                       - timedelta(days=int(self.lookback_days * 1.8))).strftime("%Y-%m-%d")

        for code in pool:
            df = self.fetcher.get_history(code, fetch_start, end_date, self.adjustflag)
            if df.empty or len(df) < 2:
                continue
            dates = df["date"].astype(str).tolist()
            close = df["close"].astype(float).to_numpy()
            open_ = df["open"].astype(float).to_numpy()
            n = len(df)

            for strat_name, strat in strategies:
                if use_prefilter:
                    candidates: Iterable[int] = self._candidate_indices(strat, n, close, df)
                else:
                    candidates = range(1, n)

                # 冷却期：按 (strategy, holding) 维护「下一次允许产生信号的信号日索引」。
                # 信号日索引 i 入场后持有 h 个交易日：T+1 开盘入场、T+1+h 收盘平仓。
                # 冷却期 = 持有期 h，平仓后才允许再次进场 → 下一个信号日索引 >= i+1+h。
                cooldown_until: dict[int, int] = {h: -1 for h in hs}

                for i in candidates:
                    if i < 1:
                        continue
                    if dates[i] < start_date or dates[i] > end_date:
                        continue
                    # 信号在 T 日收盘后生成（sub 含截至 T 日的全部数据）
                    try:
                        sig = strat.signal(df.iloc[: i + 1])
                    except Exception:
                        continue
                    if not sig.get("buy"):
                        continue

                    # 冷却期内各持仓周期是否允许入场（至少一个未冷却才记信号）
                    active_h = [h for h in hs if i >= cooldown_until[h]]
                    if not active_h:
                        continue

                    self.signals.append({
                        "strategy": strat_name,
                        "library": self.library_name,
                        "code": code,
                        "signal_date": dates[i],
                        "score": sig.get("score"),
                        "reason": sig.get("reason"),
                    })

                    # T 日收盘生成信号 → T+1 开盘价入场 → 持有 H 日后收盘卖出
                    for h in active_h:
                        entry_idx = i + 1
                        exit_idx = i + 1 + h
                        if entry_idx >= n or exit_idx >= n:
                            continue  # 入场或出场日超出样本，跳过
                        entry_price = float(open_[entry_idx])
                        if entry_price <= 0:
                            continue
                        exit_price = float(close[exit_idx])
                        gross = exit_price / entry_price - 1
                        net = gross - 2 * self.commission

                        # 入场后进入冷却：下一个信号最早在平仓日（信号日索引 + 1 + h）
                        cooldown_until[h] = i + 1 + h

                        self.trades.append({
                            "strategy": strat_name,
                            "library": self.library_name,
                            "code": code,
                            "signal_date": dates[i],
                            "entry_date": dates[entry_idx],
                            "entry_price": round(entry_price, 4),
                            "exit_date": dates[exit_idx],
                            "exit_price": round(exit_price, 4),
                            "holding_days": h,
                            "return_pct": round(net * 100, 4),
                            "win": 1 if net > 0 else 0,
                            "score": sig.get("score"),
                        })
            self.fetcher._throttle()


# --------------------------------------------------------------------------- #
# 汇总与落盘
# --------------------------------------------------------------------------- #
def _daily_nav_curve(grp: pd.DataFrame, value_col: str = "excess_return_pct") -> tuple[np.ndarray, np.ndarray]:
    """把一组交易按日历日摊成去重叠的等权组合日收益与净值。

    每笔交易在 [entry_date, exit_date] 区间内把总值（value_col，百分数）按
    几何方式摊到每一天（假设日内均匀复利）。同一天有多笔交易时取等权平均，
    得到组合日收益序列；连乘得到净值。返回 (dates, nav) 均为等长 numpy 数组，
    日期升序。这样多笔重叠交易不会重复复利，净值/回撤口径正确。
    """
    if grp.empty or "entry_date" not in grp.columns or "exit_date" not in grp.columns:
        return np.array([]), np.array([])

    # 汇总每个日历日的 (收益和, 覆盖笔数)
    daily_sum: dict[str, float] = {}
    daily_cnt: dict[str, int] = {}

    for _, r in grp.iterrows():
        e = str(r["entry_date"])
        x = str(r["exit_date"])
        total_ret = float(r[value_col]) / 100.0  # 小数
        # 持有期间的交易日集合（含首尾）
        span = grp.attrs.get("_calendar", {}).get((e, x))
        if span is None:
            # 无日历信息时退化为单日（entry 当天），保持可用性
            span = [e]

        if len(span) == 0:
            continue
        # 几何摊到每一天：总收益 (1+total_ret) 均分为 len(span) 天
        daily = (1.0 + total_ret) ** (1.0 / len(span)) - 1.0
        for d in span:
            daily_sum[d] = daily_sum.get(d, 0.0) + daily
            daily_cnt[d] = daily_cnt.get(d, 0) + 1

    if not daily_sum:
        return np.array([]), np.array([])

    dates = sorted(daily_sum)
    daily_rets = np.array([daily_sum[d] / daily_cnt[d] for d in dates])
    nav = np.cumprod(1.0 + daily_rets)
    return np.array(dates), nav


def _calendar_map(grp: pd.DataFrame, all_dates: list[str] | None = None) -> dict:
    """为每笔交易建立 (entry_date, exit_date) -> 交易日序列 的映射。

    优先用引擎运行时传入的全市场交易日历（all_dates）；否则用该组内出现过的
    entry/exit 日期排序近似。返回 dict，键为 (entry, exit) 元组。
    """
    cal: dict[tuple[str, str], list[str]] = {}
    if all_dates:
        pos = {d: i for i, d in enumerate(all_dates)}
        for _, r in grp.iterrows():
            e, x = str(r["entry_date"]), str(r["exit_date"])
            i0, i1 = pos.get(e), pos.get(x)
            if i0 is not None and i1 is not None and i1 >= i0:
                cal[(e, x)] = all_dates[i0:i1 + 1]
            else:
                cal[(e, x)] = [e] if e else []
    else:
        dates_seen = sorted(set(grp["entry_date"].astype(str)) | set(grp["exit_date"].astype(str)))
        pos = {d: i for i, d in enumerate(dates_seen)}
        for _, r in grp.iterrows():
            e, x = str(r["entry_date"]), str(r["exit_date"])
            i0, i1 = pos.get(e), pos.get(x)
            cal[(e, x)] = dates_seen[i0:i1 + 1] if i0 is not None and i1 is not None and i1 >= i0 else [e]
    return cal


def _metrics(grp: pd.DataFrame, holding_days: int, calendar: list[str] | None = None) -> dict:
    """单组（策略 × 库 × 持仓周期）的收益与风险指标。

    calendar：全市场交易日历（升序），用于把交易摊到日历日。None 时用组内日期近似。
    """
    total = len(grp)
    if total == 0:
        return {
            "total_trades": 0, "win_count": 0, "win_rate": 0.0,
            "excess_win_count": 0, "excess_win_rate": 0.0,
            "avg_return_pct": 0.0, "avg_excess_pct": 0.0, "avg_bench_pct": 0.0,
            "median_return_pct": 0.0, "max_return_pct": 0.0, "min_return_pct": 0.0,
            "cum_return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0,
        }

    ret = grp["return_pct"].astype(float)
    bench = grp["bench_return_pct"].astype(float) if "bench_return_pct" in grp else pd.Series([0.0] * total)
    excess = grp["excess_return_pct"].astype(float) if "excess_return_pct" in grp else ret

    wins = int(grp["win"].sum())
    excess_wins = int((excess > 0).sum())

    # ---- 净值类指标：基于按日历日去重叠的等权组合净值曲线 ----
    grp = grp.copy()
    grp.attrs["_calendar"] = _calendar_map(grp, calendar)
    _, nav = _daily_nav_curve(grp, value_col="excess_return_pct")

    if nav.size > 0:
        # 日收益（由净值反推）
        daily_rets = np.diff(nav) / nav[:-1]
        cum_return = float(nav[-1] / nav[0] - 1.0)
        dd = nav / np.maximum.accumulate(nav) - 1.0
        max_dd = float(dd.min())

        std = float(daily_rets.std(ddof=1)) if daily_rets.size > 1 else 0.0
        mean = float(daily_rets.mean()) if daily_rets.size > 0 else 0.0
        downside = daily_rets[daily_rets < 0]
        dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        # 年化：基于日收益，sqrt(252)
        sharpe = float(mean / std * (252.0 ** 0.5)) if std > 0 else 0.0
        sortino = float(mean / dstd * (252.0 ** 0.5)) if dstd > 0 else 0.0
    else:
        cum_return, max_dd, sharpe, sortino = 0.0, 0.0, 0.0, 0.0

    return {
        "total_trades": total,
        "win_count": wins,
        "win_rate": round(wins / total, 4),
        "excess_win_count": excess_wins,
        "excess_win_rate": round(excess_wins / total, 4),
        "avg_return_pct": round(float(ret.mean()), 4),
        "avg_excess_pct": round(float(excess.mean()), 4),
        "avg_bench_pct": round(float(bench.mean()), 4),
        "median_return_pct": round(float(ret.median()), 4),
        "max_return_pct": round(float(ret.max()), 4),
        "min_return_pct": round(float(ret.min()), 4),
        "cum_return_pct": round(cum_return * 100, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
    }


def summarize(
    trades: Iterable[dict],
    signals: Iterable[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把运行期累积的交易/信号转为四张表。

    返回 (detail, summary, combined, signals)：
        detail   : 逐笔明细（含基准与超额收益）
        summary  : 策略 × 数据源 × 持仓周期
        combined : 策略 × 持仓周期（跨数据源合并）
        signals  : 信号日志
    """
    detail = pd.DataFrame(list(trades))
    signals_df = pd.DataFrame(list(signals or []))

    if detail.empty:
        detail = pd.DataFrame(columns=DETAIL_COLUMNS)
    else:
        # 基准同期收益（沪深300 entry→exit，百分数口径对齐 return_pct）
        bench = benchmark_returns_for_trades(detail) * 100.0
        detail["bench_return_pct"] = bench.round(4)
        detail["excess_return_pct"] = (detail["return_pct"].astype(float)
                                       - detail["bench_return_pct"]).round(4)
        detail["excess_win"] = (detail["excess_return_pct"] > 0).astype(int)
        detail = detail[DETAIL_COLUMNS]

    # 全市场交易日历：detail 中出现的所有交易日并集（升序），用于净值摊薄
    calendar: list[str] = []
    if not detail.empty:
        calendar = sorted(set(detail["entry_date"].astype(str)) | set(detail["exit_date"].astype(str)))

    rows, rows_combined = [], []
    if not detail.empty:
        for (strat_name, lib_name, h), grp in detail.groupby(["strategy", "library", "holding_days"]):
            row = {"strategy": strat_name, "library": lib_name, "holding_days": int(h)}
            row.update(_metrics(grp, int(h), calendar))
            rows.append(row)
        for (strat_name, h), grp in detail.groupby(["strategy", "holding_days"]):
            row = {"strategy": strat_name, "holding_days": int(h)}
            row.update(_metrics(grp, int(h), calendar))
            rows_combined.append(row)

    summary = pd.DataFrame(rows)
    combined = pd.DataFrame(rows_combined)
    if not summary.empty:
        summary = summary.sort_values(["strategy", "library", "holding_days"]).reset_index(drop=True)
    if not combined.empty:
        combined = combined.sort_values(["strategy", "holding_days"]).reset_index(drop=True)

    if signals_df.empty and not signals_df.columns.tolist():
        signals_df = pd.DataFrame(columns=SIGNAL_COLUMNS)
    return detail, summary, combined, signals_df


def write_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    combined: pd.DataFrame,
    signals: pd.DataFrame,
    output_dir: Path | str,
    start_date: str,
    end_date: str,
) -> dict[str, Path]:
    """落盘四份 CSV，返回 {名称: 路径}。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": out / f"backtest_detail_{start_date}_{end_date}.csv",
        "summary": out / f"backtest_summary_{start_date}_{end_date}.csv",
        "combined": out / f"backtest_combined_{start_date}_{end_date}.csv",
        "signals": out / f"backtest_signals_{start_date}_{end_date}.csv",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    combined.to_csv(paths["combined"], index=False, encoding="utf-8-sig")
    signals.to_csv(paths["signals"], index=False, encoding="utf-8-sig")
    return paths
