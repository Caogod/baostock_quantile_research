"""持仓账本：记录选股信号 → 买入/卖出任务 → 收益率复盘。

设计约定（与用户需求一致）：
    - 信号日 = 选股日（selection 的 date，记为 signal_date）。
    - 买入日 = 信号日 + 1 个交易日，买入价 = 该日开盘价（open）。
    - 卖出日 = 信号日 + holding + 1 个交易日，卖出价 = 该日收盘价（close）。
    - 实际持有交易日数 = holding。

状态流转（每次运行 main 以 as_of_date 为"今天"推进）：
    pending_buy  买入日尚未到（未来），等待开盘买入
    holding      已到买入日、未到卖出日（持有中）
    settled      卖出日已到/已过，已平仓，收益率可复盘
    to_fill      已到期但价格尚未补全（本地库缺未来数据且无联网源时），
                 下次运行有数据后自动转 settled

数据源双模式：价格获取优先本地库（LocalDataLibrary），缺未来数据时回退
联网 DataFetcher 补全；两者都取不到则该记录留空、标记 to_fill 待补全。

账本文件：output/position_ledger.csv（追加式持久化，跨运行保留，便于统计）。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .strategy_loader import PROJECT_ROOT

_LEDGER_COLUMNS = [
    "signal_date", "buy_date", "sell_date",
    "code", "code_name", "strategy", "holding",
    "buy_price", "sell_price", "return_pct",
    "status", "score", "reason",
]


def _weekday_shift(date_str: str, offset: int) -> str:
    """按自然日 + 跳过周末做工作日偏移（无交易日历时的兜底）。"""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    step = 1 if offset >= 0 else -1
    moved = 0
    while moved < abs(offset):
        d += timedelta(days=step)
        if d.weekday() < 5:  # 周一~周五
            moved += 1
    return d.strftime("%Y-%m-%d")


class TradeCalendar:
    """交易日历：支持给定日期 ± N 个交易日的偏移。

    从多个数据源（本地库优先，联网兜底）按需加载日历；baostock 能返回
    精确的历史与未来交易日（含节假日），故联网日历最准。全部源都取不到
    时退化为「跳过周末」的近似工作日偏移（仅兜底，不识别节假日）。
    """

    def __init__(self, sources: list[Any], span_days: int = 60) -> None:
        self._dates: list[str] = []
        self._set: set[str] = set()
        self._loaded = False
        self._try_load(sources, span_days)

    def _try_load(self, sources: list[Any], span_days: int) -> None:
        lo = (datetime.now() - timedelta(days=span_days)).strftime("%Y-%m-%d")
        hi = (datetime.now() + timedelta(days=span_days)).strftime("%Y-%m-%d")
        for fetcher in sources:
            if fetcher is None:
                continue
            try:
                dates = fetcher.get_trade_dates(lo, hi)
                if dates:
                    self._dates = sorted(dates)
                    self._set = set(dates)
                    self._loaded = True
                    return
            except Exception:  # noqa: BLE001
                continue
        self._dates, self._set = [], set()

    @property
    def has_calendar(self) -> bool:
        return self._loaded and bool(self._dates)

    @property
    def max_date(self) -> str | None:
        """日历中的最大交易日（可能为未来）；无日历返回 None。"""
        return self._dates[-1] if self._dates else None

    def _find_next_idx(self, date_str: str) -> int | None:
        """返回日历中 >= date_str 的第一个交易日索引；找不到返回 None。"""
        for i, d in enumerate(self._dates):
            if d >= date_str:
                return i
        return None

    def shift(self, date_str: str, offset: int) -> str:
        """返回 date_str 之后第 offset 个交易日（offset>=0）。无日历则近似工作日。"""
        if offset == 0:
            return date_str
        if not self.has_calendar:
            return _weekday_shift(date_str, offset)
        idx = self._find_next_idx(date_str)
        if idx is None:
            return _weekday_shift(date_str, offset)
        # date_str 本身是交易日时，+1 应从其下一个开始；不是交易日时，起点即其后第一个交易日
        start = idx if date_str in self._set else idx - 1
        target = start + offset
        if 0 <= target < len(self._dates):
            return self._dates[target]
        return _weekday_shift(date_str, offset)


class PositionLedger:
    """持仓账本：记录信号、结算到期持仓、输出复盘与提醒。"""

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.path = Path(ledger_path) if ledger_path else PROJECT_ROOT / "output" / "position_ledger.csv"
        self.df = self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=_LEDGER_COLUMNS)
        try:
            df = pd.read_csv(self.path, dtype={"code": str, "code_name": str, "strategy": str})
        except Exception:  # noqa: BLE001
            return pd.DataFrame(columns=_LEDGER_COLUMNS)
        for col in _LEDGER_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[_LEDGER_COLUMNS]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ #
    def record_signals(
        self,
        selection_df: pd.DataFrame,
        holding_map: dict[str, int],
        calendar: TradeCalendar,
    ) -> int:
        """把选股信号登记为待买入记录，返回新增条数。

        selection_df 需含列: date, code, code_name, strategy, score, reason。
        holding_map: {strategy: holding 交易日数}。
        """
        if selection_df is None or selection_df.empty:
            return 0

        existing = set(zip(self.df["code"], self.df["signal_date"]))
        new_rows: list[dict] = []
        for _, row in selection_df.iterrows():
            sig_date = str(row["date"])
            code = str(row["code"])
            strategy = str(row["strategy"])
            holding = int(holding_map.get(strategy, 5))
            if (code, sig_date) in existing:
                continue  # 同一股票同一信号日已登记，避免重复

            new_rows.append({
                "signal_date": sig_date,
                "buy_date": calendar.shift(sig_date, 1),
                "sell_date": calendar.shift(sig_date, holding + 1),
                "code": code,
                "code_name": row.get("code_name", ""),
                "strategy": strategy,
                "holding": holding,
                "buy_price": None,
                "sell_price": None,
                "return_pct": None,
                "status": "pending_buy",
                "score": row.get("score", None),
                "reason": row.get("reason", ""),
            })

        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows, columns=_LEDGER_COLUMNS)],
                                ignore_index=True)
            self._save()
        return len(new_rows)

    # ------------------------------------------------------------------ #
    def settle(self, fetcher: Any, as_of_date: str) -> pd.DataFrame:
        """用单一数据源推进账本状态并结算到期持仓，返回已结算记录。

        对每条记录：
          - sell_date <= as_of_date：已到期，取 buy_date 开盘价 / sell_date 收盘价，
            算 return_pct，状态 → settled（价格取不到则标记 to_fill 待补全）。
          - buy_date <= as_of_date < sell_date：持有中，补买入价（若能取到），
            状态 → holding。
          - buy_date > as_of_date：保持 pending_buy（未来）。
        """
        if self.df.empty:
            return pd.DataFrame(columns=_LEDGER_COLUMNS)

        settled_rows: list[dict] = []
        for idx, row in self.df.iterrows():
            code = row["code"]
            sell_date = str(row["sell_date"])
            buy_date = str(row["buy_date"])

            if row["status"] == "settled":
                settled_rows.append(row.to_dict())
                continue

            if sell_date <= as_of_date:
                # 已到期 → 结算
                buy_price = self._price_at([fetcher], code, buy_date, "open")
                sell_price = self._price_at([fetcher], code, sell_date, "close")
                if buy_price is not None and sell_price is not None and buy_price > 0:
                    return_pct = round((sell_price / buy_price - 1.0) * 100.0, 4)
                    status = "settled"
                else:
                    return_pct = None
                    status = "to_fill"
                self.df.at[idx, "buy_price"] = buy_price
                self.df.at[idx, "sell_price"] = sell_price
                self.df.at[idx, "return_pct"] = return_pct
                self.df.at[idx, "status"] = status
                settled_rows.append(self.df.iloc[idx].to_dict())
            elif buy_date <= as_of_date:
                # 持有中 → 补买入价
                if row.get("buy_price") is None or pd.isna(row.get("buy_price")):
                    self.df.at[idx, "buy_price"] = self._price_at([fetcher], code, buy_date, "open")
                self.df.at[idx, "status"] = "holding"
            # else: 保持 pending_buy

        self._save()
        return pd.DataFrame(settled_rows, columns=_LEDGER_COLUMNS)

    # ------------------------------------------------------------------ #
    def pending_fill(self, as_of_date: str) -> pd.DataFrame:
        """返回「已到期但价格未补全」的记录（to_fill），供联网兜底补全。"""
        if self.df.empty:
            return pd.DataFrame(columns=_LEDGER_COLUMNS)
        mask = self.df["status"] == "to_fill"
        return self.df[mask].copy()

    # ------------------------------------------------------------------ #
    def fill_with_online(self, online_fetcher: Any, as_of_date: str) -> int:
        """用联网源补全 to_fill 记录的价格并转 settled，返回补全成功的条数。"""
        if self.df.empty or online_fetcher is None:
            return 0
        filled = 0
        for idx, row in self.df.iterrows():
            if row["status"] != "to_fill":
                continue
            code = row["code"]
            buy_date = str(row["buy_date"])
            sell_date = str(row["sell_date"])
            buy_price = self._price_at([online_fetcher], code, buy_date, "open")
            sell_price = self._price_at([online_fetcher], code, sell_date, "close")
            if buy_price is not None and sell_price is not None and buy_price > 0:
                self.df.at[idx, "buy_price"] = buy_price
                self.df.at[idx, "sell_price"] = sell_price
                self.df.at[idx, "return_pct"] = round((sell_price / buy_price - 1.0) * 100.0, 4)
                self.df.at[idx, "status"] = "settled"
                filled += 1
        if filled:
            self._save()
        return filled

    # ------------------------------------------------------------------ #
    @staticmethod
    def _price_at(sources: list[Any], code: str, date_str: str, field: str) -> float | None:
        """从数据源列表取某股票某交易日的 open/close 价（按顺序，首个非空生效）。"""
        for fetcher in sources:
            if fetcher is None:
                continue
            try:
                df = fetcher.get_history(code, date_str, date_str)
                if df is None or df.empty:
                    continue
                df = df[df["date"].astype(str) == date_str]
                if df.empty:
                    continue
                val = df.iloc[-1].get(field)
                if val is None or pd.isna(val):
                    continue
                return float(val)
            except Exception:  # noqa: BLE001
                continue
        return None

    # ------------------------------------------------------------------ #
    def reminders(self, as_of_date: str, calendar: TradeCalendar | None = None) -> dict[str, pd.DataFrame]:
        """按当前日期生成提醒：待买入 / 持有中 / 明日到期（提前一天提醒卖出）/
        今日到期卖出 / 待补全。

        提前提醒口径：卖出日 == as_of_date 的下一个交易日（即「明天到期」）时，
        今天发出「提前一天卖出提醒」；卖出日 == as_of_date（今天到期）时，仍给出
        当天卖出提醒。若未提供日历，则退化为「自然日 +1」近似。
        """
        empty = {"pending_buy": pd.DataFrame(), "holding": pd.DataFrame(),
                 "sell_tomorrow": pd.DataFrame(), "due_sell": pd.DataFrame(),
                 "to_fill": pd.DataFrame()}
        if self.df.empty:
            return empty

        pending = self.df[self.df["status"] == "pending_buy"].copy()
        holding = self.df[self.df["status"] == "holding"].copy()

        # 下一个交易日：有精确日历用日历偏移，否则自然日 +1（跳过周末兜底）
        if calendar is not None:
            next_trade_date = calendar.shift(as_of_date, 1)
        else:
            next_trade_date = _weekday_shift(as_of_date, 1)

        # 明日到期 = 卖出日 == 下一交易日 的持有记录（今天提前一天提醒卖出）
        sell_tomorrow = holding[holding["sell_date"].astype(str) == next_trade_date].copy()
        # 今日到期 = 卖出日 == 今天 的持有记录（今天当天卖出）
        due = holding[holding["sell_date"].astype(str) == as_of_date].copy()
        # 待补全 = 已到期但价格尚未补齐的记录
        to_fill = self.df[self.df["status"] == "to_fill"].copy()
        return {"pending_buy": pending, "holding": holding,
                "sell_tomorrow": sell_tomorrow, "due_sell": due, "to_fill": to_fill}

    # ------------------------------------------------------------------ #
    def review(self) -> pd.DataFrame:
        """按策略汇总已结算记录的收益率表现（复盘）。

        仅统计已有收益率的记录（status=settled 且 return_pct 非空）；
        to_fill（待补全）记录不计入，待价格补齐后下次运行自动纳入。
        """
        settled = self.df[self.df["status"] == "settled"].copy()
        if settled.empty:
            return pd.DataFrame(columns=["strategy", "trades", "win_rate", "avg_return", "total_return"])

        settled["return_pct"] = pd.to_numeric(settled["return_pct"], errors="coerce")
        valid = settled.dropna(subset=["return_pct"])
        if valid.empty:
            return pd.DataFrame(columns=["strategy", "trades", "win_rate", "avg_return", "total_return"])
        rows = []
        for strategy, g in valid.groupby("strategy"):
            ret = g["return_pct"]
            rows.append({
                "strategy": strategy,
                "trades": int(len(g)),
                "win_rate": round(float((ret > 0).mean()) * 100, 2),
                "avg_return": round(float(ret.mean()), 2),
                "total_return": round(float(ret.sum()), 2),
            })
        return pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)


def default_holding_map(strategies: list) -> dict[str, int]:
    """从策略实例列表提取 {strategy_name: holding}，缺省 5。"""
    return {name: int(getattr(strat, "params", {}).get("holding", 5))
            for name, strat in strategies}


def _has_new_signals(ledger: PositionLedger, selection: pd.DataFrame) -> bool:
    """判断 selection 中是否存在账本尚未登记的 (code, signal_date) 信号。"""
    if selection is None or selection.empty:
        return False
    if ledger.df.empty:
        return True
    existing = set(zip(ledger.df["code"].astype(str), ledger.df["signal_date"].astype(str)))
    for _, row in selection.iterrows():
        if (str(row["code"]), str(row["date"])) not in existing:
            return True
    return False


def _calendar_insufficient(calendar: TradeCalendar, selection: pd.DataFrame) -> bool:
    """判断本地日历是否不足以推算新信号的买入/卖出日。

    盘后选股时，信号日的 +1 买入、+holding+1 卖出都在未来，而本地库日历
    截止最新交易日，必然覆盖不到。用「日历最大日期是否晚于最晚信号日」
    做保守判断：只要存在信号日 >= 日历最大日期的信号，即认为日历不足。
    """
    if selection is None or selection.empty:
        return False  # 无新信号，无需未来日历
    if not calendar.has_calendar:
        return True
    max_cal = calendar.max_date
    if max_cal is None:
        return True
    latest_signal = max(str(d) for d in selection["date"])
    return latest_signal >= max_cal


def run_ledger_flow(
    fetcher: Any,
    selection_path: str | Path,
    strategies: list,
    as_of_date: str,
    ledger_path: str | Path | None = None,
    online_fetcher_factory: Any | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """main.py 的一次性调用入口：记录信号 + 结算 + 提醒 + 复盘。

    fetcher:                主数据源（本地库 LocalDataLibrary 或联网 DataFetcher）。
    online_fetcher_factory: 联网数据源工厂（callable，返回已 login 的联网 fetcher，
                            或 None 表示不支持联网）。**仅在本地库结算后仍有 to_fill
                            记录时才调用**，避免无效联网请求。

    流程：
      1. 交易日历：本地库优先；本地缺未来交易日时联网获取精确日历（按需）。
      2. 记录新信号（buy_date/sell_date 由精确交易日历推算，价格留空待补全）。
      3. 用主数据源结算到期持仓（本地库能取到的价格直接结算）。
      4. 若存在 to_fill 记录且提供了联网工厂，才联网补全这些记录的价格。
      5. 生成提醒与按策略复盘。

    返回 dict，含新增信号数、本次结算、提醒表、待补全、复盘表。
    """
    selection = pd.read_csv(selection_path) if Path(selection_path).exists() else pd.DataFrame()
    holding_map = default_holding_map(strategies)
    ledger = PositionLedger(ledger_path)

    # 交易日历：本地库优先；仅当存在「待登记的新信号」且本地日历不足以覆盖
    # 未来买入/卖出日时，才联网获取精确日历（按需，避免无效联网）。
    calendar = TradeCalendar([fetcher])
    if (online_fetcher_factory is not None
            and _has_new_signals(ledger, selection)
            and _calendar_insufficient(calendar, selection)):
        online = online_fetcher_factory()
        if online is not None:
            try:
                calendar = TradeCalendar([online, fetcher])
            finally:
                try:
                    online.logout()
                except Exception:  # noqa: BLE001
                    pass

    n_new = ledger.record_signals(selection, holding_map, calendar)
    settled = ledger.settle(fetcher, as_of_date)

    # 条件判断：仅当本地库结算后仍有「待补全」记录时才联网补价格
    n_online_filled = 0
    if online_fetcher_factory is not None:
        pending = ledger.pending_fill(as_of_date)
        if not pending.empty:
            online = online_fetcher_factory()
            if online is not None:
                try:
                    n_online_filled = ledger.fill_with_online(online, as_of_date)
                finally:
                    try:
                        online.logout()
                    except Exception:  # noqa: BLE001
                        pass

    reminders = ledger.reminders(as_of_date, calendar=calendar)
    review = ledger.review()

    if verbose:
        _print_flow(n_new, reminders, settled, review, as_of_date, n_online_filled)

    return {
        "n_new": n_new,
        "settled": settled,
        "n_online_filled": n_online_filled,
        "reminders": reminders,
        "review": review,
        "ledger": ledger,
    }


def _print_flow(n_new: int, reminders: dict, settled: pd.DataFrame,
                review: pd.DataFrame, as_of_date: str, n_online_filled: int = 0) -> None:
    print("\n" + "=" * 60)
    print(f"持仓账本（今天 {as_of_date}）")
    print("=" * 60)
    print(f"本次新增买入任务: {n_new} 条")
    if n_online_filled:
        print(f"联网补全价格: {n_online_filled} 条")

    if not reminders["pending_buy"].empty:
        print(f"\n[待买入] 共 {len(reminders['pending_buy'])} 条（次日开盘价买入）:")
        cols = ["code", "code_name", "strategy", "buy_date", "sell_date"]
        print(reminders["pending_buy"][cols].to_string(index=False))

    if not reminders["holding"].empty:
        print(f"\n[持有中] 共 {len(reminders['holding'])} 条:")
        cols = ["code", "code_name", "strategy", "buy_date", "sell_date", "buy_price"]
        print(reminders["holding"][cols].to_string(index=False))

    if not reminders["sell_tomorrow"].empty:
        print(f"\n[提前一天提醒卖出] 共 {len(reminders['sell_tomorrow'])} 条（明日收盘价卖出）:")
        cols = ["code", "code_name", "strategy", "sell_date", "buy_price"]
        print(reminders["sell_tomorrow"][cols].to_string(index=False))

    if not reminders["due_sell"].empty:
        print(f"\n[今日到期卖出] 共 {len(reminders['due_sell'])} 条（收盘价卖出）:")
        cols = ["code", "code_name", "strategy", "buy_price"]
        print(reminders["due_sell"][cols].to_string(index=False))

    if not reminders["to_fill"].empty:
        print(f"\n[待补全数据] 共 {len(reminders['to_fill'])} 条（已到期但价格待补齐）:")
        cols = ["code", "code_name", "strategy", "buy_date", "sell_date"]
        print(reminders["to_fill"][cols].to_string(index=False))

    if not settled.empty:
        print(f"\n[本次结算] 共 {len(settled)} 条:")
        cols = ["code", "code_name", "strategy", "buy_price", "sell_price", "return_pct"]
        print(settled[cols].to_string(index=False))

    if not review.empty:
        print("\n[按策略复盘收益率]")
        print(review.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(0)
