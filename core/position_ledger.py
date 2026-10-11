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
    "status", "score", "win_prob", "reason",
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

    def has_offset(self, date_str: str, offset: int) -> bool:
        """判断日历中 date_str 之后是否还有 >= offset 个交易日。

        用于「日历够不够推算未来买入/卖出日」的判断：盘后选股时卖出日在未来，
        光看"日历最大日期 >= 信号日"是不够的——还得能往后推出 holding+1 个交易日。
        """
        if not self.has_calendar:
            return False
        return sum(1 for d in self._dates if d > date_str) >= offset

    def is_trading_day(self, date_str: str) -> bool:
        """判断该日期是否为日历中的真实交易日。

        无日历时不做校验（返回 True），避免因日历缺失而误杀正常信号。
        """
        if not self.has_calendar:
            return True
        return date_str in self._set

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
        # 本次运行发生"停牌顺延"的记录：[{code, plan_sell_date, actual_sell_date}, ...]
        self.rollforward_log: list[dict] = []

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
    ) -> tuple[int, list[str]]:
        """把选股信号登记为待买入记录，返回 (新增条数, 被跳过的非交易日信号)。

        selection_df 需含列: date, code, code_name, strategy, score, reason。
        holding_map: {strategy: holding 交易日数}。

        非交易日信号（如选股日误取当天、而当天尚无行情）会被跳过并返回，
        避免把 score/close 实为上一交易日、日期却标成未来日期的脏记录写进账本。
        """
        if selection_df is None or selection_df.empty:
            return 0, []

        existing = set(zip(self.df["code"].astype(str), self.df["signal_date"].astype(str)))
        new_rows: list[dict] = []
        backfilled = False
        skipped: list[str] = []
        for _, row in selection_df.iterrows():
            sig_date = str(row["date"])
            code = str(row["code"])
            strategy = str(row["strategy"])
            holding = int(holding_map.get(strategy, 5))

            # 信号日必须是真实交易日，否则 buy/sell 日期推算全部失真
            if not calendar.is_trading_day(sig_date):
                skipped.append(f"{code}@{sig_date}")
                continue

            if (code, sig_date) in existing:
                # 已登记：不重复添加，但若本次带 win_prob 而账本为空则回填。
                # 场景：先用不带 --filter 跑过、或补跑历史日期时，需要把预测
                # 胜率补进既有记录，否则这些记录永远缺 win_prob。
                wp = row.get("win_prob", None)
                if wp is not None and not pd.isna(wp):
                    mask = (self.df["code"].astype(str) == code) & \
                           (self.df["signal_date"].astype(str) == sig_date)
                    cur = pd.to_numeric(self.df.loc[mask, "win_prob"], errors="coerce")
                    if not mask.any() or cur.isna().all():
                        self.df.loc[mask, "win_prob"] = float(wp)
                        backfilled = True
                continue

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
                # 二次过滤（胜率模型）预测概率；未启用 --filter 时为 None
                "win_prob": row.get("win_prob", None),
                "reason": row.get("reason", ""),
            })

        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows, columns=_LEDGER_COLUMNS)],
                                ignore_index=True)
            self._save()
        elif backfilled:
            self._save()
        return len(new_rows), skipped

    # ------------------------------------------------------------------ #
    @staticmethod
    def _price_with_rollforward(
        sources: list[Any],
        code: str,
        date_str: str,
        field: str,
        as_of_date: str,
        max_lookahead: int = 10,
    ) -> tuple[float | None, str | None]:
        """取某交易日价格；若该日无行情（停牌/数据缺失），向后顺延到最近的有数据日。

        停牌等情况下原定卖出日没有成交价，此时**顺延到最近的有行情交易日**取价，
        而不是把记录长期挂在 to_fill。返回 (价格, 实际成交日)；完全取不到返回 (None, None)。

        顺延上限 max_lookahead 个自然日（默认 10，足以覆盖国庆等长假），
        且实际成交日不得超过 as_of_date（不能取"尚未发生"的数据）。
        """
        # 先按原定日期取价（绝大多数情况走这条）
        price = PositionLedger._price_at(sources, code, date_str, field)
        if price is not None and price > 0:
            return price, date_str

        # 原定日期无行情 → 在窗口内向后找最近的有数据日
        end = (datetime.strptime(date_str, "%Y-%m-%d")
               + timedelta(days=max_lookahead)).strftime("%Y-%m-%d")
        for fetcher in sources:
            if fetcher is None:
                continue
            try:
                df = fetcher.get_history(code, date_str, end)
                if df is None or df.empty:
                    continue
                df = df[df["date"].astype(str) >= date_str]
                df = df[df["date"].astype(str) <= as_of_date]  # 不取未来数据
                if df.empty or field not in df.columns:
                    continue
                df = df[df[field].notna()]
                df = df[df[field].astype(float) > 0]
                if df.empty:
                    continue
                last = df.sort_values("date").iloc[-1]
                return float(last[field]), str(last["date"])
            except Exception:  # noqa: BLE001
                continue
        return None, None

    def settle(self, fetcher: Any, as_of_date: str,
               max_lookahead: int = 10) -> pd.DataFrame:
        """用单一数据源推进账本状态并结算到期持仓，返回**本次新结算**的记录。

        对每条记录：
          - sell_date <= as_of_date：已到期，取 buy_date 开盘价 / sell_date 收盘价，
            算 return_pct，状态 → settled。卖出日无行情（停牌）时顺延到最近的有
            行情交易日取价，并同步把 sell_date 更新为实际成交日。
            买卖价仍取不到则标记 to_fill 待补全。
          - buy_date <= as_of_date < sell_date：持有中，补买入价（若能取到），
            状态 → holding。
          - buy_date > as_of_date：保持 pending_buy（未来）。

        注意：早已 settled 的历史记录**不会**重复出现在返回值中，避免把
        "累计结算" 误读成 "本次新结算"。
        """
        if self.df.empty:
            return pd.DataFrame(columns=_LEDGER_COLUMNS)

        settled_rows: list[dict] = []
        for idx, row in self.df.iterrows():
            code = row["code"]
            sell_date = str(row["sell_date"])
            buy_date = str(row["buy_date"])

            if row["status"] == "settled":
                continue  # 早已结算完毕，不计入"本次结算"

            if sell_date <= as_of_date:
                # 已到期 → 结算（卖出价允许顺延到最近有行情的交易日）
                buy_price = self._price_at([fetcher], code, buy_date, "open")
                sell_price, actual_sell = self._price_with_rollforward(
                    [fetcher], code, sell_date, "close", as_of_date, max_lookahead)
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
                # 停牌顺延时同步实际成交日，保持账本与实际一致
                if actual_sell and actual_sell != sell_date:
                    self.df.at[idx, "sell_date"] = actual_sell
                    self.rollforward_log.append(
                        {"code": code, "plan_sell_date": sell_date,
                         "actual_sell_date": actual_sell})
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
    def fill_with_online(self, online_fetcher: Any, as_of_date: str,
                         max_lookahead: int = 10) -> int:
        """用联网源补全 to_fill 记录的价格并转 settled，返回补全成功的条数。

        与 settle 一样支持卖出日顺延：联网源上若原定卖出日仍无行情（停牌），
        顺延到最近的有行情交易日取收盘价。
        """
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
            sell_price, actual_sell = self._price_with_rollforward(
                [online_fetcher], code, sell_date, "close", as_of_date, max_lookahead)
            if buy_price is not None and sell_price is not None and buy_price > 0:
                self.df.at[idx, "buy_price"] = buy_price
                self.df.at[idx, "sell_price"] = sell_price
                self.df.at[idx, "return_pct"] = round((sell_price / buy_price - 1.0) * 100.0, 4)
                self.df.at[idx, "status"] = "settled"
                if actual_sell and actual_sell != sell_date:
                    self.df.at[idx, "sell_date"] = actual_sell
                    self.rollforward_log.append(
                        {"code": code, "plan_sell_date": sell_date,
                         "actual_sell_date": actual_sell})
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
    def settlement_coverage(self, as_of_date: str) -> pd.DataFrame:
        """结算覆盖率：按策略统计「已到期记录」中完成收益结算的比例。

        口径说明：
          total      该策略账本全部记录
          due        应结算 = sell_date <= as_of_date（已到卖出日）
          settled    已结算 = status 为 settled 且 return_pct 有效
          to_fill    已到期但价格缺失，阻塞了结算（需联网补价）
          holding    持有中（未到卖出日，不应计入分母）
          pending_buy 买入日未到（未到期）
          coverage   **结算覆盖率** = settled / due（到期记录中已算出收益的比例）
          progress   总体进度 = settled / total（含未到期记录的分母口径）

        coverage 才是真正衡量"结算是否覆盖得住"的口径：to_fill 越多、coverage 越低，
        说明账本里有收益算不出来的空洞。
        """
        cols = ["strategy", "total", "due", "settled", "to_fill", "holding",
                "pending_buy", "coverage", "progress"]
        if self.df.empty:
            return pd.DataFrame(columns=cols)

        df = self.df.copy()
        df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
        df["sell_date"] = df["sell_date"].astype(str)
        # 已完成结算：有有效收益率（比只看 status 更严格，防止脏状态蒙过）
        df["_settled"] = (df["status"] == "settled") & df["return_pct"].notna()
        df["_due"] = df["sell_date"] <= as_of_date
        df["_to_fill"] = df["status"] == "to_fill"
        df["_holding"] = df["status"] == "holding"
        df["_pending"] = df["status"] == "pending_buy"

        rows = []
        # 末行追加"合计"，便于一眼看全局
        for strategy, g in list(df.groupby("strategy")) + [("__total__", df)]:
            due = int(g["_due"].sum())
            settled = int(g["_settled"].sum())
            total = int(len(g))
            rows.append({
                "strategy": strategy,
                "total": total,
                "due": due,
                "settled": settled,
                "to_fill": int(g["_to_fill"].sum()),
                "holding": int(g["_holding"].sum()),
                "pending_buy": int(g["_pending"].sum()),
                "coverage": round(settled / due * 100, 2) if due else None,
                "progress": round(settled / total * 100, 2) if total else None,
            })
        return pd.DataFrame(rows, columns=cols)

    def monthly_returns(self) -> pd.DataFrame:
        """按卖出月份汇总已结算收益（观察收益的时间分布与滚动兑现情况）。"""
        cols = ["month", "trades", "win_rate", "avg_return", "total_return",
                "best", "worst"]
        settled = self.df[self.df["status"] == "settled"].copy()
        if settled.empty:
            return pd.DataFrame(columns=cols)
        settled["return_pct"] = pd.to_numeric(settled["return_pct"], errors="coerce")
        settled = settled.dropna(subset=["return_pct"])
        if settled.empty:
            return pd.DataFrame(columns=cols)
        settled["month"] = settled["sell_date"].astype(str).str.slice(0, 7)

        rows = []
        for month, g in settled.groupby("month"):
            ret = g["return_pct"]
            rows.append({
                "month": month,
                "trades": int(len(g)),
                "win_rate": round(float((ret > 0).mean()) * 100, 2),
                "avg_return": round(float(ret.mean()), 2),
                "total_return": round(float(ret.sum()), 2),
                "best": round(float(ret.max()), 2),
                "worst": round(float(ret.min()), 2),
            })
        return pd.DataFrame(rows, columns=cols).sort_values("month").reset_index(drop=True)

    def review(self, high_threshold: float = 0.5) -> pd.DataFrame:
        """按策略汇总已结算记录的收益率表现（复盘）。

        仅统计已有收益率的记录（status=settled 且 return_pct 非空）；
        to_fill（待补全）记录不计入，待价格补齐后下次运行自动纳入。

        二次过滤的预测效果用**分组实际收益**衡量，而非预测值均值：
          high_*  win_prob >= high_threshold 的记录（模型看多）的实际表现
          low_*   win_prob <  high_threshold 的记录（模型看空）的实际表现
        若 high_avg_return 明显高于 low_avg_return，说明预测有区分度；
        若两者接近甚至倒挂，则该模型对本策略无效，应考虑停用二次过滤。
        avg_win_prob（预测值均值）已移除——它只反映模型输出的分布，
        与实际盈亏无关，看不出模型好坏。
        """
        cols = ["strategy", "trades", "win_rate", "avg_return", "total_return",
                "high_trades", "high_win_rate", "high_avg_return", "high_total_return",
                "low_trades", "low_win_rate", "low_avg_return", "low_total_return",
                "prob_edge"]
        settled = self.df[self.df["status"] == "settled"].copy()
        if settled.empty:
            return pd.DataFrame(columns=cols)

        settled["return_pct"] = pd.to_numeric(settled["return_pct"], errors="coerce")
        settled["win_prob"] = pd.to_numeric(settled.get("win_prob"), errors="coerce")
        valid = settled.dropna(subset=["return_pct"])
        if valid.empty:
            return pd.DataFrame(columns=cols)

        def _grp(g: pd.DataFrame) -> dict:
            r = g["return_pct"]
            if r.empty:
                return {"trades": 0, "win_rate": None, "avg_return": None,
                        "total_return": None}
            return {
                "trades": int(len(g)),
                "win_rate": round(float((r > 0).mean()) * 100, 2),
                "avg_return": round(float(r.mean()), 2),
                "total_return": round(float(r.sum()), 2),
            }

        rows = []
        for strategy, g in valid.groupby("strategy"):
            has_prob = g["win_prob"].notna().any()
            # 无 win_prob（全NaN）时不做分组，两组都留空，避免把"没数据"当成"模型看空"
            hi = g[g["win_prob"] >= high_threshold] if has_prob else g.iloc[0:0]
            lo = g[g["win_prob"] < high_threshold] if has_prob else g.iloc[0:0]
            h, low = _grp(hi), _grp(lo)
            edge = (round(h["avg_return"] - low["avg_return"], 2)
                    if h["avg_return"] is not None and low["avg_return"] is not None else None)
            rows.append({
                "strategy": strategy,
                **_grp(g),
                "high_trades": h["trades"],
                "high_win_rate": h["win_rate"],
                "high_avg_return": h["avg_return"],
                "high_total_return": h["total_return"],
                "low_trades": low["trades"],
                "low_win_rate": low["win_rate"],
                "low_avg_return": low["avg_return"],
                "low_total_return": low["total_return"],
                # 正=模型有区分度；负/0=模型无效
                "prob_edge": edge,
            })
        return pd.DataFrame(rows, columns=cols).sort_values(
            "total_return", ascending=False).reset_index(drop=True)


def default_holding_map(strategies: list) -> dict[str, int]:
    """从策略实例列表提取 {strategy_name: holding}，缺省 5。"""
    return {name: int(getattr(strat, "params", {}).get("holding", 5))
            for name, strat in strategies}


def export_review(ledger: "PositionLedger", out_dir: str | Path,
                  as_of_date: str, basename: str = "ledger_review") -> list[Path]:
    """把复盘结果导出为**多个单表 CSV**，每个文件都是「一行表头 + 数据行」。

    为什么不写成一个文件：早前把各段落用 `#` 标题和空行拼进同一个 csv，
    Excel 打开后会出现大量空行隔断、每段表头样式也不对，机器读取同样困难。
    这里改为按内容拆分成独立文件，各自结构规整、可直接透视。

    生成文件：
      <basename>_settled.csv    已结算明细（含收益、预测胜率）
      <basename>_open.csv       未到期（持有中 / 待买入）
      <basename>_to_fill.csv    已到期但价格待补齐（无则不生成）
      <basename>_coverage.csv   结算覆盖率
      <basename>_strategy.csv   按策略收益汇总（含 high/low 分组收益）
      <basename>_monthly.csv    按月收益兑现

    返回实际生成的文件路径列表。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = ledger.df.copy()
    if df.empty:
        return []
    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    df["win_prob"] = pd.to_numeric(df["win_prob"], errors="coerce")

    base_cols = ["signal_date", "buy_date", "sell_date", "code", "code_name",
                 "strategy", "status"]
    written: list[Path] = []

    def _dump(frame: pd.DataFrame, suffix: str) -> None:
        if frame is None or frame.empty:
            return
        path = out / f"{basename}_{suffix}.csv"
        # 固定列顺序（存在的才保留），不写 index、不留空行
        frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
        written.append(path)

    settled = df[df["status"] == "settled"].sort_values("sell_date")
    _dump(settled[base_cols + ["holding", "buy_price", "sell_price",
                                "return_pct", "win_prob", "score", "reason"]],
          "settled")

    open_pos = df[df["status"].isin(["holding", "pending_buy"])].sort_values("sell_date")
    _dump(open_pos[base_cols + ["holding", "buy_price", "win_prob"]], "open")

    _dump(df[df["status"] == "to_fill"].sort_values("sell_date"),
          "to_fill")
    _dump(ledger.settlement_coverage(as_of_date), "coverage")
    _dump(ledger.review(), "strategy")
    _dump(ledger.monthly_returns(), "monthly")
    return written


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


def _calendar_insufficient(calendar: TradeCalendar, selection: pd.DataFrame,
                          max_holding: int = 5) -> bool:
    """判断本地日历是否不足以推算新信号的买入/卖出日。

    盘后选股时，信号日的 +1 买入、+holding+1 卖出都在未来，而本地库日历截止最新
    交易日，必然覆盖不到。**关键**：不能只比较"信号日 vs 日历最大日期"——那样会
    在日历"刚好差一点"时误判为充足，进而用「跳过周末」的兜底偏移算出 10-01
    这类国庆假期的假交易日。正确做法是要求日历在信号日之后至少还能提供
    max_holding+1 个交易日。
    """
    if selection is None or selection.empty:
        return False  # 无新信号，无需未来日历
    if not calendar.has_calendar:
        return True
    latest_signal = max(str(d) for d in selection["date"])
    return not calendar.has_offset(latest_signal, max_holding + 1)


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
    # 未来买入/卖出日（信号日之后还需 max_holding+1 个交易日）时，才联网获取
    # 精确日历（按需，避免无效联网）。
    calendar = TradeCalendar([fetcher])
    max_holding = max(holding_map.values()) if holding_map else 5
    if (online_fetcher_factory is not None
            and _has_new_signals(ledger, selection)
            and _calendar_insufficient(calendar, selection, max_holding)):
        online = online_fetcher_factory()
        if online is not None:
            try:
                calendar = TradeCalendar([online, fetcher])
            finally:
                try:
                    online.logout()
                except Exception:  # noqa: BLE001
                    pass

    n_new, skipped_signals = ledger.record_signals(selection, holding_map, calendar)
    if skipped_signals:
        print(f"[warn] 以下信号的 signal_date 不是真实交易日（当天无行情），已跳过登记: "
              f"{skipped_signals}", file=sys.stderr)
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
    coverage = ledger.settlement_coverage(as_of_date)
    monthly = ledger.monthly_returns()

    # 复盘结果导出为规整的单表 CSV（每份都是一行表头 + 数据行，无空行隔断）
    export_paths = export_review(ledger, PROJECT_ROOT / "output", as_of_date)

    if verbose:
        _print_flow(n_new, reminders, settled, review, as_of_date, n_online_filled,
                    coverage=coverage, monthly=monthly, export_paths=export_paths)
        _print_rollforward(ledger.rollforward_log)

    return {
        "n_new": n_new,
        "settled": settled,
        "n_online_filled": n_online_filled,
        "reminders": reminders,
        "review": review,
        "coverage": coverage,
        "monthly": monthly,
        "export_paths": export_paths,
        "rollforward": list(ledger.rollforward_log),
        "ledger": ledger,
    }


def _print_rollforward(rollforward: list[dict]) -> None:
    """打印因停牌/无行情而顺延卖出日的记录（顺延后的实际成交日已写回账本）。"""
    if not rollforward:
        return
    print(f"\n[卖出日顺延] 共 {len(rollforward)} 条（原定卖出日无行情，顺延至最近有数据的交易日）:")
    print(pd.DataFrame(rollforward).to_string(index=False))


def _with_prob(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """列选择器：仅当 win_prob 至少有一个有效值时才附加上该列。"""
    if "win_prob" not in df.columns:
        return cols
    prob = pd.to_numeric(df["win_prob"], errors="coerce")
    return cols + ["win_prob"] if prob.notna().any() else cols


def _print_flow(n_new: int, reminders: dict, settled: pd.DataFrame,
                review: pd.DataFrame, as_of_date: str, n_online_filled: int = 0,
                coverage: pd.DataFrame | None = None,
                monthly: pd.DataFrame | None = None,
                export_paths: list[Path] | None = None) -> None:
    print("\n" + "=" * 60)
    print(f"持仓账本（今天 {as_of_date}）")
    print("=" * 60)
    print(f"本次新增买入任务: {n_new} 条")
    if n_online_filled:
        print(f"联网补全价格: {n_online_filled} 条")

    if not reminders["pending_buy"].empty:
        df = reminders["pending_buy"]
        print(f"\n[待买入] 共 {len(df)} 条（次日开盘价买入）:")
        print(df[_with_prob(df, ["code", "code_name", "strategy", "buy_date",
                                 "sell_date"])].to_string(index=False))

    if not reminders["holding"].empty:
        df = reminders["holding"]
        print(f"\n[持有中] 共 {len(df)} 条:")
        print(df[_with_prob(df, ["code", "code_name", "strategy", "buy_date",
                                 "sell_date", "buy_price"])].to_string(index=False))

    if not reminders["sell_tomorrow"].empty:
        df = reminders["sell_tomorrow"]
        print(f"\n[提前一天提醒卖出] 共 {len(df)} 条（明日收盘价卖出）:")
        print(df[_with_prob(df, ["code", "code_name", "strategy", "sell_date",
                                 "buy_price"])].to_string(index=False))

    if not reminders["due_sell"].empty:
        df = reminders["due_sell"]
        print(f"\n[今日到期卖出] 共 {len(df)} 条（收盘价卖出）:")
        print(df[_with_prob(df, ["code", "code_name", "strategy",
                                 "buy_price"])].to_string(index=False))

    if not reminders["to_fill"].empty:
        df = reminders["to_fill"]
        print(f"\n[待补全数据] 共 {len(df)} 条（已到期但价格待补齐）:")
        print(df[_with_prob(df, ["code", "code_name", "strategy", "buy_date",
                                 "sell_date"])].to_string(index=False))

    if not settled.empty:
        print(f"\n[本次结算] 共 {len(settled)} 条:")
        print(settled[_with_prob(settled, ["code", "code_name", "strategy", "buy_price",
                                           "sell_price", "return_pct"])].to_string(index=False))

    if coverage is not None and not coverage.empty:
        total_row = coverage[coverage["strategy"] == "__total__"]
        if not total_row.empty:
            r = total_row.iloc[0]
            print(f"\n[结算覆盖率] 已到期 {int(r['due'])} 条 / 账本共 {int(r['total'])} 条，"
                  f"已结算 {int(r['settled'])} 条")
        print(f"\n[结算覆盖率明细]（coverage = 已结算 / 已到期）:")
        print(coverage.to_string(index=False))

    if review is not None and not review.empty:
        print("\n[按策略复盘收益率]")
        print(review.to_string(index=False))

    if monthly is not None and not monthly.empty:
        print("\n[按月收益兑现]")
        print(monthly.to_string(index=False))

    if export_paths:
        print(f"\n[复盘导出] {len(export_paths)} 个 CSV（一表一份，无空行隔断）:")
        for p in export_paths:
            print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(0)
