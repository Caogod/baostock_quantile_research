"""策略：连续 5 日收阳 + 5 日累计涨幅超阈值（动量加速型）。

筛选条件（全部满足）：
    1. 最近 5 个交易日**每日收盘价均高于前一日收盘价**（连续 5 根阳线推进）；
    2. 这 5 日的累计涨幅 > min_gain_pct（默认 20%）；
    3. 非 ST / 非退市（优先用日线逐日 isST，无前视偏差）；
    4. 非次新股：上市满 min_listed_days 个自然日（默认 180 天）；
    5. 非科创板（代码 sh.688 开头，可通过 exclude_boards 调整）。

【条件顺序：按过滤代价由低到高手工排定（2026-10-12 取消向量化预筛）】
    全部条件为 AND 关系，顺序不影响判定结果，只影响性能：
        常量（板块 / ST / 上市天数）
        → 累计涨幅（单次除法，O(1)，且过滤力最强）
        → 连续收阳（window 次相邻比较，O(window)）
    把过滤力最强的累计涨幅提到连续收阳之前，可让绝大多数样本在第一、
    第二个条件就被挡掉，避免后续 diff 与切片计算。

说明：
- 「连续 5 天收益率连续正」按收盘价口径实现（close[i] > close[i-1]），
  与 pctChg > 0 等价；用收盘价可避免个别日期 pctChg 缺字段的问题。
- 上市天数用证券元数据的 ipoDate 换算日历天数，与 b1_oversold 口径一致。
- 科创板默认剔除（688 开头）；如需放宽可把 exclude_boards 设为 ""。

参数：
    window        : 5     连续上涨天数与累计涨幅窗口
    min_gain_pct  : 20.0  窗口内累计涨幅下限(%)
    exclude_st    : True  是否剔除 ST / 退市
    min_listed_days : 180 上市天数下限（自然日，0 表示不校验）
    exclude_boards : "688"  需剔除的板块代码前缀（逗号分隔，空表示不剔除）
    min_bars      : 65    单只股票所需最小历史长度
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "consecutive_up5"
    description = "连续5日上涨且5日累计涨幅>20%（非ST/非次新/非科创）"

    # ------------------------------------------------------------------ #
    # 上下文（上市天数需要 ipoDate）
    # ------------------------------------------------------------------ #
    def bind_context(self, ctx: dict | None) -> None:
        ctx = ctx or {}
        self._meta = ctx.get("meta")

    def _meta_value(self, code: str, column: str) -> str:
        meta = getattr(self, "_meta", None)
        if meta is None or code not in getattr(meta, "index", []):
            return ""
        try:
            val = meta.loc[code, column]
        except Exception:  # noqa: BLE001
            return ""
        return "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)

    # ------------------------------------------------------------------ #
    def signal(self, df: pd.DataFrame) -> dict:
        p = self.params
        window = int(p.get("window", 5))
        min_gain_pct = float(p.get("min_gain_pct", 20.0))
        exclude_st = bool(p.get("exclude_st", True))
        min_listed_days = int(p.get("min_listed_days", 180))
        exclude_boards = [b.strip() for b in str(p.get("exclude_boards", "688")).split(",") if b.strip()]
        min_bars = int(p.get("min_bars", 65))

        n = len(df)
        if n < max(min_bars, window + 1):
            return self._no_signal(f"历史不足{max(min_bars, window + 1)}日({n})")

        code = str(df["code"].iloc[-1]) if "code" in df.columns else ""
        num = code.split(".")[-1]

        # ---- 条件 5：非科创板（板块前缀过滤）----
        if exclude_boards and any(num.startswith(b) for b in exclude_boards):
            return self._no_signal(f"板块排除({num})")

        # ---- 条件 3：非 ST / 非退市（优先用逐日 isST，无前视偏差）----
        if exclude_st:
            if "isST" in df.columns:
                if str(df["isST"].iloc[-1]).strip() in {"1", "1.0", "True", "true"}:
                    return self._no_signal("ST/风险警示股")
            else:
                name = self._meta_value(code, "name")
                if ("ST" in name) or ("退" in name):
                    return self._no_signal(f"ST/退市({name})")

        # ---- 条件 4：非 180 天内新股 ----
        if min_listed_days > 0:
            ipo = self._meta_value(code, "ipoDate")
            if ipo:
                try:
                    d0 = datetime.strptime(ipo, "%Y-%m-%d")
                    d1 = datetime.strptime(str(df["date"].iloc[-1]), "%Y-%m-%d")
                    if (d1 - d0).days < min_listed_days:
                        return self._no_signal(f"上市未满{min_listed_days}天({ipo})")
                except Exception:  # noqa: BLE001
                    pass

        close = df["close"].astype(float).reset_index(drop=True)
        i = n - 1
        last_close = float(close.iloc[i])

        # ============================================================== #
        # 条件顺序：按「过滤代价」由低到高手工排定（2026-10-12 取消向量化预筛）
        # 全部条件为 AND 关系，顺序不影响判定结果，只影响性能：
        #   常量（板块/ST/上市天数）→ O(1) 累计涨幅（单次除法，且过滤力最强）
        #   → O(window) 连续收阳（window 次相邻比较）
        # 累计涨幅 > 20% 是三者中最强的筛选条件，先执行能挡掉绝大多数样本。
        # ============================================================== #

        # ---- 1) 累计涨幅 > min_gain_pct（O(1)，过滤力最强）----
        base_close = float(close.iloc[i - window])
        if base_close <= 0:
            return self._no_signal("基准价异常")
        gain_pct = (last_close / base_close - 1.0) * 100.0
        if gain_pct <= min_gain_pct:
            return self._no_signal(
                f"近{window}日累计涨幅{gain_pct:.2f}%≤{min_gain_pct}%")

        # ---- 2) 最近 window 日每日收盘 > 前一日收盘（O(window) 相邻比较）----
        diffs = close.diff().iloc[i - window + 1: i + 1]
        if diffs.isna().any():
            return self._no_signal("连续性数据不足")
        if not bool((diffs > 0).all()):
            n_up = int((diffs > 0).sum())
            return self._no_signal(f"近{window}日仅{n_up}日收阳，未连续上涨")

        # ---- 打分：累计涨幅为主，短期加速（末日涨幅）为辅 ----
        last_day_pct = float(diffs.iloc[-1] / close.iloc[i - 1] * 100.0)
        score = round(gain_pct + last_day_pct, 4)
        return {
            "buy": True,
            "score": score,
            "reason": (
                f"连续{window}日收阳；近{window}日累计涨幅{gain_pct:.2f}%>{min_gain_pct}%；"
                f"末日涨幅{last_day_pct:.2f}%；收盘{last_close:.2f}"
            ),
        }
