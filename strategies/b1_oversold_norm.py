"""升级策略：B1 稳健超跌 + 归一化超跌门槛（b1_oversold 的增强版）。

本策略**不改动** `b1_oversold` 本身，而是复用它的全部条件，在其前面加一道
**归一化超跌门槛**——来自 `docs/归一化特征信号研究.md` 的全库分档筛选结论：
归一化维度里唯一有信号的就是"超跌"，且是**阈值型**（只有 Q1 有效）。

为什么这样组合
--------------
- `b1_oversold` 是**日内形态型超跌**：RSI6<20、当日振幅>8%、连跌 2 日、全市场下跌。
  它刻画的是"当天的恐慌形态"，但不看**中期位置**——一只刚冲高回落的强势股
  和一只已经跌了两个月的弱势股，只要当天形态符合，b1 一视同仁。
- 归一化超跌条件（MA5/MA60、close/MA60 低）补的正是**中期位置**这一维：
  股价显著低于 60 日均线，即"已经跌到中期均线下方很深"。
- 研究报告显示该维度单独用于全库时，单笔均值为正但组合口径为负（信号同时聚集），
  所以它**不适合做独立策略，更适合做正交过滤器**——这正是本策略的用法。

条件顺序（性能）
    归一化比值只需 1 次 60 元素尾切片 + 2 次均值，且筛掉约 87% 的候选，
    因此放在最前面；通过后再交给 b1 原逻辑（含较贵的 RSI6 EWM，仍在其内部最后执行）。

阈值参考（全库 5 档分位，147,984 个股票日）：
    MA5/MA60   Q1 上界 0.9058（50.23% 胜率 / +0.375%）→ Q2 上界 0.9575（43.10% / -0.654%）
    close/MA60 Q1 上界 0.8995（50.04% / +0.416%）      → Q2 上界 0.9543（43.64% / -0.659%）
    MA20 斜率  Q1 上界 -2.1382（50.20% / +0.425%）     → Q2 上界 -0.7307（44.44% / -0.487%）
**务必用 Q1 档**：Q2 档的均收益本身已转负，放宽会把 edge 稀释掉
（已用独立回测验证：Q2 版均收益 -0.132%，与全库基准 -0.103% 无差别）。

参数：
    r_ma5_ma60_max  : 0.9058  MA5/MA60 上限（Q1）
    r_ma60_max      : 0.8995  close/MA60 上限（Q1）
    use_ma5_ma60    : True    是否启用 MA5/MA60 条件
    use_ma60        : True    是否启用 close/MA60 条件
    slope_ma20_5_max: None    MA20 五日斜率上限(%)；设数值即启用（推荐 -2.1382）
    其余参数（rsi_period / rsi_max / amp_min / price_min / lu_window / rmax_window /
    zt_main / zt_gem / boards / exclude_st / min_listed_bars / min_bars / holding）
    透传给 b1_oversold，含义与它完全一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .b1_oversold import Strategy as B1Strategy
from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "b1_oversold_norm"
    description = ("B1 稳健超跌（RSI6<20+振幅>8%+连跌2日+近20日无涨停+市场下跌）"
                   "外加归一化超跌门槛（MA5/MA60 与 close/MA60 低于 Q1 阈值）")

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = self.params
        self._r_ma5_ma60_max = float(p.get("r_ma5_ma60_max", 0.9058))
        self._r_ma60_max = float(p.get("r_ma60_max", 0.8995))
        self._use_ma5_ma60 = bool(p.get("use_ma5_ma60", True))
        self._use_ma60 = bool(p.get("use_ma60", True))
        slope_raw = p.get("slope_ma20_5_max", None)
        self._slope_max = float(slope_raw) if slope_raw not in (None, "", "null") else None
        # b1 原逻辑（同一份参数，含 rsi_period 等）
        self._base = B1Strategy(params)

    # ------------------------------------------------------------------ #
    def bind_context(self, ctx: dict | None) -> None:
        # 归一化条件不需要上下文，但 b1 需要（指数下跌、ipoDate）
        self._base.bind_context(ctx)

    # ------------------------------------------------------------------ #
    def _norm_ok(self, df: pd.DataFrame) -> tuple[bool, str]:
        """归一化超跌门槛：只做比值判断，成本极低。"""
        if df.empty:
            return False, "空数据"
        tail = df["close"].astype(float).to_numpy()
        if tail.size < 60:
            return False, f"历史不足60日({tail.size})"
        tail = tail[-60:]
        last_close = float(tail[-1])
        ma5 = float(tail[-5:].mean())
        ma60 = float(tail.mean())
        if ma60 <= 0:
            return False, "MA60 异常"
        parts = []
        if self._use_ma5_ma60:
            r = ma5 / ma60
            if r > self._r_ma5_ma60_max:
                return False, f"MA5/MA60={r:.4f}>{self._r_ma5_ma60_max}"
            parts.append(f"MA5/MA60={r:.4f}")
        if self._use_ma60:
            r2 = last_close / ma60
            if r2 > self._r_ma60_max:
                return False, f"close/MA60={r2:.4f}>{self._r_ma60_max}"
            parts.append(f"close/MA60={r2:.4f}")
        if self._slope_max is not None:
            arr = df["close"].astype(float).to_numpy()
            if arr.size < 25:
                return False, f"斜率历史不足25日({arr.size})"
            ma20_now = float(arr[-20:].mean())
            ma20_prev5 = float(arr[-25:-5].mean())
            if ma20_prev5 <= 0:
                return False, "MA20 前值异常"
            slope = (ma20_now / ma20_prev5 - 1.0) * 100.0
            if slope > self._slope_max:
                return False, f"MA20五日斜率{slope:+.2f}%>{self._slope_max}%"
            parts.append(f"MA20斜率{slope:+.2f}%")
        return True, "、".join(parts)

    # ------------------------------------------------------------------ #
    def signal(self, df: pd.DataFrame) -> dict:
        # ---- 第一道：归一化超跌门槛（便宜、且筛掉约 87%）----
        ok, note = self._norm_ok(df)
        if not ok:
            return self._no_signal(f"归一化超跌未过：{note}")

        # ---- 第二道：b1_oversold 原逻辑 ----
        res = self._base.signal(df)
        if res.get("buy"):
            # score 沿用 b1 的打分（保证与 b1 结果可横向比较），
            # 只在 reason 里标注归一化门槛的实际取值
            res["reason"] = f"{res.get('reason', '')}｜归一化超跌：{note}"
        return res
