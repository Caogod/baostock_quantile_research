"""策略：归一化超跌（尺度无关的 MA60 族超跌条件）。

来源：`docs/归一化特征信号研究.md` 的全库分档筛选。该研究发现**唯一有信号的
归一化维度是"超跌"**，且是**阈值型**——只有最低档（Q1）有效，其余档不再单调，
所以正确用法是把阈值当门槛。

本策略只用**尺度无关的比值**，不含任何绝对量（成交额、股价水平），
避免上一轮报告里"起涨股恰好是大市值活跃股"那种样本选择偏差。

条件（可选组合，均为"低于阈值"）：
    1. MA5 / MA60 ≤ r_ma5_ma60_max          短期均线显著低于长期均线（默认启用）
    2. close / MA60 ≤ r_ma60_max            收盘价显著低于 60 日均线（默认启用）
    3. MA20 五日斜率 ≤ slope_ma20_5_max     20 日均线快速下滑（默认**关闭**）
    4. 非 ST / 非退市、非科创板、上市满 min_listed_days 天

阈值参考（全库 5 档分位，147,984 个股票日，2025-11-12 ~ 2026-10-09）：
    MA5/MA60   Q1 上界 0.9058（50.23% 胜率 / +0.375%）→ Q2 上界 0.9575（43.10% / -0.654%）
    close/MA60 Q1 上界 0.8995（50.04% / +0.416%）      → Q2 上界 0.9543（43.64% / -0.659%）
    MA20 斜率  Q1 上界 -2.1382（50.20% / +0.425%）     → Q2 上界 -0.7307（44.44% / -0.487%）
注意 Q2 档本身的均收益已是**负**的，所以"放宽到 Q2"会把 edge 稀释掉——
这不是猜测，是分档统计里能直接看到的（Q2 起点即由正转负）。
两个 MA60 族条件同时满足时：
    Q1 严 → 18,524 股票日（12.5%），50.60% / +0.502%
    Q2 宽 → 38,850 股票日（26.3%），47.01% / -0.091%（≈ 全库基准 45.48% / -0.103%）
    再加 MA20 斜率 → 14,463 股票日（9.8%），51.23% / +0.688%（最强）

【性能】刻意不用 pandas.rolling：只需末值，直接用 numpy 尾切片求均值，
避免每个交易日对整条序列做 O(n) 的滚动计算（原来那样会退化成 O(n²)）。

参数：
    r_ma5_ma60_max  : 0.9575  MA5/MA60 上限（Q2 档默认；Q1 严档用 0.9058）
    r_ma60_max      : 0.9543  close/MA60 上限（Q2 档默认；Q1 严档用 0.8995）
    use_ma5_ma60    : True    是否启用条件 1
    use_ma60        : True    是否启用条件 2
    slope_ma20_5_max: None    MA20 五日斜率上限(%)；设数值即启用（推荐 -2.1382）
    exclude_st      : True    剔除 ST / 退市
    exclude_boards  : "688"   剔除的板块前缀（逗号分隔，空表示不剔除）
    min_listed_days : 180     上市天数下限（自然日，0 不校验）
    min_bars        : 60      需要 MA60，故至少 60 根
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "norm_oversold"
    description = ("归一化超跌：MA5/MA60 与 close/MA60 低于阈值（尺度无关），"
                   "可选叠加 MA20 五日斜率；非ST/非科创/非次新")

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
        r_ma5_ma60_max = float(p.get("r_ma5_ma60_max", 0.9575))
        r_ma60_max = float(p.get("r_ma60_max", 0.9543))
        use_ma5_ma60 = bool(p.get("use_ma5_ma60", True))
        use_ma60 = bool(p.get("use_ma60", True))
        slope_raw = p.get("slope_ma20_5_max", None)
        slope_max = float(slope_raw) if slope_raw not in (None, "", "null") else None
        exclude_st = bool(p.get("exclude_st", True))
        exclude_boards = [b.strip() for b in str(p.get("exclude_boards", "688")).split(",") if b.strip()]
        min_listed_days = int(p.get("min_listed_days", 180))
        # 历史长度：MA60 与 close/MA60 需要 60 根；MA20 五日斜率只需 25 根，
        # 所以 60 就是硬要求（不要因为开了斜率就再加码，否则会多排除 5 根、
        # 与分析口径对不上）。
        min_bars = int(p.get("min_bars", 60))

        n = len(df)
        if n < min_bars:
            return self._no_signal(f"历史不足{min_bars}日({n})")

        code = str(df["code"].iloc[-1]) if "code" in df.columns else ""
        num = code.split(".")[-1]
        i = n - 1

        # ============================================================== #
        # 条件顺序按过滤代价由低到高（AND 关系，顺序不影响判定结果）
        # ============================================================== #

        # ---- 1) 非科创板（常量）----
        if exclude_boards and any(num.startswith(b) for b in exclude_boards):
            return self._no_signal(f"板块排除({num})")

        # ---- 2) 非 ST / 非退市（常量，优先逐日 isST）----
        if exclude_st:
            if "isST" in df.columns:
                if str(df["isST"].iloc[i]).strip() in {"1", "1.0", "True", "true"}:
                    return self._no_signal("ST/风险警示股")
            else:
                name = self._meta_value(code, "name")
                if ("ST" in name) or ("退" in name):
                    return self._no_signal(f"ST/退市({name})")

        # ---- 3) 非次新股（meta 查表）----
        if min_listed_days > 0:
            ipo = self._meta_value(code, "ipoDate")
            if ipo:
                try:
                    d0 = datetime.strptime(ipo, "%Y-%m-%d")
                    d1 = datetime.strptime(str(df["date"].iloc[i]), "%Y-%m-%d")
                    if (d1 - d0).days < min_listed_days:
                        return self._no_signal(f"上市未满{min_listed_days}天({ipo})")
                except Exception:  # noqa: BLE001
                    pass

        # ---- 4) 均线比值（尾切片求均值，避免整序列 rolling）----
        need = max(60, 25 if slope_max is not None else 0)
        if n < need:
            return self._no_signal(f"历史不足{need}日({n})")
        arr = df["close"].astype(float).to_numpy()
        tail = arr[-60:]
        last_close = float(tail[-1])
        ma5 = float(tail[-5:].mean())
        ma60 = float(tail.mean())
        if ma60 <= 0:
            return self._no_signal("MA60 异常")

        r_ma5_ma60 = ma5 / ma60
        r_ma60 = last_close / ma60

        if use_ma5_ma60 and r_ma5_ma60 > r_ma5_ma60_max:
            return self._no_signal(f"MA5/MA60={r_ma5_ma60:.4f}>{r_ma5_ma60_max}（未超跌）")
        if use_ma60 and r_ma60 > r_ma60_max:
            return self._no_signal(f"close/MA60={r_ma60:.4f}>{r_ma60_max}（未超跌）")

        # ---- 5) MA20 五日斜率（可选，需要往前 5 根）----
        slope = None
        if slope_max is not None:
            ma20_now = float(arr[-20:].mean())
            ma20_prev5 = float(arr[-25:-5].mean())
            if ma20_prev5 <= 0:
                return self._no_signal("MA20 前值异常")
            slope = (ma20_now / ma20_prev5 - 1.0) * 100.0
            if slope > slope_max:
                return self._no_signal(f"MA20五日斜率{slope:+.2f}%>{slope_max}%（未快速下滑）")

        # ---- 打分：越深地跌破 MA60 → 分数越高（仅作排序启发式）----
        score = round((1.0 - r_ma5_ma60) * 100.0 + (1.0 - r_ma60) * 100.0, 4)
        parts = [
            f"MA5/MA60={r_ma5_ma60:.4f}≤{r_ma5_ma60_max}",
            f"close/MA60={r_ma60:.4f}≤{r_ma60_max}",
        ]
        if slope is not None:
            parts.append(f"MA20五日斜率{slope:+.2f}%≤{slope_max}%")
        return {
            "buy": True,
            "score": score,
            "reason": "归一化超跌：" + "；".join(parts) + f"；收盘{last_close:.2f}",
        }
