"""策略：上升趋势中的「缩量阴线回调」（起涨前夜型）。

来源：`docs/起涨点前兆分析报告.md` 的「候选信号检验」。
把 consecutive_up5 起涨点的**前兆画像**落成可执行条件，那个报告里
表现最好的组合是「当日跌幅 ≤ −2.61% 且 量比 ≤ 0.90」，
但它是在**「站上 MA20 且 MA20 向上」**这个位置约束之内被检验的，
所以本策略把这个位置约束一并带上——否则条件不等价，回测结果无法与
报告里的 49.5% 胜率 / +1.00% 均收益对照。

条件（全部满足）：
    1. 收盘价 > MA20                    位置约束，趋势向上
    2. MA20 向上（今日 MA20 > 昨日 MA20）
    3. 当日涨跌幅 ≤ pct_max（默认 −2.61%）   起涨前夜的"阴线"
    4. 量比 = 当日量 / 前 5 日均量 ≤ vol_ratio_max（默认 0.90）  缩量
    5. 非 ST / 非退市（优先日线逐日 isST）
    6. 非科创板（688）
    7. 上市满 min_listed_days 个自然日（默认 180）

【为什么用"阴线 + 缩量"】报告里全库检验显示：在站上 MA20 的趋势股中，
「当日跌 ≤ −2.6%」单独带来 +1.85pp 胜率，「量比 ≤ 0.90」单独 +1.15pp，
两者叠加 +4.09pp（49.5% vs 对照 45.4%）。可解读为：上涨途中缩量回踩
均线附近，卖压衰竭、次日常有反抽。

【必须打折看待的地方（沿用报告的保留意见）】
- 报告里 t = 4.16 被**重叠样本**高估（同股票连续多日、5 日前向窗口重叠），
  有效样本远少于名义股票日数；
- 该组合是对 6 个特征 × 2 方向做筛选后挑出的最优项，存在**多重比较**问题；
- 报告里该组合均值 +1.00% 而中位 −0.09%，收益**右偏**：多数交易小亏，
  靠少数大赢拉平，实盘体验会比均值差。
因此本策略的正确定位是「把报告假设拿来做样本外/独立回测检验」，
而不是已验证的盈利策略。

【条件顺序：按过滤代价由低到高手工排定】
    O(1) 常量（板块 / ST / 上市天数）→ O(1) 当日涨跌幅
    → O(vol_ma_window) 定长切片（量比）→ rolling(20) 均线（最贵，放最后）
    AND 关系，顺序不影响结果，只影响性能。

参数：
    ma_long         : 20     趋势基准均线
    pct_max         : -2.61  当日涨跌幅上限(%)，≤ 该值才入选
    vol_ratio_max   : 0.90   量比上限（量 / 前 N 日均量）
    vol_ma_window   : 5      量比所用的均量窗口
    vol_include_today : True 均量是否含当日（True 与分析脚本口径一致；False 为A股常规量比）
    exclude_st      : True   是否剔除 ST / 退市
    exclude_boards  : "688"  需剔除的板块前缀（逗号分隔，空表示不剔除）
    min_listed_days : 180    上市天数下限（自然日，0 不校验）
    min_bars        : 25     单只股票所需最小历史长度（MA20+5，尽量贴近分析样本口径）
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy


class Strategy(BaseStrategy):
    name = "ma20_pullback_shrink"
    description = ("上升趋势缩量回调：收盘>MA20 且 MA20 向上，当日跌幅≤-2.61%，"
                   "量比≤0.90（缩量）；非ST/非科创/非次新")

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
        ma_long = int(p.get("ma_long", 20))
        pct_max = float(p.get("pct_max", -2.61))
        vol_ratio_max = float(p.get("vol_ratio_max", 0.90))
        vol_ma_window = int(p.get("vol_ma_window", 5))
        vol_include_today = bool(p.get("vol_include_today", True))
        exclude_st = bool(p.get("exclude_st", True))
        exclude_boards = [b.strip() for b in str(p.get("exclude_boards", "688")).split(",") if b.strip()]
        min_listed_days = int(p.get("min_listed_days", 180))
        min_bars = int(p.get("min_bars", 65))

        need = max(min_bars, ma_long + 1, vol_ma_window + 1)
        n = len(df)
        if n < need:
            return self._no_signal(f"历史不足{need}日({n})")

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

        # ---- 4) 当日涨跌幅 ≤ pct_max（O(1)，起涨前夜的"阴线"）----
        if "pctChg" in df.columns and not pd.isna(df["pctChg"].iloc[i]):
            pct_chg = float(df["pctChg"].iloc[i])
        else:
            close_s = df["close"].astype(float)
            prev = float(close_s.iloc[i - 1])
            pct_chg = (float(close_s.iloc[i]) / prev - 1.0) * 100.0 if prev > 0 else float("nan")
        if pd.isna(pct_chg):
            return self._no_signal("涨跌幅数据缺失")
        if pct_chg > pct_max:
            return self._no_signal(f"当日涨跌幅{pct_chg:.2f}%>-{abs(pct_max)}%（未构成回调阴线）")

        # ---- 5) 量比 ≤ vol_ratio_max（O(vol_ma_window) 定长切片，缩量）----
        # 口径必须与产出该条件的分析脚本一致，否则不是同一个条件：
        #   分析里用的是 volume.rolling(N).mean() —— **含当日**；
        #   而 A 股常规"量比"是 当日量 / 过去 N 日均量 —— **不含当日**。
        # 两者结果有差异（缩量日会把含当日的均量拉低），故做成参数，默认与分析一致。
        volume = df["volume"].astype(float).reset_index(drop=True)
        last_vol = float(volume.iloc[i])
        if vol_include_today:
            vol_base = float(volume.iloc[i - vol_ma_window + 1:i + 1].mean())
        else:
            vol_base = float(volume.iloc[i - vol_ma_window:i].mean())
        if vol_base <= 0:
            return self._no_signal("均量数据异常")
        vol_ratio = last_vol / vol_base
        if vol_ratio > vol_ratio_max:
            return self._no_signal(f"量比{vol_ratio:.2f}>{vol_ratio_max}（未缩量）")

        # ---- 6) 收盘 > MA20 且 MA20 向上（rolling 最贵，放最后）----
        close = df["close"].astype(float).reset_index(drop=True)
        ma = close.rolling(ma_long).mean()
        ma_now = float(ma.iloc[i])
        ma_prev = float(ma.iloc[i - 1])
        if pd.isna(ma_now) or pd.isna(ma_prev) or ma_now <= 0:
            return self._no_signal(f"MA{ma_long}数据不足")
        last_close = float(close.iloc[i])
        if last_close <= ma_now:
            return self._no_signal(f"收盘{last_close:.2f}未站上MA{ma_long}({ma_now:.2f})")
        if ma_now <= ma_prev:
            return self._no_signal(f"MA{ma_long}未向上({ma_prev:.2f}→{ma_now:.2f})")

        # ---- 打分（仅作排序用的启发式，未经独立验证）----
        # 跌幅越深 + 缩量越明显 → 分数越高
        score = round(-pct_chg + (vol_ratio_max - vol_ratio) * 10.0, 4)
        return {
            "buy": True,
            "score": score,
            "reason": (
                f"收盘{last_close:.2f}>MA{ma_long}({ma_now:.2f})且MA{ma_long}向上"
                f"({ma_prev:.2f}→{ma_now:.2f})；当日涨跌幅{pct_chg:.2f}%≤{pct_max}%；"
                f"缩量（量{last_vol:.0f} / {vol_ma_window}日均量{vol_base:.0f}=量比{vol_ratio:.2f}"
                f"≤{vol_ratio_max}）"
            ),
        }
