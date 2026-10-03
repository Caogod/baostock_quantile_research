"""策略：均线单次金叉突破 + 低波动 + 动量爆发。

条件按"廉价且筛选力强"的顺序执行（顺序不影响结果，只影响性能——
一旦前面条件不满足即提前返回，后面的滚动均线/标准差就不必计算）：

1. 动量爆发：最近 surge_days（默认 2）个交易日累计涨幅 > surge_pct。
2. 量能放大：最新换手率 > 换手率的 turn_ma 日（默认 10）均线 × turn_ratio。
3. 均线单次金叉：过去 lookback 个交易日内，MA5/MA10/MA20 从下方上穿 MA60 的金叉
   次数恰好为 1，且最新收盘价站上 MA60（突破 MA60）。
4. 低波动（二选一，或关系），判定窗口为"突破前窗口"：
   近 lookback 个交易日、截止突破前一天（排除最近 surge_days 个突破日），
   且**先做 MA5 连续下降修剪**（见下）：
   (a) 布林带：窗口内所有收盘价均落在 MA20 ± boll_k 倍标准差之内；
   (b) 极差/最小值：(max-min)/min < vol_ratio_max。
   注：突破日自身常冲出上轨，故波动约束仅作用于突破前的横盘窗口。

MA5 连续下降修剪（trim_decline）：
    从"突破前窗口"的**头部**开始看 MA5 是否连续下降；若连续下降，则把这段下降
    区间从窗口中剔除，**从停止下降的时刻**（首个不再低于前一日的 MA5 所处日期）
    开始计算波动率。目的是排除窗口头部残留的下跌段，只度量其后真正的横盘整理
    阶段，避免下跌段把波动率撑大导致"低波动"条件被误判。修剪后窗口长度若不足
    min_vol_days 日，则直接判定不满足（横盘期太短）。

参数：
    lookback      : 60    均线金叉统计窗口（交易日）
    surge_days    : 2     动量爆发累计天数（同时定义"突破前一天"= T-surge_days）
    surge_pct     : 0.15  累计涨幅阈值
    turn_ma       : 10    换手率均线窗口
    turn_ratio    : 1.5   换手率相对其均线的放大倍数
    boll_ma       : 20    布林带中轨均线窗口
    boll_k        : 3.0   布林带标准差倍数
    vol_ratio_max : 0.15  极差/最小值 阈值（与布林带构成或关系）
    trim_decline  : True  是否启用 MA5 连续下降修剪波动窗口
    min_vol_days  : 10    修剪后波动窗口的最小长度（不足则不出信号）
    cross_ma_short: 5     金叉判定的短期均线
    cross_ma_mid  : 10    金叉判定的中期均线
    cross_ma_long : 20    金叉判定的长期均线
    cross_ma_base : 60    金叉穿越的基准均线

阈值选取辅助（静态方法）：
    Strategy.analyze_volatility_distribution(daily, window=60, metric="close")
        对样本库做「极差/最小值」横截面分位数统计，返回 (stats, quantiles)，
        用于给 vol_ratio_max 选值。metric 支持 "close" / "highlow" 两种口径，
        与 base.volatility_ratio 一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy, volatility_ratio


def _count_golden_cross(series: pd.Series, base: pd.Series) -> int:
    """统计 series 从下方上穿 base 的金叉次数。

    金叉定义：前一交易日 series <= base，当日前一交易日 series > base。
    """
    below = (series.shift(1) <= base.shift(1)) & (series > base)
    return int(below.sum())


class Strategy(BaseStrategy):
    name = "ma_cross_breakout"
    description = ("60日单次金叉破MA60+突破前低波动(3σ布林带内 或 极差/最小值<阈值)"
                   "+2日涨>阈值+换手率>MA10×倍数；波动窗口前用MA5连续下降修剪")

    def signal(self, df: pd.DataFrame) -> dict:
        lookback = int(self.params.get("lookback", 60))
        surge_days = int(self.params.get("surge_days", 2))
        surge_pct = float(self.params.get("surge_pct", 0.15))
        turn_ma = int(self.params.get("turn_ma", 10))
        turn_ratio = float(self.params.get("turn_ratio", 1.5))
        boll_ma = int(self.params.get("boll_ma", 20))
        boll_k = float(self.params.get("boll_k", 3.0))
        vol_ratio_max = float(self.params.get("vol_ratio_max", 0.15))
        trim_decline = bool(self.params.get("trim_decline", True))
        min_vol_days = int(self.params.get("min_vol_days", 10))
        cross_ma_short = int(self.params.get("cross_ma_short", 5))
        cross_ma_mid = int(self.params.get("cross_ma_mid", 10))
        cross_ma_long = int(self.params.get("cross_ma_long", 20))
        cross_ma_base = int(self.params.get("cross_ma_base", 60))

        # 需要足够的均线预热 + 观察窗口
        need = max(cross_ma_base, boll_ma) + lookback
        if len(df) < need:
            return self._no_signal(f"数据不足{need}日")

        close = df["close"].astype(float)
        last_close = float(close.iloc[-1])

        # ---- 条件 1：动量爆发（最廉价、筛选力最强，最先算）----
        if len(close) < surge_days + 1:
            return self._no_signal(f"数据不足{surge_days + 1}日")
        base_close = float(close.iloc[-1 - surge_days])
        if base_close == 0:
            return self._no_signal("累计涨幅基准为0")
        surge = last_close / base_close - 1
        if surge <= surge_pct:
            return self._no_signal(f"近{surge_days}日涨幅{surge:.1%}未达{surge_pct:.0%}")

        # ---- 条件 2：量能放大（换手率 > 换手率 MA(turn_ma) × turn_ratio）----
        if "turn" not in df.columns:
            return self._no_signal("缺少换手率字段")
        turn = df["turn"].astype(float)
        last_turn = float(turn.iloc[-1])
        last_turn_ma = float(turn.rolling(turn_ma).mean().iloc[-1])
        if pd.isna(last_turn_ma) or last_turn_ma == 0:
            return self._no_signal("换手率均线数据不足")
        if last_turn <= last_turn_ma * turn_ratio:
            return self._no_signal(
                f"换手率{last_turn:.2f}%未达{turn_ma}日均线{last_turn_ma:.2f}%×{turn_ratio}"
            )

        # ---- 条件 3：过去 lookback 日内金叉恰好一次，且当前站上 MA60 ----
        ma_base = close.rolling(cross_ma_base).mean()
        last_base = float(ma_base.iloc[-1])
        if pd.isna(last_base):
            return self._no_signal("MA60数据不足")

        win_base = ma_base.iloc[-lookback:]
        cross_total = 0
        for period in (cross_ma_short, cross_ma_mid, cross_ma_long):
            win_ma = close.rolling(period).mean().iloc[-lookback:]
            cross_total += _count_golden_cross(win_ma, win_base)
        if cross_total != 1:
            return self._no_signal(f"近{lookback}日金叉次数={cross_total}，不等于1")

        if last_close <= last_base:
            return self._no_signal(f"收盘{last_close:.2f}未站上MA60={last_base:.2f}")

        # ---- 条件 4：突破前窗口低波动（先按 MA5 连续下降修剪窗口头部）----
        mid = close.rolling(boll_ma).mean()
        std = close.rolling(boll_ma).std()
        upper = mid + boll_k * std
        lower = mid - boll_k * std

        # 突破前窗口右端（不含最近 surge_days 个突破日）
        end = len(close) - surge_days
        win_start = end - lookback

        # MA5 连续下降修剪：跳过窗口头部的连续下降段，从"停止下降"处开始算波动率
        trim_note = ""
        if trim_decline:
            ma5 = close.rolling(5).mean().to_numpy()
            w5 = ma5[win_start:end]
            j = 1
            while (j < len(w5)
                   and not np.isnan(w5[j]) and not np.isnan(w5[j - 1])
                   and w5[j] < w5[j - 1]):
                j += 1
            if j > 1:
                win_start += j
                trim_note = f"，MA5连降{j}日后起算"

        seg = close.iloc[win_start:end]
        if len(seg) < min_vol_days:
            return self._no_signal(f"突破前低波动窗口不足{min_vol_days}日{trim_note}")

        # (a) 布林带：窗口内所有收盘均在带内
        win_upper = upper.iloc[win_start:end]
        win_lower = lower.iloc[win_start:end]
        valid = win_upper.notna() & win_lower.notna()
        band_ok = bool(valid.any()) and bool(
            ((seg[valid] < win_upper[valid]) & (seg[valid] > win_lower[valid])).all()
        )

        # (b) 极差/最小值：(max-min)/min < vol_ratio_max
        vol_ratio = volatility_ratio(seg, metric="close")
        ratio_ok = vol_ratio < vol_ratio_max

        if not (band_ok or ratio_ok):
            return self._no_signal(
                f"突破前{len(seg)}日非低波动{trim_note}"
                f"（布林带外 且 极差/最小值{vol_ratio:.2%}≥{vol_ratio_max:.0%}）"
            )
        vol_note = "布林带内" if band_ok else f"极差/最小值{vol_ratio:.2%}<{vol_ratio_max:.0%}"

        score = round(surge * 100, 4)
        return {
            "buy": True,
            "score": score,
            "reason": (
                f"近{surge_days}日涨{surge:.1%}；"
                f"换手{last_turn:.2f}%>{turn_ma}日均线{last_turn_ma:.2f}%×{turn_ratio}；"
                f"近{lookback}日金叉1次且站上MA60({last_close:.2f}>{last_base:.2f})；"
                f"低波动({vol_note}{trim_note})"
            ),
        }

    def prefilter_mask(self, close: np.ndarray) -> np.ndarray:
        """向量化预筛（必要条件超集，不改变信号语义）。

        候选日必须同时满足：
        1. 数据量足够：i >= need - 1（与 signal 的 len(df) < need 早退一致）；
        2. 动量必要条件：close[i] / close[i - surge_days] - 1 > surge_pct。

        二者都是 signal() 通过时的必要条件，故掩码不会漏信号；
        却能在回溯中跳过绝大多数交易日，避免逐日全量计算。
        """
        lookback = int(self.params.get("lookback", 60))
        surge_days = int(self.params.get("surge_days", 2))
        surge_pct = float(self.params.get("surge_pct", 0.15))
        boll_ma = int(self.params.get("boll_ma", 20))
        cross_ma_base = int(self.params.get("cross_ma_base", 60))

        close = np.asarray(close, dtype=float)
        n = close.shape[0]
        mask = np.zeros(n, dtype=bool)
        need = max(cross_ma_base, boll_ma) + lookback
        if n <= max(surge_days, need - 1):
            return mask

        # 与 signal 中 `surge <= surge_pct 即拒绝` 严格对齐：这里用 > (阈值 - eps)
        # 放宽下界，避免浮点边界把真正的信号日筛掉（漏筛不可容忍，多筛只是稍慢）。
        with np.errstate(divide="ignore", invalid="ignore"):
            surge = close[surge_days:] / close[:-surge_days] - 1.0
        mask[surge_days:] = surge > (surge_pct - 1e-9)
        mask[: need - 1] = False
        return mask

    # ------------------------------------------------------------------ #
    # 阈值选取辅助：极差/最小值 横截面分布分析（用于给 vol_ratio_max 定值）
    # ------------------------------------------------------------------ #
    @staticmethod
    def analyze_volatility_distribution(daily: pd.DataFrame, window: int = 60,
                                        metric: str = "close") -> tuple[pd.DataFrame, pd.Series]:
        """对样本库做「极差/最小值」横截面分布统计。

        参数：
            daily  : 样本库日 K DataFrame（需含 code/date/close，metric=highlow 时
                     还需 high/low）。
            window : 观察窗口（交易日，每只股票取最近 window 日，默认 60）。
            metric : "close" 或 "highlow"（与 volatility_ratio 一致）。

        返回：
            (stats, quantiles)
            stats      : DataFrame，每只股票一行，列 code / ratio（按 ratio 升序）。
            quantiles  : pd.Series，分位数（index 为 0/0.1/0.25/0.5/0.75/0.9/1.0）。
        """
        rows: list[dict] = []
        for code, g in daily.groupby("code"):
            g = g.sort_values("date").tail(window)
            if len(g) < window:
                continue
            if metric == "highlow":
                ratio = volatility_ratio(g["close"], g["high"], g["low"], metric="highlow")
            else:
                ratio = volatility_ratio(g["close"], metric="close")
            rows.append({"code": code, "ratio": ratio})
        stats = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
        quantiles = pd.Series(
            {q: float(stats["ratio"].quantile(q)) for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]}
        ) if not stats.empty else pd.Series(dtype=float)
        return stats, quantiles
