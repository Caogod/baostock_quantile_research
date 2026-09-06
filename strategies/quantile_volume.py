"""策略 1：分位区间 + 量能 t 检验。

在 quantile_range（价格分位落在 25%~50%）基础上，增加交易量条件：
最近 recent_days（默认 10）日的成交量，相对 window（默认 60）日基准，
通过单尾 Welch t 检验（Student's t-test）判断是否"显著上升"。

参数：
    window        : 60    基准窗口（交易日）
    recent_days   : 10    近期成交量窗口（交易日）
    significance  : 0.01  t 检验显著性水平（单尾）
    quantile_low  : 0.25  分位下限（透传给 quantile_range）
    quantile_high : 0.50  分位上限

注：t 检验在策略内自实现（正则化不完全 Beta → Student t 单尾 p 值），
    不额外依赖 scipy。
"""
from __future__ import annotations

import math

import pandas as pd

from .base import BaseStrategy
from .quantile_range import Strategy as QuantileStrategy


# --------------------------------------------------------------------------- #
# Student's t 分布单尾 p 值（无外部依赖）
# --------------------------------------------------------------------------- #
def _regularized_beta(x: float, a: float, b: float) -> float:
    """正则化不完全 Beta 函数 I_x(a,b)，Numerical Recipes 连分数。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return front * h


def _t_sf(t: float, df: float) -> float:
    """Student t 分布单尾生存函数 P(T > t)，要求 t >= 0。"""
    x = df / (df + t * t)
    return 0.5 * _regularized_beta(x, df / 2.0, 0.5)


def _welch_ttest_pvalue(a: pd.Series, b: pd.Series) -> float:
    """单尾 Welch t 检验 p 值，备择假设 H1: mean(a) > mean(b)。"""
    na, nb = len(a), len(b)
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    if va == 0.0 and vb == 0.0:
        return 1.0 if ma <= mb else 0.0
    se = math.sqrt(va / na + vb / nb)
    if se == 0.0:
        return 1.0 if ma <= mb else 0.0
    t = (ma - mb) / se
    if t <= 0.0:
        return 1.0
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2
    return _t_sf(t, df)


# --------------------------------------------------------------------------- #
class Strategy(BaseStrategy):
    name = "quantile_volume"
    description = "分位区间(25%~50%) + 近10日量能较60日显著上升(t检验)"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        quantile_keys = ("window", "quantile_low", "quantile_high")
        qp = {k: params[k] for k in quantile_keys if k in params}
        self._quantile = QuantileStrategy(qp)
        self.window = int(self.params.get("window", 60))
        self.recent_days = int(self.params.get("recent_days", 10))
        self.significance = float(self.params.get("significance", 0.01))

    def signal(self, df: pd.DataFrame) -> dict:
        # 1. 先满足分位筛选
        base = self._quantile.signal(df)
        if not base.get("buy"):
            return self._no_signal(f"分位不满足：{base.get('reason', '')}")

        if len(df) < self.window:
            return self._no_signal(f"数据不足{self.window}日")

        volume = df["volume"].astype(float).dropna()
        if len(volume) < self.window:
            return self._no_signal("量能数据不足")

        # 2. 近 recent_days 日（含当日 T）vs 历史基准期（剔除近 recent_days 日与当日）
        # P0 修正：基准期与近期窗口不得重叠，否则 Welch t 检验违反"两组独立样本"前提，
        # 协方差被低估、t 值偏大、p 值系统性偏小（"量能显著上升"被夸大）。
        # 基准期 = T-window ... T-recent_days-1（长度 window-recent_days）
        # 近期   = T-recent_days ... T（长度 recent_days，含当日已知）
        recent_vol = volume.iloc[-self.recent_days:]
        base_vol = volume.iloc[-self.window:-self.recent_days]
        if len(base_vol) < 5:  # 基准期样本过少，t 检验不稳定
            return self._no_signal("基准期样本不足")
        p = _welch_ttest_pvalue(recent_vol, base_vol)

        if p < self.significance:
            ratio = float(recent_vol.mean()) / float(base_vol.mean()) if base_vol.mean() else 0.0
            score = round(float(base.get("score", 0.0)) + (1.0 - p) * 0.1, 4)
            return {
                "buy": True,
                "score": score,
                "reason": (
                    f"{base['reason']}；近{self.recent_days}日量能显著上升"
                    f"(p={p:.3f},量比{ratio:.2f})"
                ),
            }
        return self._no_signal(f"量能未显著上升(p={p:.3f})")
