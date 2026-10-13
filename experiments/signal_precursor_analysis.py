"""信号前兆分析：解读「起涨点」之前有什么共同特征，以及这些特征能否当信号用。

输入为 signal_precursor_dataset.py 的产出（precursor_*.csv / outcomes_*.csv）。

【符号约定（重要）】
    数据集里的 `close_norm = 该日收盘 / 锚点日收盘 - 1`，**负值表示该日价格低于
    锚点日价格，即锚点之前是上涨的**。为避免误读，本报告统一改用
    `cum_gain = 锚点日收盘 / 该日收盘 - 1`（区间累计涨幅，正值 = 上涨），
    即「从该偏移日买入、持有到锚点日」的涨幅。

分析维度：
    A. 锚点概览        数量 / 覆盖股票 / 上涨序列长度 / 锚点日涨幅 / 序列总涨幅
    B. 前兆价格路径    锚点前 60 日的区间累计涨幅（起涨之前是什么形态）
    C. 前一日特征对比  锚点前一日特征 vs. 全库股票日基准（z 分数）
    D. 盈亏分化        后续盈利 vs 亏损的锚点，其**前兆差异**（t 值排序）
    E. 前兆→收益关系   前兆特征与前向 5 日收益的秩相关 + 五档分位
    F. 候选信号检验    **关键**：把前兆特征变成"事前可执行"的条件，在全库逐日
                       回算该条件的胜率/收益，与全库基准对比 → 判断能否当信号用
    G. 月份分布与集中度

产出：
    docs/<report>                        Markdown 报告（主交付物）
    research_output/analysis_<s>[_tag]_*.csv  各维度明细表
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 参与对比的特征（锚点前一日）。
# 注意：刻意不含 ret1 —— 它由 close 推算，与数据字段 pctChg 近乎等价，
# 同时列出会让报告出现两行完全相同的"特征"，误导阅读。
FEATURES = ["turn", "vol_ratio", "pctChg", "ret5", "dist_ma20", "amount"]
FEATURE_CN = {
    "turn": "换手率(%)", "vol_ratio": "量比(量/5日均量)", "ret1": "当日涨幅(%)",
    "ret5": "近5日涨幅(%)", "dist_ma20": "20日乖离率(%)",
    "pctChg": "涨跌幅(%)", "amount": "成交额(亿元)",
}
MARK_OFFSETS = [-60, -50, -40, -30, -20, -15, -10, -5, -3, -1]


def cum_gain_from_norm(cn: pd.Series) -> pd.Series:
    """close_norm(百分数) → 区间累计涨幅(%)：anchor/close - 1。"""
    return (1.0 / (1.0 + cn / 100.0) - 1.0) * 100.0


def universe_mask(d: pd.DataFrame, min_listed_days: int = 180) -> pd.Series:
    """与策略一致的可交易股票池：非科创(688)、非 ST、上市满 180 天。"""
    m = ~d["code"].str.startswith("sh.68")
    if "isST" in d.columns:
        m &= ~d["isST"].astype(str).str.strip().isin({"1", "1.0", "True"})
    if min_listed_days > 0:
        m &= d["_listed_days"] >= min_listed_days
    return m


def enrich_daily(daily: pd.DataFrame, commission: float, ipo: pd.Series | None) -> pd.DataFrame:
    """全库股票日：派生特征 + 与回测完全一致的前向 5 日净收益。"""
    d = daily.sort_values(["code", "date"], kind="stable").copy()
    g = d.groupby("code", sort=False)
    c, v, o = d["close"].astype(float), d["volume"].astype(float), d["open"].astype(float)
    vol_ma5 = g["volume"].transform(lambda s: s.rolling(5).mean())
    ma20 = g["close"].transform(lambda s: s.rolling(20).mean())
    d["vol_ratio"] = v / vol_ma5.replace(0.0, np.nan)
    d["ret5"] = g["close"].transform(lambda s: (s / s.shift(5) - 1.0) * 100.0)
    d["dist_ma20"] = (c / ma20 - 1.0) * 100.0
    d["ret1"] = g["close"].transform(lambda s: s.pct_change() * 100.0)
    d["amount"] = d["amount"].astype(float) / 1e8
    # 与回测一致：T 日信号 → T+1 开盘入场 → T+1+5 收盘卖出，扣双边手续费
    entry = g["open"].shift(-1)
    exit_ = g["close"].shift(-6)
    d["fwd_ret_5d"] = (exit_ / entry - 1.0 - 2 * commission) * 100.0
    d["_listed_days"] = np.inf
    if ipo is not None:
        dt = pd.to_datetime(d["date"], errors="coerce")
        ipo_dt = pd.to_datetime(d["code"].map(ipo), errors="coerce")
        d["_listed_days"] = (dt - ipo_dt).dt.days
    return d


def baseline_features(d: pd.DataFrame) -> pd.DataFrame:
    """全库（可交易池内）股票日基准特征，用于 z 分数对比。"""
    m = universe_mask(d) & d["fwd_ret_5d"].notna()
    return d.loc[m, FEATURES].dropna()


def signal_precision(d: pd.DataFrame, cond: pd.Series, label: str,
                     near: pd.Series | None = None,
                     ref: pd.Series | None = None) -> dict:
    """在给定条件下评估全库前向 5 日表现，并与对照样本做显著性检验。

    near：布尔序列，表示"该日之前 close 站上 MA20 且 MA20 向上"这类位置约束。
    ref ：对照组的前向收益序列；None 时用全库基准。用于给出胜率差与 t 统计量，
          避免"胜率从 45.5% 到 47.3% 到底有没有意义"这类无法判断的结论。
    """
    base = universe_mask(d) & d["fwd_ret_5d"].notna()
    if near is not None:
        base &= near.fillna(False)
    sub = d.loc[base & cond.fillna(False)]
    if len(sub) == 0:
        return {"方案": label, "股票日数": 0, "5日胜率%": np.nan,
                "5日均收益%": np.nan, "5日中位收益%": np.nan,
                "胜率差(pp)": np.nan, "收益差(pp)": np.nan, "t值": np.nan}
    r = sub["fwd_ret_5d"].astype(float)
    out = {"方案": label, "股票日数": len(sub),
           "5日胜率%": float((r > 0).mean() * 100),
           "5日均收益%": float(r.mean()),
           "5日中位收益%": float(r.median()),
           "胜率差(pp)": np.nan, "收益差(pp)": np.nan, "t值": np.nan}
    if ref is not None and len(ref) > 30:
        ref = ref.astype(float)
        out["胜率差(pp)"] = float((r > 0).mean() * 100 - (ref > 0).mean() * 100)
        out["收益差(pp)"] = float(r.mean() - ref.mean())
        se = float(np.sqrt(r.var(ddof=1) / len(r) + ref.var(ddof=1) / len(ref)))
        out["t值"] = float((r.mean() - ref.mean()) / se) if se > 0 else np.nan
    return out


def _quintile_whatif(last: pd.DataFrame, feat: str) -> pd.DataFrame:
    s = last[[feat, "ret_5d"]].dropna().copy()
    if len(s) < 50:
        return pd.DataFrame()
    try:
        s["q"] = pd.qcut(s[feat], 5, labels=False, duplicates="drop")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    def stat(sub: pd.DataFrame, label: str) -> dict:
        return {"方案": label, "锚点数": len(sub),
                "5日胜率%": float((sub["ret_5d"] > 0).mean() * 100),
                "5日均收益%": float(sub["ret_5d"].mean()),
                "5日中位收益%": float(sub["ret_5d"].median())}

    rows = [stat(s, "全部锚点（基准）")]
    if s["q"].nunique() >= 5:
        rows.append(stat(s[s["q"] < 4], f"剔除最高分位（{feat} 最大 20%）"))
        rows.append(stat(s[s["q"] > 0], f"剔除最低分位（{feat} 最小 20%）"))
        rows.append(stat(s[s["q"].isin([0, 1, 2])], "仅保留最低 3 档"))
    return pd.DataFrame(rows)


def analyse_one(name: str, prec: pd.DataFrame, outc: pd.DataFrame,
                base: pd.DataFrame, dlib: pd.DataFrame, out_dir: Path,
                tag: str) -> dict:
    sfx = f"_{tag}" if tag else ""
    rep: dict = {"strategy": name}
    sig = outc.copy()
    rep["n_signals"] = len(sig)
    rep["n_codes"] = sig["code"].nunique()
    rep["date_min"] = sig["signal_date"].min() if len(sig) else ""
    rep["date_max"] = sig["signal_date"].max() if len(sig) else ""
    rep["run_len_med"] = float(sig["run_len"].median()) if len(sig) else np.nan
    rep["surge_med"] = float(sig["surge_pct"].median()) if len(sig) else np.nan
    rep["anchor_pct_med"] = float(sig["anchor_pct_chg"].median()) if len(sig) else np.nan
    rep["anchor_turn_med"] = float(sig["anchor_turn"].median()) if len(sig) else np.nan

    ev = sig.dropna(subset=["ret_5d"])
    rep["n_eval"] = len(ev)
    if not len(ev):
        return rep

    for h in (3, 5, 10, 20):
        r = ev[f"ret_{h}d"].dropna()
        rep[f"win_rate_{h}d"] = float((r > 0).mean() * 100) if len(r) else np.nan
        rep[f"avg_ret_{h}d"] = float(r.mean()) if len(r) else np.nan
    rep["med_ret_5d"] = float(ev["ret_5d"].median())
    rep["excess_5d"] = float((ev["ret_5d"] - ev["index_ret_5d"]).mean())
    # 起涨点→原信号日的涨幅（本次上涨序列幅度），用于刻画"错过了多少"
    rep["surge_med"] = float(sig["surge_pct"].median())

    ev = ev.copy()
    ev["month"] = ev["signal_date"].str[:7]
    monthly = ev.groupby("month").agg(
        n=("ret_5d", "size"), win_rate=("ret_5d", lambda s: float((s > 0).mean() * 100)),
        avg_ret=("ret_5d", "mean")).reset_index()
    monthly.to_csv(out_dir / f"analysis_{name}{sfx}_monthly.csv", index=False, encoding="utf-8-sig")
    rep["monthly"] = monthly

    top_codes = ev["code"].value_counts().head(5)
    rep["top_codes"] = top_codes
    rep["top_code_share"] = float(top_codes.iloc[0] / len(ev) * 100) if len(top_codes) else 0.0

    # ---- B. 前兆路径（区间累计涨幅，正值=上涨）----
    p = prec.copy()
    p["cum_gain"] = cum_gain_from_norm(p["close_norm"])
    path = p.groupby("offset")["cum_gain"].agg(
        median="median", mean="mean",
        p25=lambda s: float(s.quantile(0.25)),
        p75=lambda s: float(s.quantile(0.75))).reset_index()
    path.to_csv(out_dir / f"analysis_{name}{sfx}_path.csv", index=False, encoding="utf-8-sig")
    rep["path"] = path

    # ---- C. 锚点前一日特征 vs 全库基准 ----
    last = prec[prec["offset"] == -1].copy()
    last = last.merge(ev[["code", "signal_date", "ret_5d", "win_5d"]],
                      on=["code", "signal_date"], how="inner")
    last["amount"] = last["amount"] / 1e8
    comp_rows = []
    for f in FEATURES:
        if f not in last.columns or f not in base.columns:
            continue
        b_mean, b_std = float(base[f].mean()), float(base[f].std())
        s_mean = float(last[f].mean())
        comp_rows.append({
            "特征": FEATURE_CN.get(f, f), "锚点前一日均值": s_mean,
            "全库均值": b_mean, "全库标准差": b_std,
            "z": (s_mean - b_mean) / b_std if b_std else np.nan,
            "锚点前一日中位数": float(last[f].median()),
            "全库中位数": float(base[f].median()),
            "方向": "高于全库" if s_mean > b_mean else "低于全库",
        })
    comp = pd.DataFrame(comp_rows)
    if not comp.empty:
        comp = comp.reindex(comp["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    comp.to_csv(out_dir / f"analysis_{name}{sfx}_features_vs_baseline.csv",
                index=False, encoding="utf-8-sig")
    rep["comp"] = comp

    # ---- D. 盈亏分化 ----
    last["grp"] = np.where(last["ret_5d"] > 0, "盈利", "亏损")
    diff_rows = []
    for f in FEATURES:
        if f not in last.columns:
            continue
        w = last.loc[last["grp"] == "盈利", f].dropna()
        l = last.loc[last["grp"] == "亏损", f].dropna()
        if len(w) < 3 or len(l) < 3:
            continue
        sw, sl = float(w.var(ddof=1)), float(l.var(ddof=1))
        se = float(np.sqrt(sw / len(w) + sl / len(l)))
        diff_rows.append({
            "特征": FEATURE_CN.get(f, f), "盈利均值": float(w.mean()),
            "亏损均值": float(l.mean()), "差值": float(w.mean() - l.mean()),
            "t值": float((w.mean() - l.mean()) / se) if se > 0 else np.nan,
        })
    dif = pd.DataFrame(diff_rows)
    if not dif.empty:
        dif = dif.reindex(dif["t值"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    dif.to_csv(out_dir / f"analysis_{name}{sfx}_win_loss.csv", index=False, encoding="utf-8-sig")
    rep["dif"] = dif

    pw = prec.merge(ev[["code", "signal_date", "win_5d"]], on=["code", "signal_date"], how="inner")
    pw = pw.dropna(subset=["win_5d"]).copy()
    if len(pw):
        pw["cum_gain"] = cum_gain_from_norm(pw["close_norm"])
        pwp = pw.groupby(["offset", "win_5d"])["cum_gain"].mean().unstack()
        pwp.columns = ["亏损路径" if int(c) == 0 else "盈利路径" for c in pwp.columns]
        pwp = pwp.reset_index()
        pwp.to_csv(out_dir / f"analysis_{name}{sfx}_path_by_outcome.csv",
                   index=False, encoding="utf-8-sig")
        rep["pwp"] = pwp

    # ---- E. 前兆特征 → 前向收益 ----
    corr_rows = []
    for f in FEATURES:
        if f not in last.columns:
            continue
        s = last[[f, "ret_5d"]].dropna()
        if len(s) < 10:
            continue
        corr_rows.append({"特征": FEATURE_CN.get(f, f),
                          "spearman": float(s[f].corr(s["ret_5d"], method="spearman")),
                          "pearson": float(s[f].corr(s["ret_5d"]))})
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr = corr.reindex(corr["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    corr.to_csv(out_dir / f"analysis_{name}{sfx}_corr.csv", index=False, encoding="utf-8-sig")
    rep["corr"] = corr

    dec = pd.DataFrame()
    top_f = top_col = None
    if not corr.empty:
        top_f = corr.iloc[0]["特征"]
        top_col = next((k for k, v in FEATURE_CN.items() if v == top_f), top_f)
        s = last[[top_col, "ret_5d"]].dropna().copy()
        if len(s) >= 25:
            try:
                s["q"] = pd.qcut(s[top_col], 5, duplicates="drop")
                dec = s.groupby("q", observed=True).agg(
                    n=("ret_5d", "size"), 均值=("ret_5d", "mean"),
                    胜率=("ret_5d", lambda x: float((x > 0).mean() * 100)),
                    区间下限=(top_col, "min"), 区间上限=(top_col, "max")).reset_index(drop=True)
                dec.to_csv(out_dir / f"analysis_{name}{sfx}_quintile.csv",
                           index=False, encoding="utf-8-sig")
            except Exception:  # noqa: BLE001
                dec = pd.DataFrame()
        rep["quintile_feature"] = top_f
        rep["quintile"] = dec
        rep["whatif"] = _quintile_whatif(last, top_col)

    # ---- F. 候选信号检验（全库逐日回算，判断能否当信号用）----
    bl = dlib.loc[universe_mask(dlib) & dlib["fwd_ret_5d"].notna()]
    ref_ret = bl["fwd_ret_5d"].astype(float)
    base_stat = {"方案": "全库基准（可交易池全部股票日）", "股票日数": len(bl),
                 "5日胜率%": float((ref_ret > 0).mean() * 100),
                 "5日均收益%": float(ref_ret.mean()),
                 "5日中位收益%": float(ref_ret.median()),
                 "胜率差(pp)": np.nan, "收益差(pp)": np.nan, "t值": np.nan}
    rows = [base_stat]

    # 位置约束：贴近起涨点场景 —— 当日收盘站上 MA20 且 MA20 向上
    g = dlib.groupby("code", sort=False)["close"]
    ma20 = g.transform(lambda s: s.rolling(20).mean())
    near = (dlib["close"].astype(float) > ma20) & (ma20 > ma20.groupby(dlib["code"]).shift(1))
    # 对照：位置约束内、但不加其他条件的样本（比全库基准更公平）
    ref_near = dlib.loc[universe_mask(dlib) & dlib["fwd_ret_5d"].notna()
                        & near.fillna(False), "fwd_ret_5d"].astype(float)
    # 位置约束自身即后续条件的对照，故不计算它的 t 值（留给下一行起算）
    rows.append(signal_precision(dlib, pd.Series(True, index=dlib.index),
                                 "位置约束：站上MA20 且 MA20 向上", near, None))

    # 逐个候选条件：**全部特征**都测（不只 top3），阈值用锚点中位数，方向取"高于/低于全库"
    single: list[tuple] = []
    if comp is not None and not comp.empty:
        for _, r in comp.iterrows():
            col = next((k for k, v in FEATURE_CN.items() if v == r["特征"]), None)
            if col is None or col not in dlib.columns:
                continue
            med = float(last[col].median())
            up = str(r["方向"]) == "高于全库"
            cond = (dlib[col] >= med) if up else (dlib[col] <= med)
            lab = f"{r['特征']} {'≥' if up else '≤'} {med:.2f}（锚点中位数）"
            res = signal_precision(dlib, cond, lab, near, ref_near)
            rows.append(res)
            single.append((col, med, up, lab, res.get("t值")))

    # 组合条件：用**实测 t 值最高**的两个/三个单条件组合（而非按 |z| 选），
    # 因为 |z| 只说明"起涨前与全库不同"，不代表该差异能带来正收益。
    valid = [s for s in single if s[4] is not None and not np.isnan(s[4])]
    valid.sort(key=lambda x: x[4], reverse=True)

    def _mk(cs: list[tuple]) -> pd.Series:
        c = pd.Series(True, index=dlib.index)
        for col, med, up, *_ in cs:
            c &= (dlib[col] >= med) if up else (dlib[col] <= med)
        return c

    if len(valid) >= 2:
        rows.append(signal_precision(dlib, _mk(valid[:2]),
                                     f"{valid[0][3]} 且 {valid[1][3]}", near, ref_near))
    if len(valid) >= 3:
        rows.append(signal_precision(dlib, _mk(valid[:3]),
                                     "三个最优单条件同时满足", near, ref_near))

    prec_tbl = pd.DataFrame([r for r in rows if pd.notna(r.get("5日胜率%"))])
    prec_tbl.to_csv(out_dir / f"analysis_{name}{sfx}_signal_test.csv",
                    index=False, encoding="utf-8-sig")
    rep["signal_test"] = prec_tbl
    return rep


def _md_table(df: pd.DataFrame, nd: int = 2) -> str:
    if df is None or df.empty:
        return "_（无数据）_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append("n/a" if (np.isnan(v) or np.isinf(v)) else f"{v:.{nd}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _conclusion(r: dict) -> list[str]:
    out: list[str] = []
    p = r.get("path")
    if p is not None and not p.empty:
        def gv(off: int) -> float:
            s = p.loc[p["offset"] == off, "median"]
            return float(s.iloc[0]) if len(s) else float("nan")
        out.append(f"- **起涨前的形态**：锚点前 60 日累计涨幅中位 {gv(-60):.1f}%、"
                   f"20 日前 {gv(-20):.1f}%、5 日前 {gv(-5):.1f}%、1 日前 {gv(-1):.1f}%"
                   f"（正值＝当时价格低于起涨点价格，即「涨幅」）。")
    comp = r.get("comp")
    if comp is not None and not comp.empty:
        z_hi = comp[comp["z"] > 0].head(2)
        z_lo = comp[comp["z"] < 0].head(2)
        hi = "、".join(f"{x['特征']}(z=+{x['z']:.2f})" for _, x in z_hi.iterrows()) or "无"
        lo = "、".join(f"{x['特征']}(z={x['z']:.2f})" for _, x in z_lo.iterrows()) or "无"
        out.append(f"- **相对全库偏高**：{hi}；**相对全库偏低**：{lo}。")
    st = r.get("signal_test")
    if st is not None and not st.empty:
        body = st.iloc[2:] if len(st) > 2 else st.iloc[0:0]
        body = body.dropna(subset=["胜率差(pp)"])
        if not body.empty:
            good = body[body["胜率差(pp)"] > 0].sort_values("胜率差(pp)", ascending=False)
            bad = body[body["胜率差(pp)"] <= 0]
            if not bad.empty:
                names = "、".join(str(x).split("（")[0].strip() for x in bad["方案"].tolist())
                out.append(f"- ⚠️ **伪信号警示**：以下条件在全库检验中胜率差为负，"
                           f"尽管它们在「起涨前夜的画像」里看起来偏高/偏低：{names}。"
                           f"这类差异属于**样本选择偏差**（起涨股本身更活跃、市值更大），"
                           f"不能当信号用。")
            if not good.empty:
                b = good.iloc[0]
                skew = (f"注意均值 {b['5日均收益%']:.2f}% 远高于中位 {b['5日中位收益%']:.2f}%，"
                        f"收益呈右偏——多数交易小亏、靠少数大赢拉平，"
                        f"实际体验会比均值差。" if b["5日均收益%"] > b["5日中位收益%"] else "")
                out.append(f"- **最佳候选信号**：「{b['方案']}」→ 5 日胜率 {b['5日胜率%']:.1f}%"
                           f"（对照 +{b['胜率差(pp)']:.2f}pp）、均收益 {b['5日均收益%']:.2f}%，"
                           f"命中 {int(b['股票日数'])} 个股票日。{skew}")
                out.append("- **务必打折看待**：上表是对 6 个特征 × 两个方向做筛选后挑出的最优项，"
                           "存在多重比较问题；叠加 t 值本身被重叠样本高估，"
                           "因此这个结果只能作为**值得进一步验证的假设**，"
                           "需要在更长样本上做样本外检验后再决定是否落成策略。")
    dif = r.get("dif")
    if dif is not None and not dif.empty:
        t0 = dif.iloc[0]
        out.append(f"- **最能区分后续盈亏的前兆特征**：{t0['特征']}"
                   f"（t = {t0['t值']:.2f}）。")
    out.append(f"- **事件幅度**：上涨序列长度中位 {r.get('run_len_med', float('nan')):.0f} 个交易日，"
               f"锚点日涨幅中位 {r.get('anchor_pct_med', float('nan')):.2f}%，"
               f"锚点→原信号日涨幅中位 {r.get('surge_med', float('nan')):.2f}%。")
    return out


def build_report(reps: list[dict], out_md: Path, anchor_mode: str) -> None:
    mode_txt = {
        "signal": "锚点 = 策略信号日（上涨序列**末端**）",
        "runstart": "锚点 = 该轮连续上涨序列的**首日**（即「大阳线起点」/ 起涨点）",
    }.get(anchor_mode, anchor_mode)

    md: list[str] = []
    md.append("# 起涨点前兆分析报告\n")
    md.append("**样本库**：`data_library/combined_daily.parquet`（2025-11-12 ~ 2026-10-09，788 只）  ")
    md.append(f"**锚点模式**：{mode_txt}  ")
    md.append("**锚点来源**：策略逐日全量判定的全部信号（不施加回测冷却期）  ")
    md.append("**前兆窗口**：锚点之前的 60 个交易日（offset = −60 … −1）  ")
    md.append("**前向结局**：锚点次日开盘入场、持有 3/5/10/20 个交易日收盘卖出，扣除双边 0.1% 手续费  ")
    md.append("**符号约定**：正文「区间累计涨幅」= 锚点日收盘 / 该日收盘 − 1，**正值表示上涨**。\n")
    if anchor_mode == "runstart":
        md.append("> ### ⚠️ 读这份报告前必须知道的两件事\n")
        md.append("> **1. 锚点后的前向收益是「重言式」，不是可交易结论。**\n"
                  "> 锚点由「连续上涨序列」定义——事件本身要求后面必须涨够 20% 才会被选中，\n"
                  "> 所以第 一 / 五 / 七 节里 100% 的胜率与 +25% 的均收益是**定义导致的必然结果**，\n"
                  "> 只能用来刻画「事件幅度」（这段行情有多大），**不能当作策略收益**。\n")
        md.append("> **2. 真正有信息量的是第 三 节和第 六 节。**\n"
                  "> 第 三 节刻画「起涨前夜」相对普通交易日的特征偏移；\n"
                  "> 第 六 节把这些特征放回**全库**逐日检验精度（能 unbiased 地判断能不能当信号）。\n")
        md.append("> 另：锚点已按 (股票, 起涨日) 去重——一轮长上涨序列会让策略连续多日重复触发，\n"
                  "> 但它们映射到同一个起涨点，不去重会重复计数。\n")
    md.append("> 关键点：本模式的锚点前 60 天窗口**完全落在起涨之前**，"
              "因此窗口内的共同特征才具备「事前埋伏」的意义。\n")

    md.append("## 一、锚点概览\n")
    if anchor_mode == "runstart":
        md.append("（下表的「N日胜率 / 均收益」是**事件幅度**的刻画，非策略收益，理由见开头警告）\n\n")
    ov = [{
        "策略": r["strategy"], "锚点总数": r.get("n_signals", 0),
        "覆盖股票": r.get("n_codes", 0), "可评估": r.get("n_eval", 0),
        "首锚点": r.get("date_min", ""), "末锚点": r.get("date_max", ""),
        "序列长度中位": r.get("run_len_med", np.nan),
        "锚点日涨幅中位%": r.get("anchor_pct_med", np.nan),
        "锚点→信号涨幅中位%": r.get("surge_med", np.nan),
        "3日胜率%": r.get("win_rate_3d", np.nan),
        "5日胜率%": r.get("win_rate_5d", np.nan),
        "10日胜率%": r.get("win_rate_10d", np.nan),
        "20日胜率%": r.get("win_rate_20d", np.nan),
        "5日均收益%": r.get("avg_ret_5d", np.nan),
        "20日均收益%": r.get("avg_ret_20d", np.nan),
    } for r in reps]
    md.append(_md_table(pd.DataFrame(ov)))

    md.append("\n## 二、起涨点前的价格路径（区间累计涨幅）\n")
    md.append("回答「**起涨之前股票长什么样**」——是横盘蓄势、缓慢上行，还是先跌后稳。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        p = r.get("path")
        if p is not None and not p.empty:
            sel = p[p["offset"].isin(MARK_OFFSETS)].rename(columns={
                "offset": "偏移日", "median": "中位涨幅%", "mean": "均涨幅%",
                "p25": "p25涨幅%", "p75": "p75涨幅%"})
            md.append(_md_table(sel))
        if r.get("pwp") is not None and not r["pwp"].empty:
            md.append("\n**后续盈利 vs 亏损的锚点，其起涨前路径（均值）**\n")
            sel = r["pwp"][r["pwp"]["offset"].isin(MARK_OFFSETS)].rename(columns={"offset": "偏移日"})
            md.append(_md_table(sel))

    md.append("\n## 三、锚点前一日特征 vs 全库基准\n")
    md.append("z =（锚点前一日均值 − 全库均值）/ 全库标准差。**|z| 大 = 该特征把"
              "「起涨前夜」与普通交易日区分开了**，即最可能的候选信号维度。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("comp")))

    md.append("\n## 四、后续盈亏的前兆差异\n")
    md.append("按锚点后 5 日收益分盈利/亏损两组，比较锚点前一日各特征均值。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("dif")))

    md.append("\n## 五、前兆特征 → 前向 5 日收益\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("corr")))
        dec = r.get("quintile")
        if dec is not None and not dec.empty:
            md.append(f"\n**按「{r.get('quintile_feature')}」五档分位**\n")
            md.append(_md_table(dec.rename(columns={"n": "锚点数", "均值": "5日均收益%",
                                                    "胜率": "5日胜率%"})))
        if r.get("whatif") is not None and not r["whatif"].empty:
            md.append("\n**锚点内的过滤假设检验**\n")
            md.append(_md_table(r["whatif"]))

    md.append("\n## 六、候选信号检验（关键：能否当信号用）\n")
    md.append("前面只看了「起涨点长什么样」，但**样本只有正例**，无法判断这些特征是否稀有。"
              "本节的检验方法：把这些特征变成全库逐日可执行的条件，"
              "在同一可交易池上回算条件命中后的前向 5 日表现。\n")
    md.append("对照样本 = 「位置约束内但不再加其他条件」的样本（比拿全库做对照更公平），"
              "因此**胜率差 / 收益差 / t 值都是相对该对照计算的**；|t| > 2 才算统计上显著。\n")
    md.append("位置约束 = 当日收盘站上 MA20 且 MA20 向上（贴近起涨点场景，"
              "否则条件会在全库无关位置大量命中）。\n")
    md.append("> ⚠️ **t 值会被高估**：命中样本里存在大量同日相邻、同股票连续的重叠观测"
              "（且前向 5 日窗口互相重叠），有效样本数远小于名义股票日数。"
              "因此 |t| 只能当作**排序依据**，不能按字面解读为显著性水平；"
              "真正的验证需要在更长样本上做时间序列切分的样本外检验。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("signal_test")))

    md.append("\n## 七、月份分布与集中度\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        m = r.get("monthly")
        if m is not None and not m.empty:
            md.append(_md_table(m.rename(columns={"month": "月份", "n": "锚点数",
                                                  "win_rate": "5日胜率%", "avg_ret": "5日均收益%"})))
        tc = {k: int(v) for k, v in dict(r.get("top_codes", {})).items()}
        md.append(f"\n锚点最集中的 5 只股票：{tc}；单只股票最多占 {r.get('top_code_share', 0):.2f}% 的锚点。\n")

    md.append("\n## 八、结论\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        for line in _conclusion(r):
            md.append(line)
    md.append("\n> 说明：结论数值全部由本次样本自动计算。注意「候选信号检验」里若条件的胜率"
              "与全库基准差别不大、或命中样本数极小，则说明该特征**尚不足以单独作为信号**，"
              "需要更多样本或更多条件组合复验。\n")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] -> {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser(description="起涨点前兆分析")
    ap.add_argument("--strategies", required=True, help="策略名，逗号分隔")
    ap.add_argument("--library", default="data_library/combined_daily.parquet")
    ap.add_argument("--data-dir", default="research_output")
    ap.add_argument("--tag", default="", help="数据集文件名后缀（与构建脚本保持一致）")
    ap.add_argument("--anchor", choices=["signal", "runstart"], default="signal")
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--report", default="docs/起涨点前兆分析报告.md")
    args = ap.parse_args()

    names = [x.strip() for x in args.strategies.split(",") if x.strip()]
    ddir = ROOT / args.data_dir
    sfx = f"_{args.tag}" if args.tag else ""

    daily = pd.read_parquet(ROOT / args.library)
    daily["date"] = daily["date"].astype(str)

    ipo = None
    meta_path = ROOT / "data_library" / "_ctx" / "stock_basic.csv"
    if meta_path.exists():
        m = pd.read_csv(meta_path, dtype={"code": str})
        if {"code", "ipoDate"}.issubset(m.columns):
            ipo = dict(zip(m["code"], m["ipoDate"]))

    print("[lib] 构建全库股票日特征与前向收益…")
    dlib = enrich_daily(daily, args.commission, ipo)
    base = baseline_features(dlib)
    print(f"[lib] 可交易池基准 {len(base)} 个股票日")

    reps = []
    for n in names:
        ppath = ddir / f"precursor_{n}{sfx}.csv"
        opath = ddir / f"outcomes_{n}{sfx}.csv"
        if not ppath.exists() or not opath.exists():
            print(f"[warn] 缺少 {n} 的数据集（{ppath.name}），跳过")
            continue
        prec = pd.read_csv(ppath, encoding="utf-8-sig")
        outc = pd.read_csv(opath, encoding="utf-8-sig")
        print(f"[load] {n}: 长表 {len(prec)} 行 / 锚点 {len(outc)} 个")
        reps.append(analyse_one(n, prec, outc, base, dlib, ddir, args.tag))

    if not reps:
        print("[error] 无可用数据")
        return 1
    build_report(reps, ROOT / args.report, args.anchor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
