"""信号前兆分析：解读新策略「信号前 60 天」在发生什么，以及什么条件下会亏。

输入为 signal_precursor_dataset.py 的产出（precursor_*.csv / outcomes_*.csv）。

【符号约定（重要）】
    数据集里的 `close_norm = 该日收盘 / 信号日收盘 - 1`，**负值表示该日价格低于
    信号日价格，即信号之前是上涨的**。为避免误读，本报告统一改用
    `cum_gain = 信号日收盘 / 该日收盘 - 1`（区间累计涨幅，正值 = 上涨），
    即「从该偏移日买入、持有到信号日」的涨幅。

分析维度：
    A. 信号概览        数量 / 覆盖股票 / 前向 3/5/10 日胜率与收益
    B. 前兆价格路径    信号前 60 日的区间累计涨幅（策略在追什么形态）
    C. 前一日特征对比  信号前一日特征 vs. 全库股票日基准（z 分数）
    D. 盈亏分化        盈利 vs 亏损信号的**前兆差异**（t 值排序）
    E. 前兆→收益关系   前兆特征与前向 5 日收益的 Spearman 秩相关 + 分位表
    F. 过滤假设检验    基于上述发现的「砍掉某一分位」what-if，量化胜率/收益改善
    G. 月份分布与集中度

产出：
    docs/信号前兆分析报告.md            Markdown 报告（主交付物）
    research_output/analysis_<s>_*.csv  各维度明细表
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

FEATURES = ["turn", "vol_ratio", "ret1", "ret5", "dist_ma20", "pctChg", "amount"]
FEATURE_CN = {
    "turn": "换手率(%)", "vol_ratio": "量比(量/5日均量)", "ret1": "当日涨幅(%)",
    "ret5": "近5日涨幅(%)", "dist_ma20": "20日乖离率(%)",
    "pctChg": "涨跌幅(%)", "amount": "成交额(亿元)",
}
MARK_OFFSETS = [-60, -50, -40, -30, -20, -15, -10, -5, -3, -1]


def cum_gain_from_norm(cn: pd.Series) -> pd.Series:
    """close_norm(百分数) → 区间累计涨幅(%)：anchor/close - 1。"""
    return (1.0 / (1.0 + cn / 100.0) - 1.0) * 100.0


def baseline_features(daily: pd.DataFrame) -> pd.DataFrame:
    """全库股票日基准特征（与信号前一日同口径），用于 z 分数对比。"""
    d = daily.sort_values(["code", "date"], kind="stable").copy()
    g = d.groupby("code", sort=False)
    d["vol_ma5"] = g["volume"].transform(lambda s: s.rolling(5).mean())
    d["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    d["vol_ratio"] = d["volume"] / d["vol_ma5"].replace(0.0, np.nan)
    d["ret5"] = g["close"].transform(lambda s: (s / s.shift(5) - 1.0) * 100.0)
    d["dist_ma20"] = (d["close"] / d["ma20"] - 1.0) * 100.0
    d["ret1"] = g["close"].transform(lambda s: s.pct_change() * 100.0)
    d["amount"] = d["amount"] / 1e8
    return d[FEATURES].dropna()


def _quintile_whatif(last: pd.DataFrame, feat: str) -> pd.DataFrame:
    """对特征 feat 做五档分位，并给出「砍掉某一端」的 what-if 表现。"""
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

    rows = [stat(s, "全部信号（基准）")]
    if s["q"].nunique() >= 5:
        rows.append(stat(s[s["q"] < 4], f"剔除最高分位（{feat} 最大 20%）"))
        rows.append(stat(s[s["q"] > 0], f"剔除最低分位（{feat} 最小 20%）"))
        rows.append(stat(s[s["q"].isin([0, 1, 2])], "仅保留最低 3 档"))
        rows.append(stat(s[s["q"].isin([1, 2, 3])], "剔除两端（中间 3 档）"))
    return pd.DataFrame(rows)


def analyse_one(name: str, prec: pd.DataFrame, outc: pd.DataFrame,
                base: pd.DataFrame, out_dir: Path) -> dict:
    rep: dict = {"strategy": name}
    sig = outc.copy()
    rep["n_signals"] = len(sig)
    rep["n_codes"] = sig["code"].nunique()
    rep["date_min"] = sig["signal_date"].min() if len(sig) else ""
    rep["date_max"] = sig["signal_date"].max() if len(sig) else ""

    ev = sig.dropna(subset=["ret_5d"])
    rep["n_eval"] = len(ev)
    if not len(ev):
        return rep

    rep["win_rate_3d"] = float((ev["ret_3d"].dropna() > 0).mean() * 100)
    rep["win_rate_5d"] = float((ev["ret_5d"] > 0).mean() * 100)
    rep["win_rate_10d"] = float((ev["ret_10d"].dropna() > 0).mean() * 100)
    rep["avg_ret_5d"] = float(ev["ret_5d"].mean())
    rep["med_ret_5d"] = float(ev["ret_5d"].median())
    rep["excess_5d"] = float((ev["ret_5d"] - ev["index_ret_5d"]).mean())
    rep["sig_pct_med"] = float(sig["signal_pct_chg"].median())
    rep["sig_turn_med"] = float(sig["signal_turn"].median())

    ev = ev.copy()
    ev["month"] = ev["signal_date"].str[:7]
    monthly = ev.groupby("month").agg(
        n=("ret_5d", "size"), win_rate=("ret_5d", lambda s: float((s > 0).mean() * 100)),
        avg_ret=("ret_5d", "mean")).reset_index()
    monthly.to_csv(out_dir / f"analysis_{name}_monthly.csv", index=False, encoding="utf-8-sig")
    rep["monthly"] = monthly

    top_codes = ev["code"].value_counts().head(5)
    rep["top_codes"] = top_codes
    rep["top_code_share"] = float(top_codes.iloc[0] / len(ev) * 100) if len(top_codes) else 0.0

    # ---- B. 前兆路径：改用区间累计涨幅（正值 = 上涨）----
    p = prec.copy()
    p["cum_gain"] = cum_gain_from_norm(p["close_norm"])
    path = p.groupby("offset")["cum_gain"].agg(
        median="median", mean="mean",
        p25=lambda s: float(s.quantile(0.25)),
        p75=lambda s: float(s.quantile(0.75))).reset_index()
    path.to_csv(out_dir / f"analysis_{name}_path.csv", index=False, encoding="utf-8-sig")
    rep["path"] = path

    # ---- C. 信号前一日特征 vs 全库基准 ----
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
            "特征": FEATURE_CN.get(f, f), "信号前一日均值": s_mean,
            "全库均值": b_mean, "全库标准差": b_std,
            "z": (s_mean - b_mean) / b_std if b_std else np.nan,
            "信号前一日中位数": float(last[f].median()),
            "全库中位数": float(base[f].median()),
        })
    comp = pd.DataFrame(comp_rows)
    comp = comp.reindex(comp["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    comp.to_csv(out_dir / f"analysis_{name}_features_vs_baseline.csv",
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
    dif.to_csv(out_dir / f"analysis_{name}_win_loss.csv", index=False, encoding="utf-8-sig")
    rep["dif"] = dif

    pw = prec.merge(ev[["code", "signal_date", "win_5d"]], on=["code", "signal_date"], how="inner")
    pw = pw.dropna(subset=["win_5d"]).copy()
    if len(pw):
        pw["cum_gain"] = cum_gain_from_norm(pw["close_norm"])
        pwp = pw.groupby(["offset", "win_5d"])["cum_gain"].mean().unstack()
        pwp.columns = ["亏损路径" if int(c) == 0 else "盈利路径" for c in pwp.columns]
        pwp = pwp.reset_index()
        pwp.to_csv(out_dir / f"analysis_{name}_path_by_outcome.csv",
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
    corr.to_csv(out_dir / f"analysis_{name}_corr.csv", index=False, encoding="utf-8-sig")
    rep["corr"] = corr

    dec = pd.DataFrame()
    top_f = None
    if not corr.empty:
        top_f = corr.iloc[0]["特征"]
        col = next((k for k, v in FEATURE_CN.items() if v == top_f), top_f)
        s = last[[col, "ret_5d"]].dropna().copy()
        if len(s) >= 25:
            try:
                s["q"] = pd.qcut(s[col], 5, duplicates="drop")
                dec = s.groupby("q", observed=True).agg(
                    n=("ret_5d", "size"), 均值=("ret_5d", "mean"),
                    胜率=("ret_5d", lambda x: float((x > 0).mean() * 100)),
                    区间下限=(col, "min"), 区间上限=(col, "max")).reset_index(drop=True)
                dec.to_csv(out_dir / f"analysis_{name}_quintile.csv",
                           index=False, encoding="utf-8-sig")
            except Exception:  # noqa: BLE001
                dec = pd.DataFrame()
        rep["quintile_feature"] = top_f
        rep["quintile_col"] = col
        rep["quintile"] = dec
        rep["whatif"] = _quintile_whatif(last, col)

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
    """用已算出的数字生成结论要点（不写死数值，避免报告与数据脱节）。"""
    out: list[str] = []
    p = r.get("path")
    if p is not None and not p.empty:
        def gv(off: int, col: str = "median") -> float:
            s = p.loc[p["offset"] == off, col]
            return float(s.iloc[0]) if len(s) else float("nan")
        g60, g20, g5 = gv(-60), gv(-20), gv(-5)
        out.append(f"- **前兆形态**：信号日之前的价格路径（中位）——60 日前累计涨幅 "
                   f"{g60:.1f}%、20 日前 {g20:.1f}%、5 日前 {g5:.1f}%、1 日前 {gv(-1):.1f}%"
                   f"（正值＝当时价格低于信号日价格）。")
        # 路径上的最低价点（cum_gain 最大处）：区分「持续上行」与「先跌后涨」。
        # 判据用「峰值是否出现在窗口头部 20% 之内」而非严格等于 -window，
        # 避免中位数路径在首两个偏移点之间的微小抖动把形态误判成 V 形。
        try:
            pk = p.loc[p["median"].idxmax()]
            pk_off, pk_v = int(pk["offset"]), float(pk["median"])
            w = abs(int(p["offset"].min()))
            sig_chg = r.get("sig_pct_med", float("nan"))
            day_txt = (f"信号日仍收阳（中位 +{sig_chg:.2f}%）" if sig_chg > 0
                       else f"信号日本身转为阴线回调（中位 {sig_chg:.2f}%）")
            if pk_off <= -0.8 * w:
                share = (g5 / g60 * 100) if g60 > 0 else float("nan")
                out.append(f"  - 属于**持续上行**形态：最低价出现在窗口起点附近"
                           f"（{pk_off} 日前，低于信号日 {pk_v:.1f}%），此后单边上行；"
                           f"最近 5 日贡献 +{g5:.1f}%，占整段 60 日涨幅的 {share:.0f}%。")
            else:
                # cum_gain 是「该日 → 信号日」的涨幅，故从最低点到信号前一日
                # 的回升幅度 = (1+mx/100)/(1+g(-1)/100) - 1
                base = gv(-1)
                rise = (1.0 + pk_v / 100.0) / (1.0 + base / 100.0) - 1.0
                out.append(f"  - 属于**先跌后涨（V 形）**形态：最低于信号日 "
                           f"{pk_v:.1f}% 出现在 {pk_off} 日前，此后一路反弹——"
                           f"从最低点回升 {rise * 100:.1f}% 到信号前一日；{day_txt}。")
        except Exception:  # noqa: BLE001
            pass
    dif = r.get("dif")
    if dif is not None and not dif.empty:
        t0 = dif.iloc[0]
        out.append(f"- **最能区分盈亏的前兆特征**：{t0['特征']}"
                   f"（盈利均值 {t0['盈利均值']:.2f} vs 亏损均值 {t0['亏损均值']:.2f}，t = {t0['t值']:.2f}）。")
    wf = r.get("whatif")
    if wf is not None and not wf.empty:
        base = wf.iloc[0]
        best = wf.iloc[1:].sort_values("5日均收益%", ascending=False).iloc[0]
        out.append(f"- **可执行改进**：基准（全部锚点）5 日胜率 {base['5日胜率%']:.1f}%、"
                   f"均收益 {base['5日均收益%']:.2f}%；"
                   f"采用「{best['方案']}」后为 {best['5日胜率%']:.1f}% / {best['5日均收益%']:.2f}%"
                   f"（锚点数 {int(base['锚点数'])} → {int(best['锚点数'])}）。")
    m = r.get("monthly")
    if m is not None and not m.empty and len(m) >= 3:
        worst = m.sort_values("win_rate").iloc[0]
        bestm = m.sort_values("win_rate").iloc[-1]
        out.append(f"- **月份波动极大**：最差 {worst['month']}（胜率 {worst['win_rate']:.1f}%、"
                   f"均收益 {worst['avg_ret']:.2f}%），最好 {bestm['month']}"
                   f"（{bestm['win_rate']:.1f}%、{bestm['avg_ret']:.2f}%）——"
                   f"说明收益高度依赖市场风格，而非稳定 alpha。")
    return out


def build_report(reps: list[dict], out_md: Path) -> None:
    md: list[str] = []
    md.append("# 新策略信号前兆分析报告\n")
    md.append("**样本库**：`data_library/combined_daily.parquet`（2025-11-12 ~ 2026-10-09，788 只）  ")
    md.append("**锚点**：策略逐日全量判定的**全部信号**（不施加回测冷却期，避免锚点被稀疏化）  ")
    md.append("**前兆窗口**：信号日之前的 60 个交易日（offset = −60 … −1，−1 = 信号前一日）  ")
    md.append("**前向结局**：T+1 开盘入场、持有 3/5/10 个交易日收盘卖出，扣除双边 0.1% 手续费  ")
    md.append("**符号约定**：正文中的「区间累计涨幅」= 信号日收盘 / 该日收盘 − 1，"
              "即从该偏移日持有到信号日的收益，**正值表示上涨**。\n")

    md.append("## 一、信号概览\n")
    ov = [{
        "策略": r["strategy"], "锚点总数": r.get("n_signals", 0),
        "覆盖股票": r.get("n_codes", 0), "可评估": r.get("n_eval", 0),
        "首信号": r.get("date_min", ""), "末信号": r.get("date_max", ""),
        "信号日涨幅中位%": r.get("sig_pct_med", np.nan),
        "信号日换手中位%": r.get("sig_turn_med", np.nan),
        "3日胜率%": r.get("win_rate_3d", np.nan),
        "5日胜率%": r.get("win_rate_5d", np.nan),
        "10日胜率%": r.get("win_rate_10d", np.nan),
        "5日均收益%": r.get("avg_ret_5d", np.nan),
        "5日中位收益%": r.get("med_ret_5d", np.nan),
        "5日超额%": r.get("excess_5d", np.nan),
    } for r in reps]
    md.append(_md_table(pd.DataFrame(ov)))

    md.append("\n## 二、前兆价格路径（信号前 60 日的区间累计涨幅）\n")
    md.append("该表直接回答「**这个策略在追什么形态**」——信号发生时，股票此前已经走了多远。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        p = r.get("path")
        if p is not None and not p.empty:
            sel = p[p["offset"].isin(MARK_OFFSETS)].rename(columns={
                "offset": "偏移日", "median": "中位涨幅%", "mean": "均涨幅%",
                "p25": "p25涨幅%", "p75": "p75涨幅%"})
            md.append(_md_table(sel))
        if r.get("pwp") is not None and not r["pwp"].empty:
            md.append("\n**盈利组 vs 亏损组的前兆路径（区间累计涨幅均值）**\n")
            sel = r["pwp"][r["pwp"]["offset"].isin(MARK_OFFSETS)].rename(columns={"offset": "偏移日"})
            md.append(_md_table(sel))

    md.append("\n## 三、信号前一日特征 vs 全库基准\n")
    md.append("z =（信号前一日均值 − 全库均值）/ 全库标准差。**|z| 越大，说明该特征越能把信号日"
              "与非信号日区分开**，即策略在特征空间里真正捕捉的维度。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("comp")))

    md.append("\n## 四、盈亏信号的前兆差异（核心）\n")
    md.append("按前向 5 日收益分成盈利/亏损两组，比较**信号前一日**各特征均值。"
              "|t| 越大，该前兆特征越能区分后续盈亏（|t| > 2 视为显著）。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("dif")))

    md.append("\n## 五、前兆特征 → 前向 5 日收益\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        md.append(_md_table(r.get("corr")))
        dec = r.get("quintile")
        if dec is not None and not dec.empty:
            md.append(f"\n**按前兆特征「{r.get('quintile_feature')}」五档分位的后续表现**\n")
            dec2 = dec.rename(columns={"n": "锚点数", "均值": "5日均收益%", "胜率": "5日胜率%"})
            md.append(_md_table(dec2))

    md.append("\n## 六、过滤假设检验（把发现变成可执行规则）\n")
    md.append("基于上节最显著的前兆特征做五档分位，检验「砍掉某一端」能否改善表现。\n")
    for r in reps:
        md.append(f"\n### {r['strategy']} — 特征「{r.get('quintile_feature')}」\n")
        md.append(_md_table(r.get("whatif")))

    md.append("\n## 七、信号月份分布\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        m = r.get("monthly")
        if m is not None and not m.empty:
            m = m.rename(columns={"month": "月份", "n": "锚点数",
                                  "win_rate": "5日胜率%", "avg_ret": "5日均收益%"})
            md.append(_md_table(m))
        tc = {k: int(v) for k, v in dict(r.get("top_codes", {})).items()}
        md.append(f"\n锚点最集中的 5 只股票：{tc}；单只股票最多占 {r.get('top_code_share', 0):.2f}% 的锚点。\n")

    md.append("\n## 八、结论与改进方向\n")
    for r in reps:
        md.append(f"\n### {r['strategy']}\n")
        for line in _conclusion(r):
            md.append(line)
    md.append("\n> 说明：以上结论均由本次样本（2025-11-12 ~ 2026-10-09，788 只）自动计算得出；"
              "样本期约 11 个月，且策略信号在月份间胜率波动极大，"
              "任何「改进方案」都需在更长样本上复验后再落地。\n")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] -> {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser(description="信号前兆分析")
    ap.add_argument("--strategies", required=True, help="策略名，逗号分隔")
    ap.add_argument("--library", default="data_library/combined_daily.parquet")
    ap.add_argument("--data-dir", default="research_output")
    ap.add_argument("--report", default="docs/信号前兆分析报告.md")
    args = ap.parse_args()

    names = [x.strip() for x in args.strategies.split(",") if x.strip()]
    ddir = ROOT / args.data_dir

    daily = pd.read_parquet(ROOT / args.library)
    daily["date"] = daily["date"].astype(str)
    print("[base] 计算全库股票日基准特征…")
    base = baseline_features(daily)
    print(f"[base] {len(base)} 个股票日")

    reps = []
    for n in names:
        ppath, opath = ddir / f"precursor_{n}.csv", ddir / f"outcomes_{n}.csv"
        if not ppath.exists() or not opath.exists():
            print(f"[warn] 缺少 {n} 的数据集，跳过")
            continue
        prec = pd.read_csv(ppath, encoding="utf-8-sig")
        outc = pd.read_csv(opath, encoding="utf-8-sig")
        print(f"[load] {n}: 长表 {len(prec)} 行 / 锚点 {len(outc)} 个")
        reps.append(analyse_one(n, prec, outc, base, ddir))

    if not reps:
        print("[error] 无可用数据")
        return 1
    build_report(reps, ROOT / args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
