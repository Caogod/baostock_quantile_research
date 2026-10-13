"""实验C：**回归**任务 —— 用 volume_breakout 的回测数据训练回归模型，并可视化测试样本误差曲线。

与分类任务的关系（核心：只换任务，不换模型策略）
    train_model.py 走的是**二分类**：先把连续收益二值化成 `win = 超额收益 > 0`，
    再训 24 个特征 → 胜率概率。二值化丢掉了**幅度**信息（+0.1% 与 +20% 同为"赢"）。
    本实验保持**完全相同的模型策略**：
      - 同样的 6 个模型族（决策树 / 随机森林 / GBDT / XGBoost / CatBoost / LightGBM）
        且超参镜像（深度、树数、学习率、子采样等逐项对应）
      - 同样的 TimeSeriesSplit(5) 时序交叉验证（按 entry_date 升序分折）
      - 同样的 Bottom-N RFE 特征消除（Bottom-3、patience 3，CV 不再提升即停）
      - 同样的"按 CV 指标选最优"留存策略
    唯一变化：模型换成对应的**回归器**，预测目标换成**连续的超额收益率**（%）。
    另加 DummyRegressor（预测训练集均值）作为"模型是否真的学到东西"的下限基准。

可视化（测试样本误差曲线）
    1. 各模型逐折测试误差（MAE）折线 —— 看折间稳定性
    2. 最优模型 OOF 测试误差 随时间（按 entry_date）曲线 + 滚动均值 —— 看误差是否漂移
    3. 学习曲线（训练集规模 vs 训练/测试 MAE）—— 看是欠拟合还是数据量不足
    4. OOF 预测 vs 实际散点（含 y=x 参考线）
    绘图用 `experiments/svg_chart.py`（无依赖 SVG），报告为自包含 HTML。
    说明：此前是因为 conda `stock` 环境的 matplotlib 二进制与 Python 版本不匹配、
    无法 import 才自绘 SVG（2026-10-14 已把该环境升级到 matplotlib 3.11.2，现在也能用
    matplotlib 出 PNG）；保留 svg_chart 是取其"零依赖 + 矢量 + 可内嵌"的好处。

用法：
    python experiments/exp_reg_volume_breakout.py
    python experiments/exp_reg_volume_breakout.py --target return_pct --no-rfe
    python experiments/exp_reg_volume_breakout.py --detail backtest_output/backtest_detail_x_y.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.datasets import resolve_detail_path, resolve_holdings, resolve_library_path  # noqa: E402
from core.features import FEATURES, precompute_features  # noqa: E402
from core.market import benchmark_returns_for_trades  # noqa: E402
from experiments.svg_chart import Chart, html_report  # noqa: E402

N_SPLITS = 5
BOTTOM_N = 3
PATIENCE = 3
MAX_ROUNDS = 20


def build_regressors() -> dict[str, object]:
    """6 个回归器，超参与 train_model.build_models() 的分类器逐项镜像。"""
    # 延迟导入，避免 lightgbm/xgboost 与 sklearn 的版本告警干扰输出
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return {
        "决策树(DT)": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=0),
        "随机森林(RF)": RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=10, n_jobs=-1, random_state=0),
        "GBDT": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=0),
        "CatBoost": CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=4, random_seed=0, verbose=0,
            allow_writing_files=False),
        "LightGBM": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=-1),
    }


def cv_mae(model_factory, X, y, tscv) -> float:
    """时序 CV 的负 MAE 取正（与分类侧 cross_val_score(scoring='accuracy') 等价位置）。"""
    errs = []
    for tr, va in tscv.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        errs.append(mean_absolute_error(y[va], m.predict(X[va])))
    return float(np.mean(errs))


def rfe_regression(X, y, feature_names, tscv, bottom_n=BOTTOM_N, patience=PATIENCE):
    """Bottom-N RFE（与 train_model.rfe_select_features 同策略），CV 指标换成 MAE。"""
    from xgboost import XGBRegressor

    def factory():
        return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=0)

    current = list(feature_names)
    best_feats, best_mae, no_improve = list(feature_names), float("inf"), 0
    rounds: list[dict] = []
    for _ in range(MAX_ROUNDS):
        Xc = X[:, [feature_names.index(f) for f in current]]
        mae = cv_mae(factory, Xc, y, tscv)
        rounds.append({"n_features": len(current), "cv_mae": mae, "features": list(current)})
        print(f"    RFE 轮次 {len(rounds)-1}: 特征 {len(current)}，CV MAE {mae:.4f}%"
              f"（历史最优 {best_mae:.4f}%）")
        if mae < best_mae - 1e-9:
            best_mae, best_feats, no_improve = mae, list(current), 0
        else:
            no_improve += 1
        if no_improve >= patience or len(current) <= bottom_n:
            break
        m = factory()
        m.fit(Xc, y)
        imp = getattr(m, "feature_importances_", None)
        if imp is None:
            break
        drop = [current[i] for i in np.argsort(imp)[:bottom_n]]
        current = [f for f in current if f not in set(drop)]
        print(f"      -> 剔除 Bottom-{bottom_n}: {drop}")
    return best_feats, rounds


def fold_evaluate(model_factory, X, y, tscv, dates=None):
    """逐折拟合，返回 (每折指标列表, OOF 预测数组, OOF 所属折号)。"""
    n = len(y)
    oof = np.full(n, np.nan)
    fold_of = np.full(n, -1, dtype=int)
    folds = []
    for k, (tr, va) in enumerate(tscv.split(X), start=1):
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        oof[va] = pred
        fold_of[va] = k
        folds.append({
            "fold": k, "n_train": len(tr), "n_test": len(va),
            "mae": mean_absolute_error(y[va], pred),
            "rmse": float(np.sqrt(mean_squared_error(y[va], pred))),
            "r2": r2_score(y[va], pred) if len(va) > 1 else float("nan"),
            "test_start": None if dates is None else str(dates[va[0]]),
            "test_end": None if dates is None else str(dates[va[-1]]),
        })
    return folds, oof, fold_of


def table_html(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            tds.append(f"<td>{float_fmt.format(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def main() -> int:
    ap = argparse.ArgumentParser(description="回归任务实验（volume_breakout 数据集）")
    ap.add_argument("--strategies", default="volume_breakout", help="逗号分隔，用于自动解析明细")
    ap.add_argument("--detail", default="", help="回溯明细 CSV（空=自动解析）")
    ap.add_argument("--library", default="", help="本地样本库 Parquet（空=环境变量/默认）")
    ap.add_argument("--holding", default="auto", help="持仓周期口径（auto=按各策略 config）")
    ap.add_argument("--target", default="excess_return_pct",
                    choices=["excess_return_pct", "return_pct"],
                    help="回归目标：超额收益（默认，与分类标签同口径）或原始净收益")
    ap.add_argument("--no-rfe", action="store_false", dest="rfe", default=True,
                    help="关闭 RFE，直接用全 24 特征")
    ap.add_argument("--out", default="research_output", help="CSV 输出目录")
    ap.add_argument("--report", default="docs/回归实验_volume_breakout.md", help="Markdown 报告路径")
    ap.add_argument("--html", default="docs/回归实验_volume_breakout.html", help="HTML 图表报告路径")
    args = ap.parse_args()

    req = [s.strip() for s in args.strategies.split(",") if s.strip()]
    t0 = time.time()
    detail_path = resolve_detail_path(args.detail or None, required=req, verbose=True)
    library_path = resolve_library_path(args.library or None, verbose=True)
    print(f"[data] 明细: {detail_path}\n[data] 样本库: {library_path}")

    lib = pd.read_parquet(library_path).sort_values("date")
    detail = pd.read_csv(detail_path)
    if "holding_days" in detail.columns:
        detail["holding_days"] = pd.to_numeric(detail["holding_days"], errors="coerce")
    detail = detail[detail["strategy"].isin(req)].copy()
    holdings = resolve_holdings(detail, args.holding, verbose=True)

    # 单策略实验：取该策略解析出的持仓周期
    if holdings:
        h = holdings[req[0]] if req[0] in holdings else list(holdings.values())[0]
        before = len(detail)
        detail = detail[detail["holding_days"] == h].copy()
        print(f"[data] 持仓周期 {h} 日：{before} -> {len(detail)} 笔")
    else:
        h = None

    print(f"[data] 对齐特征（{len(detail)} 笔交易）...")
    feat_all = precompute_features(lib)
    df = detail.merge(feat_all, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")
    bench = benchmark_returns_for_trades(df) * 100.0
    df["bench_return_pct"] = bench.round(4)
    df["excess_return_pct"] = (df["return_pct"].astype(float) - df["bench_return_pct"]).round(4)
    df["win"] = (df["excess_return_pct"] > 0).astype(int)
    df = df.dropna(subset=FEATURES + [args.target]).sort_values("entry_date").reset_index(drop=True)

    X_all = df[FEATURES].values
    y = df[args.target].astype(float).values
    # 注：pandas 3.x 默认 pyarrow 字符串，.values 是 ArrowStringArray（没有 .min()），
    # 这里统一转成 object ndarray，避免后续 min/max 与索引出错。
    date_s = df["entry_date"].astype(str)
    dates = date_s.to_numpy(dtype=object)
    dmin, dmax = str(date_s.min()), str(date_s.max())
    codes = np.asarray(df["code"].values, dtype=object)
    strats = np.asarray(df["strategy"].values, dtype=object)
    y_cls = np.asarray(df["win"].values, dtype=int)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    print(f"\n数据集: {len(df)} 个样本 | {dmin} ~ {dmax} | "
          f"{df['code'].nunique()} 只股票 | 持仓 {h} 日")
    print(f"目标 [{args.target}]: 均值 {y.mean():.3f}% 中位 {np.median(y):.3f}% "
          f"标准差 {y.std():.3f}% | 超额胜率 {df['win'].mean()*100:.1f}%")

    # ---- 1) RFE 特征消除 ----
    print("\n=== 1) Bottom-N RFE（XGBoost 回归，CV 指标 = MAE）===")
    active = list(FEATURES)
    rfe_rounds: list[dict] = []
    if args.rfe:
        active, rfe_rounds = rfe_regression(X_all, y, FEATURES, tscv)
        print(f"  RFE 特征 {len(FEATURES)} -> {len(active)}"
              f"（CV MAE {rfe_rounds[0]['cv_mae']:.4f}% -> {min(r['cv_mae'] for r in rfe_rounds):.4f}%）")
    idx = [FEATURES.index(f) for f in active]
    X = X_all[:, idx]

    # ---- 2) 6 个回归器 + 均值基准，逐折评估 ----
    print("\n=== 2) 模型横评（TimeSeriesSplit 5 折，指标 MAE / RMSE / R²）===")
    regressors = build_regressors()
    regressors["均值基准"] = DummyRegressor(strategy="mean")

    # TimeSeriesSplit 的首折只做训练、不做验证，故 OOF 预测只覆盖后 5/6 的样本；
    # 所有 OOF 口径的指标（MAE / 方向命中）只在被覆盖的样本上统计，避免把 NaN 计入。
    covered = np.zeros(len(y), dtype=bool)
    for _, va in tscv.split(X_all):
        covered[va] = True
    n_oof = int(covered.sum())
    # 分类侧的可比基线：CV 只在被覆盖的折上评估，故多数类基线也必须用同一子集，
    # 否则会拿"全样本多数类 58.99%"去比"覆盖子集上的准确率"，口径不一致。
    maj_cov = float(max(y_cls[covered].mean(), 1 - y_cls[covered].mean()) * 100)
    print(f"  OOF 覆盖 {n_oof}/{len(y)} 个样本（首折 {len(y) - n_oof} 个仅用于训练，无预测值）")
    print(f"  覆盖子集多数类基线 {maj_cov:.2f}%（全样本为 "
          f"{max(y_cls.mean(), 1 - y_cls.mean())*100:.2f}%，两者不可混用）")

    rows, folds_by_model, oof_by_model, foldid_by_model = [], {}, {}, {}
    for name, model in regressors.items():
        factory = (lambda m=model: type(m)(**m.get_params()))
        folds, oof, fold_of = fold_evaluate(factory, X, y, tscv, dates)
        folds_by_model[name] = folds
        oof_by_model[name] = oof
        foldid_by_model[name] = fold_of
        rows.append({
            "模型": name,
            "CV_MAE(%)": float(np.mean([f["mae"] for f in folds])),
            "CV_MAE_std": float(np.std([f["mae"] for f in folds])),
            "CV_RMSE(%)": float(np.mean([f["rmse"] for f in folds])),
            "CV_R2": float(np.mean([f["r2"] for f in folds])),
            "OOF_MAE(%)": float(mean_absolute_error(y[covered], oof[covered])),
            "方向命中率(%)": float((((oof > 0).astype(int))[covered] == y_cls[covered]).mean() * 100),
        })
        print(f"  {name:10s} CV MAE {rows[-1]['CV_MAE(%)']:.4f}% ± {rows[-1]['CV_MAE_std']:.4f} "
              f"| RMSE {rows[-1]['CV_RMSE(%)']:.4f}% | R² {rows[-1]['CV_R2']:.4f} "
              f"| 方向命中 {rows[-1]['方向命中率(%)']:.2f}%")

    metrics = pd.DataFrame(rows).sort_values("CV_MAE(%)").reset_index(drop=True)
    best_name = metrics.loc[0, "模型"]
    best_oof = oof_by_model[best_name]
    best_model = regressors[best_name]
    baseline_mae = float(metrics.loc[metrics["模型"] == "均值基准", "CV_MAE(%)"].iloc[0])
    y_cov = y[covered]  # OOF 覆盖到的样本（首折只训练不验证）
    print(f"\n  最优模型: {best_name}（CV MAE {metrics.loc[0,'CV_MAE(%)']:.4f}%，"
          f"均值基准 {baseline_mae:.4f}%，改善 {baseline_mae - metrics.loc[0,'CV_MAE(%)']:+.4f}pp）")

    # ---- 3) 学习曲线 ----
    print("\n=== 3) 学习曲线（最优模型）===")
    sizes, tr_sc, va_sc = learning_curve(
        type(best_model)(**best_model.get_params()), X, y, cv=tscv,
        scoring="neg_mean_absolute_error", train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1)
    lc = pd.DataFrame({
        "n_train": sizes,
        "train_MAE(%)": -tr_sc.mean(axis=1),
        "test_MAE(%)": -va_sc.mean(axis=1),
        "test_MAE_std": va_sc.std(axis=1),
    })
    for _, r in lc.iterrows():
        print(f"  训练 {int(r['n_train']):5d} -> 训练 MAE {r['train_MAE(%)']:.4f}% | "
              f"测试 MAE {r['test_MAE(%)']:.4f}%")

    # ---- 4) 与分类任务的同数据对照 ----
    print("\n=== 4) 同数据分类对照（同一套模型策略，任务换成二分类）===")
    from sklearn.model_selection import cross_val_score
    from train_model import build_models
    cls_rows = []
    for name, model in build_models().items():
        acc = cross_val_score(model, X, y_cls, cv=tscv, scoring="accuracy").mean() * 100
        cls_rows.append({"模型": name, "CV准确率(%)": float(acc)})
        print(f"  {name:10s} CV 准确率 {acc:.2f}%")
    cls_df = pd.DataFrame(cls_rows).sort_values("CV准确率(%)", ascending=False).reset_index(drop=True)
    best_cls = cls_df.iloc[0]
    print(f"  覆盖子集多数类基线 {maj_cov:.2f}% → 最优分类模型 {best_cls['模型']} "
          f"{best_cls['CV准确率(%)']:.2f}%（{best_cls['CV准确率(%)']-maj_cov:+.2f}pp）")
    reg_dir = float(metrics.iloc[0]["方向命中率(%)"])
    print(f"  对照：最优回归模型 {best_name} 的方向命中率 {reg_dir:.2f}%"
          f"（{reg_dir-maj_cov:+.2f}pp）")

    # ---- 5) 绘图（误差曲线只用 OOF 覆盖到的样本）----
    err = np.abs(y - best_oof)[covered]
    x_idx = np.arange(len(err))
    roll = pd.Series(err).rolling(40, min_periods=10).mean().to_numpy()
    palette_models = [m for m in metrics["模型"]]

    chart = Chart(
        title="volume_breakout 回归任务 · 测试样本误差曲线",
        subtitle=f"数据: volume_breakout 回测明细（持仓 {h} 日）× {len(df)} 笔 "
                 f"| 目标: {args.target}（连续收益 %） | TimeSeriesSplit {N_SPLITS} 折时序 CV "
                 f"| 模型策略与分类任务完全一致（仅换成回归器）",
        ncols=2, nrows=2)
    folds_x = list(range(1, N_SPLITS + 1))
    ax1 = chart.axes(0, 0, "① 各模型逐折测试误差（MAE）", "折", "MAE(%)")
    ax1.ylim = (0, max(float(np.max([f["mae"] for f in v])) for v in folds_by_model.values()) * 1.12)
    for i, name in enumerate(palette_models):
        ax1.line(folds_x, [f["mae"] for f in folds_by_model[name]], chart.color(i),
                 label=name, marker=True,
                 dash="5 4" if name == "均值基准" else None,
                 width=1.4 if name == "均值基准" else 2.0)

    cov_dates = dates[covered]
    step = max(1, len(err) // 6)
    tickpos = list(range(0, len(err), step))
    ax2 = chart.axes(0, 1, f"② 最优模型（{best_name}）OOF 测试误差随时间",
                     "入场日（OOF 覆盖样本按时间排序）", "绝对误差(%)",
                     xticks=tickpos, xtick_labels=[str(cov_dates[p])[5:] for p in tickpos])
    ax2.ylim = (0, float(np.nanpercentile(err, 98)) * 1.05)
    ax2.scatter(x_idx, err, chart.color(0), r=1.6, opacity=0.35, label="样本绝对误差")
    ax2.line(x_idx, roll, chart.color(1), label="40 样本滚动均值", width=2.0)

    ax3 = chart.axes(1, 0, "③ 学习曲线（最优模型）", "训练样本数", "MAE(%)")
    ax3.ylim = (0, float(max(lc["train_MAE(%)"].max(), lc["test_MAE(%)"].max())) * 1.15)
    ax3.line(lc["n_train"], lc["train_MAE(%)"], chart.color(2), label="训练集 MAE", marker=True)
    ax3.line(lc["n_train"], lc["test_MAE(%)"], chart.color(3), label="测试集 MAE", marker=True)

    ax4 = chart.axes(1, 1, "④ OOF 预测 vs 实际（最优模型）", "实际收益(%)", "预测收益(%)")
    lo = float(min(y[covered].min(), np.nanmin(best_oof[covered])))
    hi = float(max(y[covered].max(), np.nanmax(best_oof[covered])))
    ax4.xlim, ax4.ylim = (lo, hi), (lo, hi)
    ax4.scatter(y[covered], best_oof[covered], chart.color(0), r=1.8, opacity=0.35, label="样本")
    ax4.line([lo, hi], [lo, hi], chart.color(8), label="理想 y = x", dash="6 4", width=1.6)

    svg = chart.svg()
    svg_path = Path(args.out) / "reg_volume_breakout_error.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")

    # ---- 6) 落盘 ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "reg_volume_breakout_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([pd.DataFrame(folds_by_model[m]).assign(模型=m) for m in folds_by_model]) \
        .to_csv(out / "reg_volume_breakout_folds.csv", index=False, encoding="utf-8-sig")
    lc.to_csv(out / "reg_volume_breakout_learning_curve.csv", index=False, encoding="utf-8-sig")
    oof_df = pd.DataFrame({
        "code": codes, "entry_date": dates, "strategy": strats,
        "holding_days": np.asarray(df.get("holding_days", pd.Series([h] * len(df))).values),
        "y_true": y, "y_pred": best_oof, "error": best_oof - y,
        "abs_error": np.abs(y - best_oof),
        "win": y_cls, "pred_up": (best_oof > 0).astype(int),
        "fold": foldid_by_model[best_name], "oof_covered": covered.astype(int),
    })
    oof_df.to_csv(out / "reg_volume_breakout_oof.csv", index=False, encoding="utf-8-sig")

    # ---- 7) HTML + Markdown 报告 ----
    metric_tbl = metrics.copy()
    folds_tbl = pd.concat(
        [pd.DataFrame(folds_by_model[m]).assign(模型=m)[
            ["模型", "fold", "n_train", "n_test", "mae", "rmse", "r2", "test_start", "test_end"]]
         for m in ["均值基准", best_name] if m in folds_by_model],
        ignore_index=True)

    sec_metrics = table_html(metric_tbl.round(4))
    sec_folds = table_html(folds_tbl.round(4))
    sec_lc = table_html(lc.round(4))
    sec_cls = table_html(cls_df.round(2))

    gap = baseline_mae - metrics.iloc[0]["CV_MAE(%)"]
    at_bottom = lc["test_MAE(%)"].iloc[-1] - lc["test_MAE(%)"].iloc[-2]
    overfit_gap = lc["train_MAE(%)"].iloc[-1] - lc["test_MAE(%)"].iloc[-1]
    fold_min = float(np.min([f["mae"] for f in folds_by_model[best_name]]))
    fold_max = float(np.max([f["mae"] for f in folds_by_model[best_name]]))

    summary = f"""<ul>
<li><b>最优模型</b>：{best_name}，CV MAE <b>{metrics.iloc[0]['CV_MAE(%)']:.4f}%</b>
（均值基准 {baseline_mae:.4f}%，改善 <b>{gap:+.4f}pp</b>），R²
{metrics.iloc[0]['CV_R2']:.4f} —— 几乎等于"直接预测平均值"。</li>
<li><b>折间稳定性</b>：{best_name} 的逐折 MAE 在 {fold_min:.3f}% ~ {fold_max:.3f}% 之间
（标准差 {metrics.iloc[0]['CV_MAE_std']:.4f}）。</li>
<li><b>学习曲线</b>：训练集 MAE {lc['train_MAE(%)'].iloc[-1]:.4f}% vs 测试集
{lc['test_MAE(%)'].iloc[-1]:.4f}%，泛化差距 {overfit_gap:+.4f}pp；
最后一段测试误差变化 {at_bottom:+.4f}pp（接近 0 说明增加数据收益有限）。</li>
<li><b>误差水平</b>：覆盖子集目标收益标准差 {y_cov.std():.3f}%，模型 MAE
{metrics.iloc[0]['CV_MAE(%)']:.4f}% ≈ 其
{metrics.iloc[0]['CV_MAE(%)']/y_cov.std()*100:.1f}%。</li>
<li><b>预测被压缩</b>：OOF 预测值标准差 {np.std(best_oof[covered]):.3f}%
vs 实际 {y_cov.std():.3f}%（仅
{np.std(best_oof[covered])/y_cov.std()*100:.0f}%）——模型把预测挤向均值，
说明它抓不到收益幅度。</li>
<li><b>与分类任务对照</b>（同一数据、同一套模型策略、同一批折）：
分类最优 {best_cls['模型']} CV 准确率 <b>{best_cls['CV准确率(%)']:.2f}%</b>
vs 覆盖子集多数类基线 {maj_cov:.2f}%（<b>{best_cls['CV准确率(%)']-maj_cov:+.2f}pp</b>）；
回归最优 {best_name} 的方向命中率 <b>{reg_dir:.2f}%</b>（{reg_dir-maj_cov:+.2f}pp）。
即：<b>方向还有一点信号，幅度几乎无信号</b>。</li>
<li><b>口径说明</b>：TimeSeriesSplit 首折只训练不验证，OOF 覆盖 {n_oof}/{len(y)} 个样本，
以上 OOF 口径指标均只在该子集上统计。</li>
</ul>"""

    html = html_report(
        "volume_breakout 回归任务 · 测试样本误差曲线",
        f"生成于 {datetime.now():%Y-%m-%d %H:%M} ｜ 明细 {Path(detail_path).name} ｜ "
        f"样本 {len(df)} 笔（{df['code'].nunique()} 只股票，{dmin} ~ {dmax}）｜ "
        f"持仓 {h} 日 ｜ 目标 {args.target} ｜ RFE {'开启' if args.rfe else '关闭'}"
        f"（{len(FEATURES)} → {len(active)} 特征）",
        svg,
        [("结论摘要", summary),
         ("各模型 CV 误差横评（按 MAE 升序）", sec_metrics),
         (f"逐折明细（{best_name} 与均值基准）", sec_folds),
         ("学习曲线数据", sec_lc),
         ("同数据分类对照（同一模型策略，任务换成二分类）", sec_cls)])

    html_path = Path(args.html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    md = [
        "# 回归任务实验：volume_breakout 数据集",
        "",
        f"> 生成于 {datetime.now():%Y-%m-%d %H:%M}；脚本 `experiments/exp_reg_volume_breakout.py`；"
        f"图表报告 `{html_path}`。",
        "",
        "## 一、实验目的",
        "分类任务把连续收益二值化成 `win = 超额收益 > 0`，丢掉了幅度信息。本实验保持"
        "**完全相同的模型策略**（同样 6 个模型族 + 镜像超参 + TimeSeriesSplit(5) + Bottom-N RFE + "
        "按 CV 指标选最优），只把任务换成**回归**（直接预测连续超额收益率），"
        "并可视化测试样本误差曲线。",
        "",
        "## 二、数据集",
        f"- 明细：`{Path(detail_path).name}`（策略 {req}，持仓 {h} 日）",
        f"- 样本：{len(df)} 笔，{df['code'].nunique()} 只股票，{dmin} ~ {dmax}",
        f"- 特征：{len(FEATURES)} 个（`core/features.py` 的口径，与分类任务一致）"
        f"{f'，RFE 后保留 {len(active)} 个' if args.rfe else ''}",
        f"- 目标 `{args.target}`：均值 {y.mean():.3f}%、中位 {np.median(y):.3f}%、"
        f"标准差 {y.std():.3f}%、超额胜率 {df['win'].mean()*100:.1f}%",
        "",
        "## 三、结论",
        f"- **最优模型 {best_name}**：CV MAE **{metrics.iloc[0]['CV_MAE(%)']:.4f}%**"
        f"（±{metrics.iloc[0]['CV_MAE_std']:.4f}），RMSE {metrics.iloc[0]['CV_RMSE(%)']:.4f}%，"
        f"R² {metrics.iloc[0]['CV_R2']:.4f}",
        f"- 均值基准 CV MAE {baseline_mae:.4f}% → 模型改善 **{gap:+.4f}pp**（R² "
        f"{metrics.iloc[0]['CV_R2']:.4f}，几乎等于直接预测平均值）",
        f"- 覆盖子集目标标准差 {y_cov.std():.3f}%，模型 MAE 约为其 "
        f"{metrics.iloc[0]['CV_MAE(%)']/y_cov.std()*100:.1f}%：**回归对收益幅度的解释力很弱**",
        f"- 预测被压缩：OOF 预测标准差 {np.std(best_oof[covered]):.3f}% 仅为实际 "
        f"{y_cov.std():.3f}% 的 {np.std(best_oof[covered])/y_cov.std()*100:.0f}%",
        f"- **与分类任务对照**（同数据、同模型族、同折）：分类最优 {best_cls['模型']} "
        f"{best_cls['CV准确率(%)']:.2f}% vs 覆盖子集多数类基线 {maj_cov:.2f}%"
        f"（{best_cls['CV准确率(%)']-maj_cov:+.2f}pp）；回归最优 {best_name} 方向命中率 "
        f"{reg_dir:.2f}%（{reg_dir-maj_cov:+.2f}pp）→ 方向尚有微弱信号，幅度基本无信号",
        f"- 口径：TimeSeriesSplit 首折只训练不验证，OOF 覆盖 {n_oof}/{len(y)} 个样本，"
        f"OOF 口径指标只在该子集上统计",
        "",
        "## 四、图表",
        f"测试样本误差曲线（4 面板）见 `{html_path}`；矢量图 `{svg_path}`。",
        f"数据表：`{out/'reg_volume_breakout_metrics.csv'}`、`{out/'reg_volume_breakout_folds.csv'}`、"
        f"`{out/'reg_volume_breakout_learning_curve.csv'}`、`{out/'reg_volume_breakout_oof.csv'}`。",
        "",
        "## 五、保留意见（重要）",
        "- **同区间内评估**：数据、超参、特征选择都在同一段样本上完成，属数据窥探，"
        "真实误差必然高于此处的数字；要下结论需要更长的历史做样本外检验。",
        "- **折边界泄漏**：`TimeSeriesSplit` 按行切分，而同一交易日有多只股票，"
        "边界交易日会同时出现在训练/验证集（轻度信息泄漏）。此点与分类任务口径一致，"
        "若要消除需改为按交易日分折。",
        "- **OOF 只覆盖 5/6 样本**：时序 CV 的首折只作训练集、不产生验证预测，"
        "误差曲线与 OOF 指标均不包含这段样本。",
        "- **MAE 不等于可交易**：误差小于收益标准差并不代表能赚到钱——"
        "策略盈亏取决于方向与幅度分布，而非平均误差。",
        "",
        f"（总耗时 {time.time() - t0:.1f}s）",
    ]
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n输出：\n  {out/'reg_volume_breakout_metrics.csv'}\n  {out/'reg_volume_breakout_oof.csv'}"
          f"\n  {svg_path}\n  {html_path}\n  {rep}")
    print(f"总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
