"""胜率预测模型训练 + 因子重要度输出 + 模型留存（模块3）。

从本地样本库 + 回溯逐笔明细构建特征数据集，按**策略分组**各训练模型，
留存为 `models/win_model_<strategy>.joblib`（+ 同名 `.json` 特征列表），供选股后
按策略二次过滤。另输出各策略 CV 准确率汇总与因子重要度。

候选模型（TimeSeriesSplit 横向对比，任选其一留存）：
    决策树(DT) / 随机森林(RF) / GBDT / XGBoost / CatBoost / LightGBM
    --model-type 指定留存哪类（默认 best = 每个策略自动选 CV 准确率最高者）；
    --save-all-models 可一次留存全部类型。

CV 口径（已修正）：
- 使用 TimeSeriesSplit 进行时序交叉验证，杜绝随机 KFold 的时序泄漏
  （"未来样本"进入训练集预测"过去样本"导致准确率虚高）。
- 数据按 entry_date 升序排列后分折。

极短期窗口训练（捕捉市场偏好变化）：
- --recent-days N：仅用最近 N 个交易日的回测明细训练。
  市场风格/偏好会随时间漂移（如大盘 vs 小盘、价值 vs 成长、
  量价关系强弱），用近期数据训练可让模型感知"当前市场偏好什么"，
  而非全历史的平均规律。适合短周期（5~10 日持仓）策略。
- 默认 0 = 全样本训练（与历史行为一致）。

用法：
    python train_model.py                                   # 明细中全部策略，各留存 CV 最优模型
    python train_model.py --model-type xgboost              # 统一留存 XGBoost
    python train_model.py --model-type gbdt --save-all-models
    python train_model.py --strategies ma_breakout,ma_breakout_shift5
    python train_model.py --recent-days 60                  # 仅用近 60 交易日训练
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.features import FEATURES, precompute_features
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DETAIL = "backtest_output/backtest_detail_2025-11-12_2026-09-30.csv"
LIBRARY = "data_library/combined_daily.parquet"
MODEL_DIR = PROJECT_ROOT / "models"


def build_models() -> dict[str, object]:
    """构建候选分类器集合（含决策树/随机森林/GBDT + XGBoost/CatBoost/LightGBM）。

    返回 {中文名: 模型实例}。GBDT/XGB/LGBM/CatBoost 均为梯度提升族，
    供 TimeSeriesSplit CV 横向对比，选定最优者留存。
    """
    return {
        "决策树(DT)": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0),
        "随机森林(RF)": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=10, n_jobs=-1, random_state=0
        ),
        "GBDT": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=0,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4,
            random_seed=0, verbose=0, allow_writing_files=False,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=-1,
        ),
    }


# --model-type 可选值 → build_models() 的键名映射
MODEL_TYPE_KEYS = {
    "dt": "决策树(DT)",
    "rf": "随机森林(RF)",
    "gbdt": "GBDT",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
}


def rfe_select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    tscv,
    bottom_n: int = 3,
    patience: int = 3,
) -> tuple[list[str], list[dict]]:
    """递归特征消除（Bottom-N 剔除）选取最优特征子集。

    每轮训练后剔除最不重要的 bottom_n 个特征（弱特征），用剩余特征重训，
    直到 CV 连续 patience 轮不再提升或剩余特征不足以继续。返回
    (最优特征子集, 每轮记录列表)。CV 用 XGBoost（稳定的梯度提升模型）评估。

    实验验证（2026-10-03）：b1_oversold 70.29%→80.00%（+9.71pp）、
    ma_breakout_shift5 63.06%→66.94%（+3.88pp），平均 +6.79pp，有效。
    """
    current = list(feature_names)
    best_feats = list(feature_names)
    best_acc = -1.0
    no_improve = 0
    rounds: list[dict] = []

    while len(current) > bottom_n:
        Xc = X[:, [feature_names.index(f) for f in current]]
        m = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=0,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
        )
        acc = float(cross_val_score(m, Xc, y, cv=tscv, scoring="accuracy").mean())
        rounds.append({"n_features": len(current), "cv_acc": acc, "features": list(current)})

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_feats = list(current)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

        # 剔除最不重要的 bottom_n 个特征
        m.fit(Xc, y)
        imp = getattr(m, "feature_importances_", None)
        if imp is None:
            break
        bottom = [current[i] for i in np.argsort(imp)[:bottom_n]]
        current = [f for f in current if f not in set(bottom)]

    return best_feats, rounds


def main() -> int:
    parser = argparse.ArgumentParser(description="训练胜率预测模型并留存")
    parser.add_argument("--strategies", default="",
                        help="训练子集（逗号分隔策略名，空串=明细中全部策略）")
    parser.add_argument("--detail", default=DETAIL, help="回溯逐笔明细 CSV")
    parser.add_argument("--library", default=LIBRARY, help="本地样本库 Parquet")
    parser.add_argument("--recent-days", type=int, default=0,
                        help="仅用最近 N 个交易日的明细训练（0=全样本）。"
                             "极短期窗口可捕捉市场偏好/风格漂移，适合短周期策略。")
    parser.add_argument("--model-type", default="best",
                        choices=["best"] + list(MODEL_TYPE_KEYS),
                        help="留存哪种模型为 win_model_<strategy>.joblib。"
                             "best=每个策略自动选 CV 准确率最高者（默认）；"
                             "也可指定 dt/rf/gbdt/xgboost/catboost/lightgbm 统一留存该类")
    parser.add_argument("--save-all-models", action="store_true",
                        help="同时留存全部 6 类模型（win_model_<strategy>_<type>.joblib）")
    parser.add_argument("--rfe", action="store_true", default=True,
                        help="启用递归特征消除（Bottom-N 剔除弱特征，默认开启）")
    parser.add_argument("--no-rfe", action="store_false", dest="rfe",
                        help="关闭 RFE，直接用全 24 特征训练")
    parser.add_argument("--bottom-n", type=int, default=3,
                        help="RFE 每轮剔除的最不重要特征数（默认 3）")
    parser.add_argument("--patience", type=int, default=3,
                        help="RFE 中 CV 连续不提升多少轮后停止（默认 3）")
    args = parser.parse_args()

    lib = pd.read_parquet(args.library).sort_values("date")
    detail = pd.read_csv(args.detail)

    # 子集过滤（经验：模型应与前置策略筛选出的子集同分布）
    wanted = {x.strip() for x in args.strategies.split(",") if x.strip()}
    if wanted:
        detail = detail[detail["strategy"].isin(wanted)].copy()
    print(f"训练子集交易数: {len(detail)}（策略 {sorted(wanted) or '全部'}）")

    # 极短期窗口训练：仅保留最近 N 个交易日的明细
    if args.recent_days > 0:
        detail["entry_date"] = detail["entry_date"].astype(str)
        all_dates = sorted(detail["entry_date"].unique())
        if len(all_dates) > args.recent_days:
            cutoff = all_dates[-args.recent_days]
            before = len(detail)
            detail = detail[detail["entry_date"] >= cutoff].copy()
            print(f"极短期窗口: 仅保留近 {args.recent_days} 交易日（{cutoff} 起），"
                  f"{before} -> {len(detail)} 笔")

    # 特征矩阵 + 对齐交易
    feat_all = precompute_features(lib)
    df = detail.merge(feat_all, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")
    # 标签：相对沪深300同期超额收益 > 0（替代原"净收益>0"，消除牛市偏差）
    # 牛市里随机买入大概率>0，模型会学到"牛市特征"而非策略 alpha；
    # 改超额后模型学习"在当前市场环境下，该信号能否跑赢大盘"。
    from core.market import benchmark_returns_for_trades
    bench = benchmark_returns_for_trades(df) * 100.0  # 百分数口径对齐 return_pct
    df["bench_return_pct"] = bench.round(4)
    df["excess_return_pct"] = (df["return_pct"].astype(float) - df["bench_return_pct"]).round(4)
    df["win"] = (df["excess_return_pct"] > 0).astype(int)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"有效样本: {len(df)}，超额胜率 {df['win'].mean()*100:.1f}%"
          f"（基准：净收益胜率 {(df['return_pct']>0).mean()*100:.1f}%）")

    # 时序 CV 要求：按 entry_date 升序排列，避免时序泄漏
    df = df.sort_values("entry_date").reset_index(drop=True)

    MODEL_DIR.mkdir(exist_ok=True)
    (PROJECT_ROOT / "output").mkdir(exist_ok=True)

    # ---- 每个策略单独训练并留存模型 ----
    # 样本阈值：低于此数不单独训练（GBDT 在极少量样本上无意义，且容易过拟合）
    MIN_SAMPLES = 30
    tscv = TimeSeriesSplit(n_splits=5)

    strategy_groups = sorted(df["strategy"].dropna().astype(str).unique())
    if not strategy_groups:
        print("[error] 明细中无 strategy 列或为空，无法按策略训练", file=sys.stderr)
        return 1

    print(f"\n=== 按策略训练（每个策略独立模型，留存类型: {args.model_type}）===")
    print(f"策略数: {len(strategy_groups)}，样本阈值 {MIN_SAMPLES}\n")

    summary_rows: list[dict] = []
    importance_rows: list[dict] = []
    saved: list[str] = []
    saved_types: dict[str, str] = {}  # strategy -> 实际留存的模型中文名

    # 固定留存类型（非 best）时，提前解析中文键；best 时在循环内按 CV 动态选择
    save_key = MODEL_TYPE_KEYS.get(args.model_type) if args.model_type != "best" else None

    for strat in strategy_groups:
        g = df[df["strategy"] == strat]
        n = len(g)
        g = g.sort_values("entry_date").reset_index(drop=True)
        X, y = g[FEATURES].values, g["win"].values
        baseline = max(float(y.mean()), 1 - float(y.mean())) if n else 0.0

        if n < MIN_SAMPLES:
            print(f"[skip] {strat}: 仅 {n} 个样本（< {MIN_SAMPLES}），跳过训练")
            continue

        # ---- 递归特征消除（Bottom-N 剔除弱特征）----
        # 实验验证有效（b1 +9.71pp、shift5 +3.88pp），默认开启。
        active_features = list(FEATURES)
        rfe_log: list[dict] = []
        if args.rfe:
            active_features, rfe_log = rfe_select_features(
                X, y, FEATURES, tscv, bottom_n=args.bottom_n, patience=args.patience
            )
            gain = rfe_log[-1]["cv_acc"] * 100 - rfe_log[0]["cv_acc"] * 100 if rfe_log else 0.0
            print(f"  {strat}: RFE 特征 {len(FEATURES)} -> {len(active_features)}"
                  f"（CV {rfe_log[0]['cv_acc']*100:.2f}% -> {rfe_log[-1]['cv_acc']*100:.2f}%，"
                  f"{gain:+.2f}pp）")
        X_active = X[:, [FEATURES.index(f) for f in active_features]]

        # 候选模型（决策树/随机森林/GBDT + XGBoost/CatBoost/LightGBM）
        models = build_models()
        accs = {}
        imp = {}
        for mname, model in models.items():
            try:
                accs[mname] = cross_val_score(model, X_active, y, cv=tscv, scoring="accuracy").mean()
                model.fit(X_active, y)
                imp[mname] = model.feature_importances_
            except Exception as exc:  # noqa: BLE001
                # 单模型失败不阻断其余模型（例如某些库对极小数量的边界处理）
                print(f"  [warn] {strat} 的 {mname} 训练失败: {exc}", file=sys.stderr)
                accs[mname] = float("nan")

        # 留存主模型（--model-type 指定；best 时自动选 CV 准确率最高者），
        # 统一写为 win_model_<strategy>.joblib，保证二次过滤无需感知具体类型
        chosen_key = save_key
        if chosen_key is None:  # best：选 CV 准确率最高的模型
            valid = {k: v for k, v in accs.items() if not np.isnan(v)}
            if valid:
                chosen_key = max(valid, key=valid.get)
        if chosen_key in models and not np.isnan(accs.get(chosen_key, float("nan"))):
            joblib.dump(models[chosen_key], MODEL_DIR / f"win_model_{strat}.joblib")
            (MODEL_DIR / f"win_model_{strat}.json").write_text(
                json.dumps(active_features, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved.append(strat)
            saved_types[strat] = chosen_key

        # 可选：留存全部模型（win_model_<strategy>_<type>.joblib）
        if args.save_all_models:
            for mname, model in models.items():
                if np.isnan(accs.get(mname, float("nan"))):
                    continue
                short = {v: k for k, v in MODEL_TYPE_KEYS.items()}.get(mname, mname)
                joblib.dump(model, MODEL_DIR / f"win_model_{strat}_{short}.joblib")

        summary_rows.append({
            "strategy": strat,
            "samples": n,
            "n_features": len(active_features),
            "chosen_model": chosen_key if chosen_key in models else "",
            "baseline": round(baseline * 100, 2),
            "dt_acc": round(accs.get("决策树(DT)", float("nan")) * 100, 2),
            "rf_acc": round(accs.get("随机森林(RF)", float("nan")) * 100, 2),
            "gbdt_acc": round(accs.get("GBDT", float("nan")) * 100, 2),
            "xgb_acc": round(accs.get("XGBoost", float("nan")) * 100, 2),
            "cat_acc": round(accs.get("CatBoost", float("nan")) * 100, 2),
            "lgbm_acc": round(accs.get("LightGBM", float("nan")) * 100, 2),
        })
        # 因子重要度按全 FEATURES 对齐（RFE 剔除的特征记 0），保证跨策略列一致
        imp_map = dict(zip(active_features, imp.get("随机森林(RF)", [])))
        importance_rows.append({"strategy": strat, **{
            f: float(imp_map.get(f, 0.0)) for f in FEATURES
        }})
        acc_str = ", ".join(f"{k}={v*100:.2f}%" for k, v in accs.items() if not np.isnan(v))
        print(f"  {strat}: 样本 {n}，CV {acc_str}（基线 {baseline*100:.2f}%）")

    # ---- 汇总输出 ----
    if saved:
        tag = f"，近{args.recent_days}日窗口" if args.recent_days > 0 else "，全样本"
        type_desc = "best（每个策略自动选 CV 最优）" if args.model_type == "best" else args.model_type
        print(f"\n已留存 {len(saved)} 个策略模型（类型 {type_desc}）到 {MODEL_DIR}/{tag}：")
        for s in saved:
            t = saved_types.get(s, "")
            print(f"  - win_model_{s}.joblib   [{t}]")

        # 训练汇总表
        if summary_rows:
            summary = pd.DataFrame(summary_rows)
            summary.to_csv("output/train_summary_by_strategy.csv", index=False, encoding="utf-8-sig")
            print("\n=== 各策略模型 CV 准确率汇总（%）===\n")
            print(summary.to_string(index=False))

        # 因子重要度（每策略一行，列=特征）
        if importance_rows:
            imp_df = pd.DataFrame(importance_rows)
            imp_df.to_csv("output/feature_importance_by_strategy.csv", index=False, encoding="utf-8-sig")
            # 同时按 RF 平均重要度排序列，输出全局视角
            avg_imp = imp_df.drop(columns=["strategy"]).mean().sort_values(ascending=False)
            rank = pd.DataFrame({"feature": avg_imp.index, "avg_rf_importance": avg_imp.values})
            rank.to_csv("output/feature_importance.csv", index=False, encoding="utf-8-sig")
            print("\n=== 因子重要度（跨策略 RF 平均，降序）===\n")
            print(rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print(f"\n[error] 没有任何策略满足最小样本 {MIN_SAMPLES}，未留存模型", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
