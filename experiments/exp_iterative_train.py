"""实验B：递归特征消除（RFE）的独立测试 —— Bottom-N 剔除 + CV 不再提升终止。

流程（对每个策略、指定模型类型）：
  1. 第 0 轮：全特征训练，得到 CV 与特征重要性。
  2. 剔除**最不重要的 Bottom-N 特征**（弱特征），用剩余特征训练下一轮。
  3. 每轮继续剔除 Bottom-N，直到 CV 不再提升（连续 patience 轮无改善）或特征耗尽。
  4. 记录每轮 CV 与特征子集，取「最优轮次」作为 RFE 结果。

判定（RFE 正确口径）：
  - 核心：最优轮次 CV 是否显著高于首轮（全特征）CV。
  - 参考：所有轮次模型等权集成的 OOF 准确率（通常混入早期弱模型反而有害）。

数据与持仓周期口径（与主线 train_model.py 一致，见 experiments/exp_common.py）：
  - 明细/样本库路径不再硬编码：--detail/--library > 环境变量
    SQ_DETAIL_PATH/SQ_LIBRARY_PATH > 自动发现最完整的明细；
  - --holding auto（默认）按各策略 config 中声明的 holding 拆分，避免 3/5/10
    三种持有期的样本混在一起（混用会让 CV 与"该持有期的胜率"不对应）。

用法：
    python experiments/exp_iterative_train.py [--model-type xgboost] [--bottom-n 3] [--patience 3]
    python experiments/exp_iterative_train.py --holding 5
    python experiments/exp_iterative_train.py --detail backtest_output/backtest_detail_xxx.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_common import (  # noqa: E402
    describe_dataset,
    enabled_strategies,
    load_dataset,
    resolve_holdings,
)
from core.features import FEATURES  # noqa: E402
from train_model import build_models, MODEL_TYPE_KEYS  # noqa: E402

MIN_SAMPLES = 30
BOTTOM_N = 3
MAX_ROUNDS = 20
PATIENCE = 3


def bottom_k_features(model, features, k=BOTTOM_N):
    """按 feature_importances_ 提取最不重要的 k 个特征名（升序）。"""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return list(features)
    order = np.argsort(imp)[:k]  # 升序 → 最不重要者在前
    return [features[i] for i in order]


def run_iterative(X_all, y, feature_names, model_factory, tscv, bottom_n=BOTTOM_N, patience=PATIENCE):
    """迭代训练（RFE），返回 (轮次记录列表, 各轮模型列表, 各轮特征列表)。

    迭代语义：
      - 每轮训练后剔除 Bottom-N（最不重要）特征，用剩余特征训练下一轮。
      - 保留每一轮的模型与特征子集，供最终集成。

    停止条件（满足任一即停）：
      1. CV 不再提升：连续 `patience` 轮未创历史最优（early stopping）
      2. 剩余特征不足以继续剔除（< bottom_n）
    """
    rounds = []
    models = []
    feat_lists = []
    current_features = list(feature_names)
    best_acc = -1.0
    no_improve = 0

    for r in range(MAX_ROUNDS):
        X = X_all[:, [feature_names.index(f) for f in current_features]]
        model = model_factory()
        acc = float(cross_val_score(model, X, y, cv=tscv, scoring="accuracy").mean())
        model.fit(X, y)  # 全量拟合，供集成预测

        improved = acc > best_acc + 1e-6
        if improved:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        rounds.append({
            "round": r, "n_features": len(current_features),
            "cv_acc": acc, "best_acc": best_acc, "features": list(current_features),
        })
        models.append(model)
        feat_lists.append(list(current_features))
        print(f"    轮次 {r}: 特征数 {len(current_features)}，CV {acc*100:.2f}%"
              f"（历史最优 {best_acc*100:.2f}%）")

        # 停止条件 1：CV 不再提升
        if no_improve >= patience:
            print(f"    -> 连续 {patience} 轮未提升，停止迭代")
            break

        # 剔除 Bottom-N 弱特征
        bottom = bottom_k_features(model, current_features, k=bottom_n)
        next_features = [f for f in current_features if f not in set(bottom)]

        # 停止条件 2：剩余特征不足以继续
        if len(next_features) < bottom_n:
            print(f"    -> 剩余特征 {len(next_features)} 个，不足以继续剔除，停止迭代")
            break

        print(f"    -> 剔除 Bottom-{bottom_n}: {bottom}")
        current_features = next_features

    return rounds, models, feat_lists


def ensemble_predict(models, feat_lists, feature_names, X_all):
    """各轮模型（各自特征）预测概率的算术平均。"""
    probs = []
    for model, fl in zip(models, feat_lists):
        X = X_all[:, [feature_names.index(f) for f in fl]]
        p = model.predict_proba(X)[:, 1]
        probs.append(p)
    return np.mean(probs, axis=1)


def ensemble_cv_acc(models, feat_lists, feature_names, X_all, y, tscv):
    """用 TimeSeriesSplit 评估集成的准确率（OOF，无泄漏）。

    对每一折：训练折拟合各轮模型副本，验证折做概率平均。
    各轮模型用各自的特征子集；对每折、每轮模型独立拟合后对验证折预测概率，
    最终对同一验证样本的概率做算术平均 → 得到集成 OOF 概率 → 算准确率。
    """
    n = len(y)
    oof_probs = np.zeros(n)
    for tr_idx, va_idx in tscv.split(X_all):
        fold_probs = []
        for model, fl in zip(models, feat_lists):
            Xf = X_all[:, [feature_names.index(f) for f in fl]]
            m = type(model)(**model.get_params())
            m.fit(Xf[tr_idx], y[tr_idx])
            fold_probs.append(m.predict_proba(Xf[va_idx])[:, 1])
        # stack: (n_models, n_va) -> 对模型维度平均 -> (n_va,)
        oof_probs[va_idx] = np.mean(np.vstack(fold_probs), axis=0)
    acc = float((oof_probs >= 0.5).astype(int).mean())
    return acc, oof_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="xgboost", choices=list(MODEL_TYPE_KEYS))
    ap.add_argument("--strategies", default="", help="逗号分隔，空=全部")
    ap.add_argument("--detail", default="", help="回溯明细 CSV（空=自动解析最完整的一份）")
    ap.add_argument("--library", default="", help="本地样本库 Parquet（空=默认合并库）")
    ap.add_argument("--holding", default="auto",
                    help="持仓周期口径：auto=按各策略 config 声明的 holding（默认）；"
                         "或整数（如 5）统一指定，避免 3/5/10 混在一起")
    ap.add_argument("--bottom-n", type=int, default=BOTTOM_N, help="每轮剔除的最不重要特征数")
    ap.add_argument("--patience", type=int, default=PATIENCE, help="CV 连续不提升多少轮后停止")
    args = ap.parse_args()

    df = load_dataset(detail_path=args.detail or None,
                      library_path=args.library or None,
                      strategies=args.strategies or None,
                      required=sorted({s.strip() for s in args.strategies.split(",") if s.strip()})
                      or enabled_strategies())
    tscv = TimeSeriesSplit(n_splits=5)
    model_factory = lambda: build_models()[MODEL_TYPE_KEYS[args.model_type]]  # noqa: E731

    # 持仓周期拆分（避免同一策略的 3/5/10 日样本混在一起训练）
    holdings = resolve_holdings(df, args.holding, verbose=True)

    print(f"实验B：递归特征消除（Bottom-{args.bottom_n} 剔除，模型={args.model_type}，"
          f"patience={args.patience}）")
    print(f"数据集: {describe_dataset(df)}")
    if holdings:
        print("持仓周期: " + ", ".join(f"{k}={v}" for k, v in sorted(holdings.items())))
    print()

    summary_rows = []
    for strat, g in df.groupby("strategy"):
        want_h = holdings.get(str(strat))
        if holdings and want_h is None:
            print(f"[skip] {strat}: 无法确定持仓周期（holding_days 全缺失），跳过")
            continue
        if want_h is not None:
            g_h = g[g["holding_days"] == want_h]
            if g_h.empty:
                print(f"[skip] {strat}: 无 holding_days={want_h} 的样本")
                continue
            g = g_h
        n = len(g)
        if n < MIN_SAMPLES:
            print(f"[skip] {strat}: 样本 {n} < {MIN_SAMPLES}")
            continue
        g = g.dropna(subset=FEATURES)
        X_all = g[FEATURES].values
        y = g["win"].values
        baseline = max(float(y.mean()), 1 - float(y.mean()))
        h_used = int(g["holding_days"].iloc[0]) if "holding_days" in g.columns and pd.notna(
            g["holding_days"].iloc[0]) else None

        h_desc = f"持仓 {h_used} 日，" if h_used is not None else ""
        print(f"\n=== {strat}（{h_desc}样本 {len(g)}，baseline {baseline*100:.2f}%）===")
        rounds, models, feat_lists = run_iterative(
            X_all, y, FEATURES, model_factory, tscv, bottom_n=args.bottom_n, patience=args.patience
        )

        first_acc = rounds[0]["cv_acc"]
        last_acc = rounds[-1]["cv_acc"]
        best_acc = max(r["cv_acc"] for r in rounds)

        # 集成 OOF 准确率
        ens_acc, _ = ensemble_cv_acc(models, feat_lists, FEATURES, X_all, y, tscv)

        print(f"    首轮 CV {first_acc*100:.2f}% | 末轮 CV {last_acc*100:.2f}% | "
              f"最优 CV {best_acc*100:.2f}% | 集成 OOF {ens_acc*100:.2f}%")

        summary_rows.append({
            "strategy": strat,
            "holding_days": h_used,
            "samples": len(g),
            "baseline": round(baseline * 100, 2),
            "first_round_acc": round(first_acc * 100, 2),
            "best_round_acc": round(best_acc * 100, 2),
            "last_round_acc": round(last_acc * 100, 2),
            "n_rounds": len(rounds),
            "ensemble_acc": round(ens_acc * 100, 2),
            "final_n_features": len(rounds[-1]["features"]),
        })

    if summary_rows:
        out = PROJECT_ROOT / "output/exp_iterative_train_result.csv"
        out.parent.mkdir(exist_ok=True)
        res = pd.DataFrame(summary_rows)
        res.to_csv(out, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 100)
        print("实验B 汇总（Bottom-N 剔除 RFE）")
        print("=" * 100)
        print(res.to_string(index=False))
        print(f"\n结果已存: {out}")

        # 判定：以「最优轮次 CV vs 首轮 CV」为 RFE 是否有效的核心依据。
        # 集成 OOF 仅供参考（把所有轮次模型等权平均会混入早期弱模型，通常反而有害）。
        for _, row in res.iterrows():
            gain_best_vs_first = row["best_round_acc"] - row["first_round_acc"]
            print(f"{row['strategy']}: 最优轮较首轮 {gain_best_vs_first:+.2f}pp"
                  f"（集成 OOF {row['ensemble_acc']:.2f}%，仅参考）")
        avg_best_gain = (res["best_round_acc"] - res["first_round_acc"]).mean()
        if avg_best_gain > 0.5:
            print(f"\n=> 实验B 有效：Bottom-N 剔除使最优轮平均提升 {avg_best_gain:+.2f}pp，建议集成。")
        else:
            print(f"\n=> 实验B 无效：Bottom-N 剔除最优轮平均 {avg_best_gain:+.2f}pp，建议终止。")


if __name__ == "__main__":
    main()
