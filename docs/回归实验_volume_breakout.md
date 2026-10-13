# 回归任务实验：volume_breakout 数据集

> 生成于 2026-10-13 23:19；脚本 `experiments/exp_reg_volume_breakout.py`；图表报告 `docs\回归实验_volume_breakout.html`。

## 一、实验目的
分类任务把连续收益二值化成 `win = 超额收益 > 0`，丢掉了幅度信息。本实验保持**完全相同的模型策略**（同样 6 个模型族 + 镜像超参 + TimeSeriesSplit(5) + Bottom-N RFE + 按 CV 指标选最优），只把任务换成**回归**（直接预测连续超额收益率），并可视化测试样本误差曲线。

## 二、数据集
- 明细：`backtest_detail_2025-11-12_2026-10-13.csv`（策略 ['volume_breakout']，持仓 5 日）
- 样本：2519 笔，765 只股票，2026-02-05 ~ 2026-09-29
- 特征：24 个（`core/features.py` 的口径，与分类任务一致），RFE 后保留 15 个
- 目标 `excess_return_pct`：均值 -0.703%、中位 -1.531%、标准差 8.875%、超额胜率 41.0%

## 三、结论
- **最优模型 随机森林(RF)**：CV MAE **6.0338%**（±0.9547），RMSE 8.2468%，R² 0.0619
- 均值基准 CV MAE 6.3502% → 模型改善 **+0.3164pp**（R² 0.0619，几乎等于直接预测平均值）
- 覆盖子集目标标准差 8.802%，模型 MAE 约为其 68.5%：**回归对收益幅度的解释力很弱**
- 预测被压缩：OOF 预测标准差 3.239% 仅为实际 8.802% 的 37%
- **与分类任务对照**（同数据、同模型族、同折）：分类最优 随机森林(RF) 61.53% vs 覆盖子集多数类基线 59.76%（+1.77pp）；回归最优 随机森林(RF) 方向命中率 59.90%（+0.14pp）→ 方向尚有微弱信号，幅度基本无信号
- 口径：TimeSeriesSplit 首折只训练不验证，OOF 覆盖 2095/2519 个样本，OOF 口径指标只在该子集上统计

## 四、图表
测试样本误差曲线（4 面板）见 `docs\回归实验_volume_breakout.html`；矢量图 `research_output\reg_volume_breakout_error.svg`。
数据表：`research_output\reg_volume_breakout_metrics.csv`、`research_output\reg_volume_breakout_folds.csv`、`research_output\reg_volume_breakout_learning_curve.csv`、`research_output\reg_volume_breakout_oof.csv`。

## 五、保留意见（重要）
- **同区间内评估**：数据、超参、特征选择都在同一段样本上完成，属数据窥探，真实误差必然高于此处的数字；要下结论需要更长的历史做样本外检验。
- **折边界泄漏**：`TimeSeriesSplit` 按行切分，而同一交易日有多只股票，边界交易日会同时出现在训练/验证集（轻度信息泄漏）。此点与分类任务口径一致，若要消除需改为按交易日分折。
- **OOF 只覆盖 5/6 样本**：时序 CV 的首折只作训练集、不产生验证预测，误差曲线与 OOF 指标均不包含这段样本。
- **MAE 不等于可交易**：误差小于收益标准差并不代表能赚到钱——策略盈亏取决于方向与幅度分布，而非平均误差。

（总耗时 66.7s）
