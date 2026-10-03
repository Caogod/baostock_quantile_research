# 盘后策略选股应用（baostock）

> 基于 [baostock](http://baostock.com) 的 A 股盘后策略选股系统：**无界面、定时运行、策略通过 YAML 注册、结果输出 CSV**，并内置**完整回溯测试引擎**（输出胜率与涨幅）。

**一句话表述**：面向 A 股日频低频研究的盘后选股 + 回溯 + 胜率预测一体化工具链。

### 项目特色
- 🎯 **盘后选股**：盘后对最近交易日全市场扫描，YAML 注册多策略并行打分，输出 CSV。
- 🔁 **完整回溯**：信号 → 模拟持有 N 日 → 卖出，统计胜率 / 平均涨幅 / 累计收益。
- 🧠 **胜率模型**：18 个交易特征 + GBDT 二次过滤，因子重要度可解释。
- 📦 **离线回溯**：成分股落 Parquet，断点续跑，秒级回溯不再联网。
- 🧩 **策略可插拔**：继承基类实现 `signal()` 即可，选股与回溯共用一套策略。
- 🛡️ **工程健壮**：pandas 3.x 兼容补丁、重连重试、请求限速、检查点续跑。

---

## 一、前提调研结论

### 1. 是否可以使用"五价"数据落地方案？ —— ✅ 可行

baostock 日 K 线接口 `query_history_k_data_plus` 提供完整"五价"字段：

| 字段 | 含义 | 字段 | 含义 |
|------|------|------|------|
| open | 开盘价 | close | 收盘价 |
| high | 最高价 | preclose | 昨收（前收）价 |
| low | 最低价 | — | — |

同时附带 `volume`（成交量）、`amount`（成交额）、`turn`（换手率）、`pctChg`（涨跌幅）、`peTTM`/`pbMRQ`（估值）、`isST`（ST 标记）等，覆盖 **1990 年至今**，支持前复权/后复权/不复权（本应用默认前复权，保证除权后价格连续）。

> ⚠️ 名词澄清：若你所说的"五价"指**五档盘口**（买一~买五/卖一~卖五），baostock **不提供**盘口数据（仅日/周/月/分钟 K 线），需换用券商/行情数据源。本应用按"五价 K 线"（开高低收前收）落地。

### 2. 是否可以做完整回溯测试，输出胜率与涨幅？ —— ✅ 可行

基于历史日 K，对每个交易日运行策略信号，命中即模拟"买入 → 持有 N 日 → 卖出"，统计：

- 交易次数 / 盈利次数 / **胜率**
- **平均涨幅 / 中位涨幅 / 最大涨幅 / 最小涨幅**
- 累计收益（复利，见"已知限制"）

输出逐笔明细与策略汇总两份 CSV。实测 20 只样本股、2024-09~2026-09 两年区间可完整跑通（见 `backtest_output/` 示例）。

---

## 二、目录结构

```
├── main.py                  # 盘后选股入口（含 --local / --filter 二次过滤）
├── backtest.py              # 回溯测试入口（多策略 / 多持仓周期 / 多本地库，--local 离线）
├── scheduler.py             # 定时调度（常驻/--once，含 --local/--filter）
├── build_data_library.py    # 构建本地回溯样本库（--index/--days/价格过滤）
├── train_model.py           # 训练胜率预测模型 + 因子重要度 + 留存模型
├── requirements.txt
├── config/
│   ├── settings.yaml        # 全局配置（数据/输出/定时/回溯）
│   └── strategies.yaml      # 策略注册表（YAML）
├── strategies/              # 策略脚本目录（可扩展）
│   ├── base.py              # 策略基类（信号契约）
│   ├── ma_golden_cross.py   # 示例：均线金叉+放量
│   ├── volume_breakout.py   # 示例：放量突破 N 日新高
│   ├── quantile_range.py    # 基础筛选：60日分位区间(25%~50%)
│   ├── quantile_ma.py       # 分位区间 + MA5>MA20
│   ├── quantile_volume.py   # 分位区间 + 近10日量能t检验显著放大
│   ├── ma_breakout.py       # 均线粘合(±3%)+收盘破前一日MA5达5%+换手率>10%
│   ├── ma_breakout_shift.py # ma_breakout 信号延迟5日入场
│   ├── b1_oversold.py       # B1 稳健超跌型（同花顺公式移植）：RSI6<20+振幅>8%+连跌2日
│   │                        #   +近20日无涨停+当日全市场下跌+非一字板/未封板
│   └── ma_cross_breakout.py # 单次金叉破MA60 + 突破前低波动(3σ布林带内 或 极差/最小值<阈值)
│                            #   + N日涨>阈值 + 换手率>MA10×倍数；波动窗口可启用 MA5 连降修剪
│                            #   （全部数值见 config/strategies.yaml，按序执行：涨幅/量比 → 金叉 → 波动）
│                            #   内置 analyze_volatility_distribution() 做阈值横截面分位数统计
├── core/
│   ├── compat.py            # pandas 3.x 兼容补丁
│   ├── data_fetcher.py      # baostock 数据封装（重连重试/限速）
│   ├── local_library.py     # 本地样本库读取器（离线回溯）
│   ├── strategy_loader.py   # YAML 策略动态加载
│   ├── selector.py          # 选股引擎（断点续跑，按股票池隔离）
│   ├── backtester.py        # 回溯测试引擎
│   ├── features.py          # 交易特征计算（18 特征，训练/预测共用）
│   └── win_model.py         # 胜率模型加载 + 选股结果二次过滤
├── models/                  # 训练后留存的模型（joblib）
├── data_library/            # 本地回溯样本库（Parquet）
├── output/                  # 选股 CSV 输出
└── backtest_output/         # 回溯 CSV 输出
```

### 四大主干流程

| 模块 | 入口/文件 | 说明 |
|------|-----------|------|
| 1. 选股/回溯/策略 | `main.py` / `backtest.py` / `strategies/` / `core/` | baostock 数据 + YAML 策略注册 + 盘后选股 + 回溯 |
| 2. 特征扩充 | `core/features.py` | 18 个交易特征，训练与预测共用 |
| 3. 模型训练 | `train_model.py` | DT/RF/GBDT 训练、因子重要度输出、留存 GBDT 模型 |
| 4. 二次过滤 | `core/win_model.py` + `main.py --filter` | 选股后用留存模型预测胜率并过滤 |

---

## 三、环境安装

```bash
# 1. 创建虚拟环境（Python 3.10+ 均可）
python -m venv .venv

# 2. 激活
#    Windows: .venv\Scripts\activate
#    Linux/macOS: source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

> 依赖：`baostock` `pandas` `pyarrow` `pyyaml` `schedule`。
> 已内置 pandas 2.0+ 兼容补丁（baostock 仍使用已移除的 `DataFrame.append`）。

---

## 四、快速开始

### 盘后选股

```bash
python main.py                         # 对最近交易日选股（联网，全市场）
python main.py --date 2026-09-04       # 指定交易日（联网）
```

**本地选股（推荐，离线秒级）**：用合并样本库（zz500 + hs300 并集）离线选股，无需联网：

```bash
python main.py --local --date 2026-09-30                        # 指定交易日（默认取库内最新）
python main.py --local --strategy ma_cross_breakout             # 仅用指定策略
python main.py --local --date 2026-09-30 --strategy b1_oversold # 超跌策略离线选股
```

- `--library` 默认已指向 `data_library/combined_daily.parquet`（合并库），离线选股时无需显式指定；
  如需换用单个库可 `--library data_library/zz500_daily.parquet`。

结果：`output/selection_<日期>.csv`，列含 `date, code, code_name, strategy, score, reason, close, pctChg, turn, volume, amount`，按策略与打分排序。

### 回溯测试

一个入口覆盖「多策略 / 多持仓周期 / 多数据源」：

```bash
# 单策略、多持仓周期、合并样本库（推荐：一次遍历同时展开 3/5/10 日）
python backtest.py --start 2025-11-12 --end 2026-09-30 --local \
    --library data_library/combined_daily.parquet \
    --strategy ma_cross_breakout --holdings 3,5,10

# 单个持仓周期 / 单个本地库（兼容旧用法）
python backtest.py --start 2025-11-12 --end 2026-09-30 --local --holding 5 \
    --library data_library/combined_daily.parquet

# 联网全市场（baostock），跑全部启用策略
python backtest.py --start 2025-01-01 --end 2025-12-31 --holding 10
```

| 参数 | 说明 |
|------|------|
| `--strategy` | 仅跑指定策略（逗号分隔），缺省跑 YAML 中全部启用策略 |
| `--holdings` | 持仓周期，逗号分隔（如 `3,5,10`），一次遍历全部展开 |
| `--holding` | 单个持仓周期（向后兼容，优先于 `--holdings`） |
| `--libraries` | 多个本地样本库（逗号分隔）；`--library` 为单个兼容写法 |
| `--no-prefilter` | 关闭向量化预筛，逐日全量计算（用于等价性校验） |

结果（`backtest_output/`）：

- `backtest_detail_<起>_<止>.csv`：逐笔明细，含 `return_pct` / `bench_return_pct` / `excess_return_pct`
- `backtest_summary_<起>_<止>.csv`：策略 × 数据源 × 持仓周期 汇总（胜率 / 涨幅 / Sharpe / Sortino / 最大回撤）
- `backtest_combined_<起>_<止>.csv`：策略 × 持仓周期（跨数据源合并）
- `backtest_signals_<起>_<止>.csv`：信号日志（含 `score` / `reason`）

> **性能**：策略可选实现 `prefilter_mask(close)` 声明"必要条件候选日"，引擎只对候选日调用 `signal()`。
> `ma_cross_breakout` 已实现（必要条件"近 N 日涨幅 > 阈值"），788 只股票 × 3/5/10 日持仓实测：
> **12.5 秒 vs 全量扫描 33 秒**，且信号集、逐笔收益、reason 文本与全量扫描**完全一致**。

> **交易冷却期**：同一股票同一策略入场后冷却 `holding` 个交易日（= 持有期），平仓前不再重复入场，
> 避免同一标的在持有期内被反复计入。冷却期按持仓周期独立维护。

> **净值类指标口径**：`cum_return_pct` / `max_drawdown_pct` / `sharpe` / `sortino` 基于
> **按日历日去重叠的等权组合净值曲线**计算——每笔交易收益按几何方式摊到 `[entry, exit]` 区间，
> 同日多笔等权合并，再连乘得逐日净值。这与旧的"重叠交易直接连乘"不同，不会因信号重叠而被重复复利放大。
> 注意预筛提速幅度取决于该必要条件的选择性：`surge_pct` 越小（候选日越多），提速越不明显。

### 波动窗口的 MA5 连降修剪（ma_cross_breakout）

条件按"廉价且筛选力强"排序执行（顺序不影响结果，只影响性能）：
**① 涨幅 + 量比 → ② 单次金叉 → ③ 低波动**。

`trim_decline: true` 时，在判定低波动**之前**，从"突破前窗口"**头部**检查 MA5 是否连续下降；
若有连降，则把这段下降区间剔除，**从停止下降的时刻开始**计算波动率（窗口尾部不变，仍截止 T−surge_days）。
修剪后窗口不足 `min_vol_days` 日则直接不出信号。

实测（zz500+hs300，2025-11-12 ~ 2026-09-30）：

| | 信号数 | 3日胜率 | 5日胜率 | 10日胜率 |
|---|---|---|---|---|
| `trim_decline: false` | 210 | 43.1% | 45.8% | 35.2% |
| `trim_decline: true` | 213 | 43.0% | 45.6% | 34.7% |

- 48.8%（104/213）的信号实际触发了修剪（连降 2~26 日）；但**只翻转了 3 个信号**的判定，其余仅窗口口径变化而结论不变。
- 3 个新增信号全部走"极差/最小值"支路——修剪掉头部下跌段后 (max−min)/min 从 >26% 降到 <26%。
- **注意**：修剪只会**放宽**布林带分支（检查的点变少），因此它只可能新增信号、不会剔除信号。
- 对比明细：`backtest_output/backtest_trim_effect_*.csv`。


### 需要外部数据的策略（市场指数 / 证券元数据）

`signal(df)` 只接收单只股票日 K。若策略需要"自身 K 线之外"的信息（如当日全市场涨跌、
上市日期、板块归属），使用上下文机制：

```python
class Strategy(BaseStrategy):
    def bind_context(self, ctx):                 # 入口开跑前调用一次
        self._market_down = ctx["market_down"]   # Series[bool]，index=交易日字符串
        self._meta = ctx["meta"]                 # DataFrame，index=code：name/ipoDate/type/status
```

上下文由 `core/market_context.build_context()` 生成，指数行情与证券元数据缓存在
`data_library/_ctx/`（首次联网拉取，之后离线复用）。CLI：`--market-index`（默认 399317
国证A指）、`--refresh-ctx`（强制刷新缓存）。

代表策略 **`b1_oversold`**（同花顺「B1 稳健超跌型」移植）：要求当日全市场下跌 +
RSI6<20 + 当日振幅>8% + 连跌 2 日 + 近 20 日无涨停 + 非一字板/未封板。

> ⚠️ **读结果时先看信号的时间分布**：这类"恐慌买入"策略的信号高度聚集在少数暴跌日
> （实测 102 个信号仅落在 17 个交易日，84% 集中在 2026-07 那轮暴跌），
> 胜率本质上是少数几次市场事件的结果，不能当作独立样本看待。

### 胜率预测模型（训练 + 二次过滤）

三步流程，特征口径由 `core/features.py` 统一：

```bash
# 1. 生成回溯明细（训练数据来源）
python backtest.py --start 2025-11-12 --end 2026-09-04 --holding 10 --local

# 2. 训练模型 + 因子重要度输出 + 留存模型（默认 ma_breakout 家族子集）
python train_model.py --strategies ma_breakout,ma_breakout_shift5

# 3. 盘后选股后用模型二次过滤（保留胜率 >= 阈值的）
python main.py --local --date 2026-09-04 --strategy ma_breakout_shift5 --filter \
    --filter-threshold 0.5
```

- 模型留存于 `models/win_model.joblib`（GBDT），因子重要度输出 `output/feature_importance.csv`。
- 过滤结果：`output/selection_<日期>_filtered.csv`（含 `win_prob` 列）。
- **经验**：模型应与前置策略筛选出的子集同分布训练（`--strategies` 指定子集），不要用全策略混合数据。

### 本地回溯样本库（离线回溯）

把指定指数成分股最近 N 个交易日数据落地为 Parquet（pandas 快速列式格式），可加股价过滤：

```bash
python build_data_library.py                          # 默认：更新 zz500 + hs300 两库并合并
python build_data_library.py --index hs300            # 仅更新 hs300（不合并）
python build_data_library.py --indexes zz500,hs300    # 更新多个指数并合并
python build_data_library.py --index zz500 --days 200 --price-min 2 --price-max 300
```

- `--index`：只更新单个指数 hs300 / zz500 / sz50（不合并）
- `--indexes`：更新多个指数（逗号分隔），完成后自动合并
- 不带 `--index/--indexes` 时，默认更新 zz500 + hs300 两库并合并为统一库
- `--price-min/--price-max`：股价过滤范围（0 表示不限），默认 2~300 元
- `--no-combine`：多指数/默认流程构建完成后不自动合并

产物在 `data_library/`：`<index>_daily.parquet`（长表）、`<index>_stocks.parquet`（成分股）、`<index>_meta.json`（构建信息）。

**合并库**：把 zz500 与 hs300 合并为一个统一库（交集以 zz500 优先）。默认流程（不带参数）会在更新两库后自动合并；也可手动触发：

```bash
python build_data_library.py --combine
```

合并产物：`combined_daily.parquet`（并集去重长表）、`combined_stocks.parquet`、`combined_meta.json`。

之后回溯**离线、秒级**完成，不再联网：

```bash
# 合并库、单周期
python backtest.py --start 2025-11-12 --end 2026-09-04 --holding 5 --local \
    --library data_library/combined_daily.parquet

# 合并库、多周期：一次统计 zz500 + hs300 并集的 3/5/10 日表现
python backtest.py --start 2025-11-12 --end 2026-09-30 --local \
    --library data_library/combined_daily.parquet \
    --strategy ma_cross_breakout --holdings 3,5,10
```

> 说明：
> - baostock 仅顶层暴露 `query_hs300_stocks` / `query_zz500_stocks` / `query_sz50_stocks`，无中证1000成分股接口。
> - 样本库是固定窗口快照（默认 200 个交易日），回溯区间须落在库内日期范围；需更长区间请用更大 `--days` 重建。
> - 股价过滤以"最新收盘价"判断，属样本库层面的静态过滤（回溯时按信号日实时过滤可另加）。
> - 构建过程含断点续跑（按 index+价格区间隔离），中途中断重跑会自动跳过已下载股票。

### 定时运行

```bash
python scheduler.py           # 常驻，按 settings.yaml 中 schedule.time 每日运行
python scheduler.py --once    # 立即执行一次后退出（可接入 cron/Windows 任务计划）
python scheduler.py --once --local --filter            # 离线选股 + 胜率模型二次过滤
```

> 无界面常驻进程；生产环境建议用 `scheduler.py --once` 配合系统级定时任务（Linux cron / Windows 任务计划程序），或 `systemd` / `nssm` 托管常驻进程。
>
> 定时流程同样支持选股后二次过滤：加 `--filter` 即可（配合 `--local` 用本地样本库离线运行最稳妥；`--library` 默认已指向合并库 `combined_daily.parquet`）。

---

## 五、策略注册（YAML）与自定义

策略在 `config/strategies.yaml` 中注册：

```yaml
strategies:
  - name: ma_golden_cross          # 输出中的策略标识
    enabled: true                  # 是否启用
    script: strategies/ma_golden_cross.py   # 脚本相对路径
    params:                        # 策略参数（透传给构造器）
      short: 5
      long: 20
      vol_ratio: 1.2
```

**新增策略**：在 `strategies/` 下新建脚本，定义继承 `base.BaseStrategy` 的 `Strategy` 类，实现 `signal(df) -> dict`：

```python
from .base import BaseStrategy

class Strategy(BaseStrategy):
    name = "my_strategy"
    description = "我的策略"

    def signal(self, df):
        # df: 升序日 K，含 date/open/high/low/close/preclose/volume/amount/pctChg/turn
        if 满足条件:
            return {"buy": True, "score": 1.0, "reason": "说明"}
        return self._no_signal()
```

然后在 YAML 中登记即可，无需改动引擎代码。**选股与回溯共用同一套策略**。

---

## 六、配置说明（config/settings.yaml）

| 段 | 键 | 说明 |
|----|----|----|
| data | adjustflag | 复权：1 后复权 / 2 前复权 / 3 不复权 |
| data | history_days | 每只股票拉取的历史交易日数 |
| data | code_prefixes | 股票池代码前缀（沪主板 sh.60、科创 sh.68、深主板 sz.00、创业板 sz.30） |
| data | exclude_st | 剔除 ST/退市 |
| data | price_min | 收盘价下限（<=0 不限），低于此价不研究 |
| data | price_max | 收盘价上限（<=0 不限），高于此价不研究 |
| data | explicit_codes | 显式股票列表（非空则忽略全市场扫描，便于调试/限定范围） |
| data | max_stocks | 0=全市场；>0 仅取前 N 只（调试） |
| data | request_interval | 请求间隔（秒），降低免费数据源压力 |
| data | checkpoint_interval | 每处理 N 只落盘一次检查点 |
| data | resume | 断点续跑开关 |
| schedule | time | 每日运行时间（24 小时制） |
| schedule | trading_day_only | 仅交易日运行 |
| backtest | holding_days | 持仓周期（交易日） |
| backtest | lookback_days | 信号预热所需前置历史天数 |
| backtest | commission | 单边交易费率（净收益计算） |

---

## 七、健壮性设计

- **pandas 3.x 兼容**：`core/compat.py` 为 baostock 补上已移除的 `DataFrame.append`。
- **重连重试**：数据获取对空结果/异常自动重连重试（默认 3 次）。
- **断点续跑**：每 N 只股票落盘检查点；若因 baostock 底层异常导致进程退出，重跑自动跳过已处理股票继续，不重复劳动。
- **请求限速**：请求间隔可配，避免高频压垮免费数据源。

---

## 八、已知限制

1. **baostock 免费服务的脆弱性**：高频连续请求（约百次量级）后偶发服务端连接重置，甚至进程级异常（无法在 Python 内捕获）。本项目以"断点续跑 + 重连重试 + 限速"缓解；全市场扫描（约 5000 只）建议分时段或调大 `request_interval`。该现象在开发沙箱中可能因网络策略被放大，本地直连通常更稳。
2. **回溯的累计收益为"去重叠等权组合"口径**：`cum_return_pct` / `max_drawdown_pct` / `sharpe` / `sortino`
   基于按日历日摊薄、同日等权合并的组合净值曲线计算，已消除旧版"重叠交易直接连乘"导致的失真；
   `win_rate` 与 `avg/median_return` 仍为逐笔信号口径。
3. **幸存者偏差**：全市场回溯以回测期末在市的股票为样本，已退市股票未纳入。
4. **数据为日级**：不适用于盘中/高频策略，信号基于收盘数据。
5. 示例策略仅作演示，不构成投资建议。

---

## 九、免责声明

本项目仅用于量化研究与学习，输出不构成任何投资建议。股市有风险，入市需谨慎。
