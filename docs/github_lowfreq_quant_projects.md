# GitHub 低频量化开源项目探索

> 围绕 A 股低频（日频/周频）量化研究场景，对 GitHub 上主流开源项目进行调研与选型梳理。
> 数据基于 2026 年公开 Star 数与维护状态，星数为近似值，会随时间变动。

---

## 一、核心框架（回测 / 实盘 / 研究）

### 1. microsoft/qlib ⭐ 44k+
- **定位**：微软亚洲研究院出品的 AI 量化投资平台，强调因子挖掘 + 机器学习管线。
- **核心能力**：
  - 内置 **Alpha158 / Alpha360** 因子库（158/360 个技术因子，开箱即用）。
  - 20+ SOTA 模型（LightGBM / LSTM / Transformer / GRU / GATs / TCN）。
  - RD-Agent：LLM 驱动的自动因子挖掘与研报因子提取。
  - MLflow 实验管理、组合优化、TWAP/VAP 执行算法。
- **适用**：ML 因子挖掘、横截面选股、学术级量化研究。
- **局限**：实盘接入弱、上手门槛偏高（需 ML + 量化双背景）。
- **与本项目关系**：可借鉴其因子库设计与横截面选股架构，本项目 `core/features.py` 的 18 特征思路与之同源。

### 2. vnpy/veighna (vnpy) ⭐ 41k+
- **定位**：国内实盘量化交易的全栈框架，事件驱动、模块化。
- **核心能力**：
  - 20+ 交易所接口（CTP 期货、富途、IB、Binance 等），开箱即用。
  - CTA 策略引擎：止损/止盈/滑点/手续费内置，趋势跟踪策略即插即用。
  - 回测 → 实盘同代码迁移；Qt5 GUI。
  - v3.7+ 支持 Docker 部署。
- **适用**：国内期货/股票 CTA 实盘、多交易所接入团队。
- **局限**：单标的回测为主，多因子选股需自定义扩展；ML 集成弱。
- **与本项目关系**：本项目走 baostock + 盘后选股路线，若未来要接实盘可考虑 vnpy 的 CTP 接口层。

### 3. backtrader/backtrader ⭐ 22k
- **定位**：纯 Python 事件驱动回测框架，入门经典。
- **核心能力**：100+ 内置指标；Cerebro 抽象清晰；30 行实现完整回测 + 画图。
- **状态**：⚠️ 2019 年后基本停止维护，Python 3.10+ 有兼容问题；单线程、大数据量慢；无 CTP 接口。
- **适用**：学习事件驱动概念、小数据量策略原型验证。
- **价值**：API 设计优雅，适合理解回测引擎内部机制。

### 4. quantopian/zipline ⭐ 19k
- **定位**：Quantopian 开源的本地回测平台，曾驱动全球最大在线量化社区。
- **核心能力**：与 pyfolio / alphalens 无缝衔接；pipeline 因子计算体系。
- **状态**：Quantopian 已关停，社区维护的 [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) 仍在更新。
- **适用**：因子研究、美股/ETF 回测；A 股需自行适配数据源。

### 5. ricequant/rqalpha ⭐ 6k+
- **定位**：米筐开源的本地量化回测 + 实盘框架，API 与 Quantopian 保持一致。
- **核心能力**：事件驱动、插件化架构，支持股票/期货/期权/REITs；处理分红送转、滑点、手续费。
- **适用**：A 股本地回测的首选之一；License 完全排除商业用途（需注意）。

### 6. QUANTAXIS ⭐ 9k+
- **定位**：分布式全栈量化平台，数据采集 → 回测 → 模拟 → 实盘 → 可视化一体化。
- **核心能力**：v2.1 集成 Rust 核心（QARS2）性能提升 ~100 倍；内置 A 股数据爬虫；支持股票/期货/期权。
- **局限**：依赖 MongoDB/Node.js 等多组件，安装复杂；文档滞后于代码；硬件要求高（建议 16GB+）。

### 7. vectorbt (polakowo/vectorbt) ⭐ 4k+
- **定位**：基于 NumPy 的向量化极速回测框架。
- **核心能力**：比事件驱动快 100~1000 倍；参数热力图、优化曲线等专业可视化；与 Jupyter 完美集成。
- **局限**：不支持实盘；复杂订单逻辑（条件单、拆单）受限；滑点手续费模型简单。
- **适用**：策略快速验证、大规模参数寻优、纯研究场景。

### 8. quantconnect/lean ⭐ 16k+
- **定位**：QuantConnect 的云端+本地算法交易引擎，C# 核心 + Python 策略。
- **核心能力**：多资产（股票/期权/外汇/加密）、多市场；事件驱动；可云端部署。
- **适用**：国际市场全流程；A 股数据需自行接入。

## 二、数据与基础设施

### 9. akshare ⭐ 10k+
- **定位**：A 股财经数据接口库（股票/期货/期权/基金/外汇/债券/指数/加密）。
- **价值**：免费、覆盖广，是个人量化的主力数据源之一。
- **与本项目关系**：本项目用 baostock（K 线稳定、复权干净），可考虑 akshare 补充北向资金、龙虎榜、财务等 baostock 不提供的数据。

### 10. waditu/tushare ⭐ 13k+
- **定位**：老牌 A 股数据接口，pro 版覆盖广、响应快。
- **局限**：高频调用有积分限制；老版 API 有下线风险。

### 11. baostock（本项目数据源）
- **定位**：免费 A 股日/周/月/分钟 K 线 + 估值数据，覆盖 1990 至今。
- **优点**：复权干净、字段全（开高低收前收 + 量额换手 + PE/PB/ST）；无积分墙。
- **局限**：高频连续请求（百次量级）后偶发连接重置；无盘口/分钟级深度数据。
- **本项目应对**：`core/data_fetcher.py` 重连重试 + 限速 + 断点续跑；`build_data_library.py` 落地 Parquet 离线回溯。

### 12. manahl/arctic ⭐ 3k+
- **定位**：基于 MongoDB 的高性能时间序列存储，适合 tick 数据。
- **备选**：timescale（PostgreSQL 时序扩展）、InfluxDB。

## 三、绩效分析与因子分析

### 13. quantopian/pyfolio ⭐ 5k+
- **定位**：组合绩效与风险分析库，标准化的回测报告（收益、回撤、夏普、Sortino、归因）。
- **价值**：回测后报告的事实标准。

### 14. quantopian/alphalens ⭐ 3k+
- **定位**：因子表现分析库——IC 衰减、分位层收益、因子自相关。
- **价值**：因子有效性检验的事实标准。

### 15. ranaroussi/quantstats ⭐ 6k+
- **定位**：更深层次的组合分析与风险指标，输出 HTML 报告。
- **价值**：一行代码生成专业级绩效报告，适合个人低频策略汇报。

## 四、策略与研究类项目

### 16. bbfamily/abu ⭐ 16k+
- **定位**：基于 Python 的量化交易系统，模块化数据源 + 规则风控 + 机器学习信号。
- **特点**：代码示例丰富、中文文档好、面向个人；A 股/加密双轨。

### 17. zvtvz/zvt ⭐ 4k+
- **定位**：统一的数据记录 + 因子计算 + 选股 + 回测 + 实盘 + 可视化框架。
- **特点**：抽象度高、半自动化标签（量化信号 + 人工干预）。

### 18. myhhub/stock ⭐ 13k+
- **定位**：算法交易框架，含数据采集（代理轮换、会话保活）、技术规则驱动下单。

### 19. AI4Finance-Foundation/FinRL ⭐ 15k+
- **定位**：深度强化学习交易框架（NeurIPS 2020）。
- **价值**：RL 选股/择时研究；门槛高、偏学术。

### 20. stefan-jansen/machine-learning-for-trading ⭐ 3k+
- **定位**：《Machine Learning for Algorithmic Trading》一书配套代码，端到端 ML 量化示例。

## 五、资源索引

### 21. wilsonfreitas/awesome-quant
- **定位**：最权威的量化资源 awesome list（数据源、回测、交易、风险分析、ML 全覆盖）。

### 22. hikyuu ⭐ 2k+
- **定位**：基于 C++ + Python 的高性能量化研究框架，强调因子与回测速度。

---

## 选型对比总表

| 项目 | Star | 定位 | 维护 | A 股 | 实盘 | ML |
|------|------|------|------|------|------|-----|
| qlib | 44k | AI 量化研究 | ✅ 活跃 | ★★★★ | △ | ★★★★★ |
| vnpy | 41k | 全栈实盘 | ✅ 活跃 | ★★★★ | ★★★★★ | ★★ |
| backtrader | 22k | 入门回测 | ⚠️ 停更 | ★★ | ✗ | ✗ |
| zipline | 19k | 因子回测 | 社区维护 | ★★ | ✗ | ★★ |
| rqalpha | 6k | A 股回测 | △ | ★★★★ | △ | ★★ |
| QUANTAXIS | 9k | 全栈分布式 | △ alpha | ★★★★ | ★★★ | ★★ |
| vectorbt | 4k | 极速回测 | ✅ | ★★ | ✗ | ★★ |
| Lean | 16k | 云端全资产 | ✅ | ★★ | ★★★★ | ★★ |

---

## 对本项目的选型建议

`stock_quantile_research` 是一个 **A 股盘后选股 + 回溯 + 胜率模型** 的轻量自研系统，定位偏个人研究而非实盘交易。基于此定位：

1. **数据层**：继续用 **baostock** 作为 K 线主源（本项目已封装成熟），用 **akshare** 补充北向/龙虎榜/财务等 baostock 缺失字段。
2. **回测层**：本项目自研引擎已满足"信号 → 持有 N 日 → 胜率/涨幅统计"的核心需求；若需更复杂的组合级回测（多标的、调仓、换手约束），可评估 **vectorbt**（向量化快）或 **rqalpha**（A 股原生）。
3. **因子/ML 层**：本项目 `core/features.py`（18 特征）+ `train_model.py`（DT/RF/GBDT）已走通；若要上深度模型（LSTM/Transformer）可参考 **qlib** 的模型库与 Alpha158 因子定义。
4. **绩效分析层**：建议引入 **quantstats**，对 `backtest_output/` 的回测结果生成标准化 HTML 报告，提升结论可读性。
5. **实盘演进**：本项目当前为盘后研究，若未来走向实盘，**vnpy** 的 CTP 接口层是国内最成熟选择。

---

## 参考链接
- qlib: https://github.com/microsoft/qlib
- vnpy: https://github.com/vnpy/vnpy
- backtrader: https://github.com/backtrader/backtrader
- zipline-reloaded: https://github.com/stefan-jansen/zipline-reloaded
- rqalpha: https://github.com/ricequant/rqalpha
- QUANTAXIS: https://github.com/yunqisean00/QUANTAXIS
- vectorbt: https://github.com/polakowo/vectorbt
- Lean: https://github.com/QuantConnect/Lean
- akshare: https://github.com/akfamily/akshare
- tushare: https://github.com/waditu/tushare
- abu: https://github.com/bbfamily/abu
- zvt: https://github.com/zvtvz/zvt
- FinRL: https://github.com/AI4Finance-Foundation/FinRL
- pyfolio: https://github.com/quantopian/pyfolio
- alphalens: https://github.com/quantopian/alphalens
- quantstats: https://github.com/ranaroussi/quantstats
- awesome-quant: https://github.com/wilsonfreitas/awesome-quant
