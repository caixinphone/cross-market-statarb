# 跨市场特质均值回归(统计套利)

一个横跨加密与美股代理资产的**市场中性统计套利**回测,以 Binance 统一跨市场账户为执行背景。
每个资产用滚动多因子回归对系统性因子(BTC、ETH、SPY、QQQ、SMH)建模;交易其**特质残差**回归
均值,同时对冲掉因子暴露,使组合只赚"短期定价错误被修正"的钱。

```
r_i,t = α_i + Σ_f β_{i,f}·r_f,t + ε_i,t      # 滚动因子模型
signal = 残差价差的 z-score                  # 相对因子有多贵/多便宜
trade  = 反向交易偏离 + 按 β 对冲因子          # 市场中性配对
```

## 快速开始

```bash
pip install -e .                 # 按 pyproject.toml 锁版本装依赖
# 美股数据用 Alpaca:把密钥放到 config/secrets.yaml(已 gitignore)
# 或设环境变量 ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
python scripts/download_data.py  # 拉取+缓存原始数据,构建对齐面板
python scripts/run_backtest.py   # 因子模型 -> 信号 -> 组合 -> 指标
python scripts/sensitivity.py    # 参数/成本/容量网格 -> reports/
pip install -e ".[ml]"           # 可选:torch,用于 ML 加分项(见报告 §16)
python scripts/run_ml.py         # CNN 定仓 + 漂移门控 vs 基线(§16)
pytest -q                        # 20 个测试:无前视、风控、成本、资金费、门控、ML……
```

一切由 [`config/config.yaml`](config/config.yaml) 驱动——资产池、因子映射、信号阈值、风控上限、
成本——所以敏感性/成本/容量分析都是**改配置即可、无需改代码**。**主轨道为小时频**
(`frequency: hourly`,默认);切到 `frequency: daily` 可复现日频对比(两者跑在同一份 1h 缓存上)。
按 bar 计数的窗口用小时频量纲(日频日历惯例 × 7 根 RTH bar/日)。完整写作见
**[reports/strategy_report.md](reports/strategy_report.md)**。

## 架构

| 模块 | 职责 |
|---|---|
| `src/data/sources.py` | 取数:Binance 批量归档(加密 K 线+资金费)、Alpaca(美股)、Yahoo(兜底) |
| `src/data/fetch.py` | 编排 + 幂等 parquet 缓存 + 改名拼接 |
| `src/data/align.py` | 跨市场 UTC 对齐 → 收益 + 成交额面板 |
| `src/factors/factor_model.py` | 时点安全的滚动 OLS → β、残差 |
| `src/factors/diagnostics.py` | R² + 残差平稳性(ADF)表 |
| `src/signals/zscore.py` | 残差 z-score、半衰期过滤、进出场状态机、漂移门控 |
| `src/portfolio/construct.py` | 等波动率定仓,开仓即冻结 |
| `src/portfolio/risk.py` | 因子对冲 + 资产/板块/杠杆/因子上限 |
| `src/portfolio/execution.py` | 免交易带 → 实际持仓账本 |
| `src/backtest/{engine,costs}.py` | 时点安全回测 + 手续费/流动性滑点/资金费/借券 |
| `src/analysis/{metrics,attribution,plots}.py` | 指标、PnL 分解、图表 |
| `src/ml/{dataset,cnn,meta_sizing}.py` | ML 加分项:1D CNN 把握分配定仓(§16,可选) |
| `scripts/sensitivity.py` | 参数/成本/容量网格 |

## 数据来源与说明(重要)

* **Binance 实时 API 被地域封锁(HTTP 451)**。加密数据改用**官方公开归档
  `data.binance.vision`**(现货 1h K 线 + USDⓈ-M 资金费)——完整到 2023、无需密钥,本就是更适合
  回测的数据源。归档在 2025 年把时间戳单位从 ms 改成 µs;`sources.py` 逐元素归一化。
* **美股**来自 **Alpaca 行情 API**(完整 2023→至今小时级,免费层 IEX feed;SIP 需付费)。密钥放在
  已 gitignore 的 `config/secrets.yaml` 或环境变量里。Yahoo 作为日频兜底(不强依赖 `yfinance`);
  Yahoo 小时历史只有 ~730 天,所以全 2023 小时序列用 Alpaca。
* **回测设定。** Binance 的 TradFi 永续在 2023 年并不存在,所以股票腿用真实美股价格、加密腿用
  Binance;"Binance 跨市场"是执行论点,而非历史数据来源。
* **对齐。** 小时频(主)与日频(对比)面板都派生自**同一份 1h 缓存**。小时频对齐到美股 RTH 核心
  (14:00–20:00 UTC,7 根/日),加密 reindex 到这些整点;日频价 = 美股交易日历上 ≤21:00 UTC 收盘
  快照的最后一根 1h bar。≤1h 的夏令时偏移可忽略。
* **生存偏差 / 改名**由配置处理(`delistings`、`symbol_overrides`);上市前缺口是真实缺失而非回填
  (如 ARB/OP 在 2023 上线前)。**POL** 是 MATIC→POL 改名:Binance 的 `POLUSDT` 只从 ~2024-09 改名
  起才有,所以在加上 MATIC→POL 拼接前 POL 只带约 50% 历史。
* **股票 feed。** Alpaca 免费层 = IEX feed(成交量子集);流动标的的 OHLC 有代表性,成交量只是
  IEX 口径。SIP(全盘)需付费(`sources.feed: sip`)。

## 方法学要点

* **日历等价窗口(小时频轨道)。** 按 bar 计数的窗口用日频 Avellaneda-Lee 惯例 × 7 根 RTH bar/日:
  因子/z-score 窗口 420(≈60 交易日),最大持仓 140(≈20 天)。直接用 60 根小时 bar(≈8.5 天)会
  过短并制造虚假换手——这是核心方法学点。
* **无前视。** 第 `t` 根的 β/残差只用结束于 `t` 的窗口;引擎在 `t+1` 执行(`positions.shift(1)`)。
  由 `tests/test_no_lookahead.py` 强制(扰动未来 bar 不能改变过去信号)。
* **平稳性门槛。** 每条残差都做 ADF 检验;只在残差均值回归处才有边际(25 条全平稳,ADF p≈0)。
* **等波动率定仓。** `N_i = (目标波动 / σ_resid_i) · AUM/信号数`。
* **风控上限**(全部验证成立):单资产 ≤3%、板块 ≤15%、总杠杆 ≤3×、净因子暴露 ≤5% AUM。
* **成本。** 按腿分档 taker 费(股票=现货 0.10%、加密=永续 0.04%)+ 成交名义额上的流动性滑点、
  USDⓈ-M 资金费(真实 8h 序列,逐 bar 计提、隔夜结算累计到次日首根 RTH bar)、做空美股借券——
  每条腿的费用与资金费/借券口径自洽。

## 可复现性

`pyproject.toml` 锁版本依赖;配置里单一全局种子;下载带缓存且幂等(删 `data/raw` / `data/processed`
即可强制干净重建)。管道是确定性的——重跑得到完全一致的结果。

## 核心结果(小时频,全 25 标的资产池,2023→2026)

完整写作见 **[reports/strategy_report.md](reports/strategy_report.md)**。`run_backtest.py` 打印一份
PnL 分解,把*边际在哪*与*被什么吃掉*分开。所有风控上限成立(杠杆 0.27×、单资产 3.0%、净因子
4.8%);对每个因子的净 β ≈ 0。面板:5944 根小时 bar × 29 列。

| 组成 | PnL ($) |
|---|--:|
| 特质边际 `Σ held·ε` | **+599,187** |
| alpha 漂移 + 对冲误差 | **−547,306** |
| 毛利(可交易) | +51,881 |
| 手续费+滑点 / 资金费 / 借券 | −234,654 |
| **净** | **−182,772** |

头条:夏普 −0.44,CAGR −0.55%,波动 1.24%,最大回撤 −3.10%,换手 6.2×/年,平均持仓 ≈109 根 bar
(~15.6 交易日)。手续费按场所分档:股票腿 Binance 现货 taker 0.10%、加密腿永续 taker 0.04%。

* **特质回归边际真实且巨大**(毛利 +$599k),但**市场中性的残差回归组合隐含做空特质动量**——
  在 2023-25 牛市里做空高漂移名,会流失"样本内残差剔除掉、真实 β 对冲却剔除不了"的那部分漂移
  (−$547k)。这个拖累比成本还大,是核心发现。
* **方法学正确的小时频 ≈ 日频。** 用日历等价的 420-bar 窗口,持仓约 15 天、换手仅 6.2×/年——
  **不是**朴素 60-小时-bar 跑出的 43×。边际/漂移/净值结构与日频一致(+$596k / −$561k / −$147k)。
  更多 bar 给出更稳的统计,**不是一个不同的策略**。
* **容量不是约束:** 夏普在 $10M→$200M 之间持平(成交额巨大、冲击轻)。天花板是*边际*,不是流动性。
* **诚实判断:** 单薄、参数敏感、净值微负——是一个多元化中性组合里可用的*组件*(尤其配合
  漂移/动量叠加),而非独立策略。这是**方法学修正版**的小时频跑法;换手控制/盈利向调优刻意**未**
  施加。完整敏感性/容量分析与路线图见报告。
