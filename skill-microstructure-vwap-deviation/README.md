# skill-microstructure-vwap-deviation

分钟线 VWAP 偏离交易研究与 SSQuant 期货正式回测 skill。

## 用途

- 计算按品种隔离的滚动 VWAP 与价格偏离标准差。
- 研究均值回归、趋势过滤、尾盘禁止开仓和开盘失败等信号。
- 保证信号在当前 bar 生成、下一 bar 开盘成交。
- 使用冻结、哈希和审计后的 SSQuant 数据进行正式期货回测。
- 输出可复核的交易记录、净值曲线、报告和运行清单。

## 目录

```text
skill-microstructure-vwap-deviation/
├─ SKILL.md                         # Codex skill 主入口
├─ agents/                          # Codex 与编辑器元数据
├─ references/                      # 方法、数据、策略、执行和输出契约
├─ scripts/                         # 可执行入口与策略实现
│  └─ research/                     # 冻结数据、矩阵和机会研究脚本
└─ tests/                           # 回归测试
```

## 快速验证

在 skill 根目录执行：

```powershell
python scripts/test.py
```

测试脚本会把 pytest 缓存写到 skill 同级的
`skill-microstructure-vwap-deviation-runs`，不会污染 skill 包。

## Standalone 研究

合成数据只用于验证流程：

```powershell
python scripts/run_backtest.py --synthetic `
  --output-dir ..\skill-microstructure-vwap-deviation-runs\standalone\synthetic
```

真实数据运行前必须确认数据源、调整方式、交易时段、手续费和滑点参数。默认参数见
`references/parameter_policy.md`。

## SSQuant 正式回测

正式回测需要包外的 SSQuant 项目根目录、引擎路径和冻结数据清单：

```powershell
$env:SSQUANT_DATA_MODE = "frozen"
$env:SSQUANT_PROJECT_ROOT = "C:\path\to\ssquant-project"
$env:SSQUANT_ENGINE_PATH = "D:\path\to\ssquant\ssquant"
$env:SSQUANT_DATASET_MANIFEST = "C:\path\to\dataset_manifest.json"
$env:SSQUANT_OUTPUT_DIR = "C:\path\to\skill-microstructure-vwap-deviation-runs\formal\IM888\5m_120m"
python scripts/run_index_mtf_formal.py
```

正式回测必须使用官方变换后的冻结数据，记录 bars hash、调整方式、策略版本、费用和
滑点，并生成完整正式产物。具体规则见：

- `references/method_guide.md`
- `references/data_guide.md`
- `references/strategy_contract.md`
- `references/execution_contract.md`
- `references/output_contract.md`
- `references/review_checklist.md`

## 输出边界

skill 包只包含源码、文档、测试和 agent 元数据。以下内容必须放在包外：

- 冻结数据和数据清单
- `equity_curve.csv`、`equity_curve.png`
- `trades.csv`、`report.md`、`run_manifest.json`
- 比较结果、拼接结果、日志、缓存和 Python 字节码

默认外部目录为 `skill-microstructure-vwap-deviation-runs`，可通过
`VWAP_RESEARCH_OUTPUT_ROOT` 覆盖。

## 打包

打包时应以 `skill-microstructure-vwap-deviation/` 作为压缩包内的顶层目录，并排除所有
运行产物和缓存。README 是包内的人类入口；Codex 执行规则以 `SKILL.md` 为准。
