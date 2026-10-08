# BalanceRoute

**NeurIPS 2026** 录用论文 **[Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale](https://arxiv.org/abs/2605.06113)** 的官方代码。

[Paper](https://arxiv.org/abs/2605.06113) · [English](README.md)

BalanceRoute 用于数据并行 LLM 服务的请求路由。**BR-0** 根据当前负载分配请求；**BR-H** 进一步考虑有限预测窗口内的请求增长与结束。

当前发布包含核心算法、CPU 模拟器和可复现的合成负载基准。BR-H 使用运行中请求的真实剩余长度作为 oracle。包含结束预测器、服务系统集成和论文评测的完整版本计划于 NeurIPS 2026 会议前发布。

## 安装

需要 Python 3.11 或更新版本，无需 GPU。

```bash
git clone https://github.com/b010001y/BalanceRoute.git
cd BalanceRoute
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

## 运行

单个种子的快速检查：

```bash
balanceroute smoke
```

完整运行并校验结果：

```bash
balanceroute reproduce --workers 3 --output results/r16
balanceroute verify results/r16
```

基准包含 **Random、Round Robin（RR）、Power of Two Choices（P2C）、Join Shortest Queue（JSQ）、Least Token Load（LTL）、BR-0 和 BR-H**，共十个种子。同一种子下，各方法使用相同请求。

[默认配置](src/balanceroute/presets/r16.json) 使用八台模拟机器，每台最多同时运行 72 个请求；每个种子包含 8,000 个请求，平均输入/输出长度为 3,000/1,000 tokens。**R=16 表示子集选择时的候选请求数。** BR-H 使用 H=80、β=43、γ=0.86。`--workers` 指运行实验的 CPU 进程数。

运行目录保存配置、输入哈希、逐种子指标、汇总 CSV 和校验报告。每次运行需指定新目录。吞吐由模拟器的解码时间模型计算，不含路由执行耗时；指标和假设见[复现协议](docs/REPRODUCIBILITY.md)。

## 代码结构

```text
src/balanceroute/
├── strategies/
│   ├── balance_route.py   # BR-0 / BR-H 的评分、负载预测与路由
│   └── baselines.py       # Random、RR、P2C、JSQ、LTL
├── scheduler.py           # 解码模拟与指标计算
├── workload.py            # 合成请求长度与输入加载
├── benchmark.py           # 实验运行与参考结果校验
├── cli.py                 # 命令行入口
├── presets/r16.json       # 默认实验配置
└── reference/r16/         # 固定输入与预期指标
tests/                    # 算法、模拟器与复现测试
examples/plot_results.py  # 根据运行结果绘图
```

## 自定义实验

导出配置，修改后运行：

```bash
balanceroute show-config --output experiment.json
balanceroute run --config experiment.json --output results/custom
```

修改负载分布或种子时，将 `input_source` 设为 `generated`。配置说明见[复现协议](docs/REPRODUCIBILITY.md)。

## 测试

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## 引用与许可

引用格式见 [README](README.md#citation) 或 [CITATION.cff](CITATION.cff)。代码采用 [Apache-2.0](LICENSE) 许可证。
