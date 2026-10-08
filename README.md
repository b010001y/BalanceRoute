# BalanceRoute

Official code for **[Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale](https://arxiv.org/abs/2605.06113)**, accepted at **NeurIPS 2026**.

Tianci Bu, Yuan Lyu, Zixi Chen, Chendong Song, Hong Liang, Tsepten Gurung, Yuwei Fan, Yinyu Ye, Zijie Zhou

[Paper](https://arxiv.org/abs/2605.06113) · [中文说明](README.zh-CN.md)

BalanceRoute routes requests across data-parallel LLM serving workers. **BR-0** uses current loads; **BR-H** uses a finite lookahead of request growth and termination.

This initial release contains the core routing algorithms, a CPU simulator, and a reproducible synthetic benchmark. BR-H uses oracle remaining lengths for active requests. The complete release, including the termination predictor, serving integration, and paper evaluation, is planned before the NeurIPS 2026 conference.

## Installation

Python 3.11 or newer; no GPU required.

```bash
git clone https://github.com/b010001y/BalanceRoute.git
cd BalanceRoute
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

## Run

Quick check on one bundled input:

```bash
balanceroute smoke
```

Run the complete benchmark and verify the saved results:

```bash
balanceroute reproduce --workers 3 --output results/r16
balanceroute verify results/r16
```

The benchmark compares **Random, Round Robin (RR), Power of Two Choices (P2C), Join Shortest Queue (JSQ), Least Token Load (LTL), BR-0, and BR-H** across ten seeds. Each policy receives the same requests for a given seed.

The [default configuration](src/balanceroute/presets/r16.json) uses eight simulated workers, a batch limit of 72 requests per worker, and 8,000 requests with mean prompt/output lengths of 3,000/1,000 tokens. **R=16 is the number of candidate requests in subset selection.** BR-H uses H=80, β=43, and γ=0.86. `--workers` controls the number of CPU processes running experiments.

Each run writes its configuration, input hashes, per-seed metrics, summary CSVs, and verification report to the output directory. Use a new directory for each run. Throughput follows the simulator's decode-time model and excludes routing execution time; the [reproduction protocol](docs/REPRODUCIBILITY.md) defines the metrics and assumptions.

## Code structure

```text
src/balanceroute/
├── strategies/
│   ├── balance_route.py   # BR-0 / BR-H scoring, projection, and routing
│   └── baselines.py       # Random, RR, P2C, JSQ, and LTL
├── scheduler.py           # Decode simulation and metrics
├── workload.py            # Synthetic request lengths and input loading
├── benchmark.py           # Experiment runner and reference verification
├── cli.py                 # Command-line interface
├── presets/r16.json       # Default experiment configuration
└── reference/r16/         # Frozen inputs and expected metrics
tests/                    # Algorithm, simulator, and reproduction tests
examples/plot_results.py  # Optional plots from a completed run
```

## Custom experiments

Export the configuration, edit it, and run:

```bash
balanceroute show-config --output experiment.json
balanceroute run --config experiment.json --output results/custom
```

Set `input_source` to `generated` when changing the workload or seeds. See the [protocol](docs/REPRODUCIBILITY.md) for configuration details.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Citation

```bibtex
@article{bu2026balanceroute,
  title   = {Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale},
  author  = {Bu, Tianci and Lyu, Yuan and Chen, Zixi and Song, Chendong and Liang, Hong and Gurung, Tsepten and Fan, Yuwei and Ye, Yinyu and Zhou, Zijie},
  journal = {arXiv preprint arXiv:2605.06113},
  year    = {2026}
}
```

## License

[Apache-2.0](LICENSE).
