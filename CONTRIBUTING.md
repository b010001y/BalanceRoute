# Contributing

Install the pinned dependencies and test extra, then run the tests:

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e ".[test]"
python -m pytest -q
```

For routing or simulator changes, also run `balanceroute reproduce` and check `verification.json`. Describe any intended change in behavior; version reference configurations separately when changing the benchmark.

Bug reports should include the command, configuration, Python/NumPy versions, and error. Contributions are licensed under [Apache-2.0](LICENSE).
