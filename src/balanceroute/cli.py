"""Command line interface for the public reproducibility snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from . import __version__
from .benchmark import preset, run, verify


def _print_summary(report):
    print('\nPolicy                 Window tok/s   Full tok/s   TPOT P95 ms')
    for row in report['methods']:
        print(f"{row['policy']:22} {row['window_throughput_tokens_per_sec']:12.1f} "
              f"{row['throughput_tokens_per_sec']:12.1f} {row['tpot_p95_ms']:13.2f}")
    print('CPU simulation; throughput excludes routing execution time.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='BalanceRoute early algorithm snapshot: reproducible CPU experiments.')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('smoke', 'Run all seven policies on reference seed 100 and verify the subset.'),
                            ('reproduce', 'Run and verify the R=16 benchmark: ten seeds, seven policies.'),
                            ('run', 'Run an explicit JSON configuration.')]:
        p = commands.add_parser(name, help=help_text)
        p.add_argument('--output', type=Path, help='New output directory; an existing path is rejected.')
        p.add_argument('--workers', type=int, default=1, help='CPU processes (default: 1).')
        if name == 'run':
            p.add_argument('--config', type=Path, required=True)
    p = commands.add_parser('verify', help='Verify a saved run against the frozen R=16 reference.')
    p.add_argument('output', type=Path)
    p.add_argument('--allow-subset', action='store_true', help='Explicitly allow a subset of the reference seeds/policies.')
    p = commands.add_parser('show-config', help='Print the complete R=16 configuration for inspection or modification.')
    p.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'show-config':
            if args.output:
                with args.output.open('x', encoding='utf-8') as stream:
                    stream.write(json.dumps(preset(), indent=2) + '\n')
            else:
                print(json.dumps(preset(), indent=2))
            return 0
        if args.command == 'verify':
            report = verify(args.output, require_complete=not args.allow_subset)
            print(f"Verified {report['verified_runs']} runs ({report['scope']}).")
            return 0
        config = json.loads(args.config.read_text(encoding='utf-8')) if args.command == 'run' else preset()
        if args.command == 'smoke':
            config['seeds'] = [100]
        output = args.output or Path('results') / (args.command + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
        print('Results:', output, flush=True)
        report = run(config, output, workers=args.workers, mode=args.command)
        _print_summary(report)
        if args.command in ['smoke', 'reproduce']:
            validation = verify(output, require_complete=args.command == 'reproduce')
            print(f"Verified {validation['verified_runs']} runs ({validation['scope']}).")
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
