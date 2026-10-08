"""Plot the summaries from a completed run; requires the optional [plot] extra."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    methods = json.loads((args.results / 'summary.json').read_text())['methods']
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y - .18, [r['throughput_tokens_per_sec'] for r in methods], height=.34,
            color='#aeb8c6', label='Full trace (including drain)')
    ax.barh(y + .18, [r['window_throughput_tokens_per_sec'] for r in methods], height=.34,
            color='#3677b7', label='Measurement window')
    ax.set_yticks(y, [r['policy'] for r in methods])
    ax.invert_yaxis()
    ax.set_xlabel('Output throughput (tokens/s)')
    fig.suptitle('BalanceRoute CPU simulation — paired synthetic inputs', y=.99)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .93), ncol=2, frameon=False)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.02, .015, 'Simulation results; throughput excludes router execution time. Inspect window saturation in summary.json.', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .88])
    for extension in ['png', 'pdf']:
        fig.savefig(args.results / f'comparison.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
