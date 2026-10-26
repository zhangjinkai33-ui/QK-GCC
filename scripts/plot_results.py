# Copyright (c) 2026 jinkz33@XMU&UofG
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SNR = [0, 6, 12, 18, 24, 30]
T60 = [.2, .4, .6, .8, 1.0]
METHODS = [('gcc_samples', 'GCC-PHAT', '#2878B5'),
           ('pgcc_samples', 'PGCC-PHAT (Cls.)', '#239B83'),
           ('ngcc_samples', 'NGCC-PHAT', '#6AAFC7'),
           ('qkgcc_samples', 'QK-GCC', '#CE620C')]
PREVIEW_FILES = ('scatter_snr0_t60_0.2.csv', 'scatter_snr12_t60_0.4.csv',
                 'scatter_snr12_t60_0.6.csv')


def read_summary(path):
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for item in csv.DictReader(stream):
            row = dict(snr=int(item['snr']), t60=float(item['t60']), n=int(item['n']))
            for key, old in [('mae', 'v4s_mae'), ('rmse', 'v4s_rmse'), ('acc_10cm', 'v4s_acc')]:
                row[key] = float(item.get(key, item.get(old)))
            if row['n'] <= 0 or not all(math.isfinite(row[key]) for key in ('mae', 'rmse', 'acc_10cm')):
                raise ValueError('Invalid condition metrics')
            if not 0 <= row['mae'] <= row['rmse'] or not 0 <= row['acc_10cm'] <= 1:
                raise ValueError('Metrics outside valid range')
            rows.append(row)
    keys = [(row['snr'], row['t60']) for row in rows]
    if len(keys) != 30 or set(keys) != {(snr, t60) for snr in SNR for t60 in T60}:
        raise ValueError('Expected each of the 30 SNR/T60 conditions exactly once')
    return sorted(rows, key=lambda row: (row['snr'], row['t60']))


def aggregate(rows):
    n = sum(row['n'] for row in rows)
    return dict(n=n, mae=sum(row['n'] * row['mae'] for row in rows) / n,
                rmse=math.sqrt(sum(row['n'] * row['rmse'] ** 2 for row in rows) / n),
                acc_10cm=sum(row['n'] * row['acc_10cm'] for row in rows) / n)


def read_comparison(path, summary):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        records = list(csv.DictReader(stream))
    names = [method[1] for method in METHODS]
    expected = {(snr, t60, name) for snr in SNR for t60 in T60 for name in names}
    rows = []
    for record in records:
        row = dict(snr=int(record['snr']), t60=float(record['t60']), method=record['method'],
                   seed=int(record['seed']), n=int(record['n']))
        row.update({key: float(record[key]) for key in ('mae', 'rmse', 'acc_10cm')})
        if row['seed'] != 1234 or row['n'] <= 0:
            raise ValueError('Comparison must use seed1234 and nonempty conditions')
        if not all(math.isfinite(row[key]) for key in ('mae', 'rmse', 'acc_10cm')):
            raise ValueError('Nonfinite comparison metric')
        if not 0 <= row['mae'] <= row['rmse'] or not 0 <= row['acc_10cm'] <= 1:
            raise ValueError('Invalid comparison metric')
        rows.append(row)
    if len(rows) != 120 or {(row['snr'], row['t60'], row['method']) for row in rows} != expected:
        raise ValueError('Expected all 30 conditions for all four methods exactly once')
    by_condition = {(row['snr'], row['t60']): row for row in summary}
    for row in rows:
        reference = by_condition[row['snr'], row['t60']]
        if row['n'] != reference['n']:
            raise ValueError('Four-method sample counts do not match')
        if row['method'] == 'QK-GCC' and any(abs(row[key] - reference[key]) > 1e-6
                                              for key in ('mae', 'rmse', 'acc_10cm')):
            raise ValueError('QK comparison differs from the released full-grid summary')
    return rows


def comparison_tables(rows, benchmarks):
    names = [method[1] for method in METHODS]
    prefixes = ('gcc', 'pgcc', 'ngcc', 'qkgcc')
    lines = ['## Complete SNR × T60 comparison', '',
        'Seed 1234. MAE/RMSE: samples; Acc@10cm: %. **Bold**: best per condition.', '',
        '[Download full-precision CSV](../benchmarks/librispeech_four_methods_seed1234_wide.csv)', '']
    totals = {name: aggregate([row for row in rows if row['method'] == name]) for name in names}

    def formatted(value, metric, values):
        best = max(values) if metric == 'acc_10cm' else min(values)
        text = f'{100 * value:.2f}' if metric == 'acc_10cm' else f'{value:.4f}'
        return f'**{text}**' if abs(value - best) <= 1e-10 else text

    headers = ['SNR (dB)', 'T60 (s)', 'Pairs']
    for label in ('GCC', 'PGCC', 'NGCC', 'QK-GCC'):
        headers.extend([f'{label} MAE', f'{label} RMSE', f'{label} Acc (%)'])
    lines += ['| ' + ' | '.join(headers) + ' |', '|' + '---:|' * len(headers)]
    wide_rows = []
    indexed = {(row['snr'], row['t60'], row['method']): row for row in rows}
    for snr in SNR:
        for t60 in T60:
            cell = {name: indexed[snr, t60, name] for name in names}
            wide = dict(snr=snr, t60=t60, n=cell[names[0]]['n'])
            values = [str(snr), f'{t60:.1f}', f"{wide['n']:,}"]
            for prefix, name in zip(prefixes, names):
                for key in ('mae', 'rmse', 'acc_10cm'):
                    wide[f'{prefix}_{key}'] = cell[name][key]
                    values.append(formatted(cell[name][key], key, [cell[method][key] for method in names]))
            wide_rows.append(wide)
            lines.append('| ' + ' | '.join(values) + ' |')
    with (benchmarks / 'librispeech_four_methods_seed1234_wide.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(wide_rows[0]))
        writer.writeheader()
        writer.writerows(wide_rows)
    return lines, totals


def save_figure(fig, path):
    fig.savefig(Path(str(path) + '.png'), dpi=250, bbox_inches='tight')
    fig.savefig(Path(str(path) + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def scatter(path, assets, summary, reference='quantized'):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [row['sample_id'] for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise ValueError('Empty or duplicate scatter sample IDs')
    snr, t60 = int(rows[0]['snr']), float(rows[0]['t60'])
    if any(int(row['snr']) != snr or float(row['t60']) != t60 for row in rows):
        raise ValueError('A scatter CSV must contain exactly one condition')
    target = np.array([float(row['target_samples']) for row in rows])
    qk = np.array([float(row['qkgcc_samples']) for row in rows])
    expected = next(row for row in summary if row['snr'] == snr and row['t60'] == t60)
    actual = dict(n=len(rows), mae=float(np.abs(qk - target).mean()),
                  rmse=float(np.square(qk - target).mean() ** .5),
                  acc_10cm=float((np.abs(qk - target) * 343 / 160 < 10).mean()))
    if actual['n'] != expected['n'] or any(abs(actual[key] - expected[key]) > 1e-6 for key in ('mae', 'rmse', 'acc_10cm')):
        raise ValueError(f'Scatter and full-test summary disagree: {path.name}')
    if reference == 'continuous':
        target = np.array([float(row['raw_target_samples']) for row in rows])
    factor = 343 / 160
    selected = [method for method in METHODS if method[0] in rows[0]]
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4.65), squeeze=False)
    scores = {}
    for ax, (key, title, color) in zip(axes[0], selected):
        prediction = np.array([float(row[key]) for row in rows])
        error_cm = np.abs(prediction - target) * factor
        scores[title] = dict(n=len(rows), mae_cm=float(error_cm.mean()), acc_10cm=float((error_cm < 10).mean()))
        ax.plot([-52, 52], [-52, 52], '--', color='.72', linewidth=.8, zorder=0)
        ax.scatter(target * factor, prediction * factor, s=9, alpha=.22, c=color, edgecolors='none', rasterized=True)
        ax.set(xlim=(-52, 52), ylim=(-52, 52), xticks=[-50, -25, 0, 25, 50],
               yticks=[-50, -25, 0, 25, 50], xlabel='Reference delay (cm)',
               ylabel='Predicted delay (cm)', title=title)
        ax.set_aspect('equal')
        ax.text(.5, -.27, f"MAE: {error_cm.mean():.2f} cm\nAcc@10cm: {100 * (error_cm < 10).mean():.2f}%",
                transform=ax.transAxes, ha='center', va='top', linespacing=1.6)
    fig.suptitle(f'SNR = {snr} dB, $T_{{60}}$ = {t60:g} s | {len(rows):,} pairs | seed 1234', y=.98)
    fig.subplots_adjust(left=.07, right=.98, bottom=.3, top=.84, wspace=.38)
    suffix = '_continuous' if reference == 'continuous' else ''
    save_figure(fig, assets / (path.stem + suffix))
    return dict(condition=dict(snr=snr, t60=t60), reference=reference, methods=scores,
                csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description='Build condition tables and figures from saved results')
    parser.add_argument('--summary', type=Path, default=Path('benchmarks/librispeech_seed1234.csv'))
    parser.add_argument('--output-root', type=Path, default=Path('.'))
    parser.add_argument('--predictions', type=Path, nargs='*')
    parser.add_argument('--reference', choices=['quantized', 'continuous'], default='quantized')
    parser.add_argument('--comparison', type=Path)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    rows = read_summary(args.summary)
    if sum(row['n'] for row in rows) != 95850:
        raise ValueError('The released full LibriSpeech grid must contain 95,850 pairs')
    assets, benchmarks, docs = [args.output_root / name for name in ('assets', 'benchmarks', 'docs')]
    for folder in (assets, benchmarks, docs):
        folder.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    normalized = benchmarks / 'librispeech_seed1234.csv'
    with normalized.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    predictions = args.predictions if args.predictions is not None else [benchmarks / name for name in PREVIEW_FILES]
    panels = [scatter(path, assets, rows, args.reference) for path in predictions]
    overall = aggregate(rows)
    comparison_path = args.comparison or benchmarks / 'librispeech_four_methods_seed1234.csv'
    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)
    comparison = read_comparison(comparison_path, rows)
    table_lines, totals = comparison_tables(comparison, benchmarks)
    report = dict(owner='jinkz33@XMU&UofG', model='QK-GCC V4S', seed=1234,
                  dataset='LibriSpeech full V2 test grid', conditions=30, overall=overall,
                  input_summary_sha256=source_hash, scatter=panels,
                  matched_four_method_totals=totals, audio_distributed=False)
    (benchmarks / 'visualization_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Detailed test results', '', 'Maintainer: `jinkz33@XMU&UofG`', ''] + table_lines
    lines += ['', '## Prediction examples', '',
        'Three illustrative noise/reverberation conditions are shown below. Each four-method',
        'comparison includes all 3,195 pairs, including outliers, with identical sample IDs',
        'and rounded reference delays. These examples do not replace the complete results above.', '',
        'Plot axes and MAE are in **cm** (1 sample = 2.14375 cm at 16 kHz and 343 m/s).',
        'Acc@10cm is the fraction of absolute errors below 10 cm. No audio is distributed.', '']
    if args.reference == 'continuous':
        lines += ['Continuous-reference plots use raw simulated delays; their metrics differ from rounded-grid evaluation.', '']
    for path in predictions:
        suffix = '_continuous' if args.reference == 'continuous' else ''
        lines += [f'![{path.stem}](../assets/{path.stem}{suffix}.png)', '']
    (docs / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == '__main__':
    main()
