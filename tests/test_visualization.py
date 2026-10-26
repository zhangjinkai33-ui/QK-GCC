# Copyright (c) 2026 jinkz33@XMU&UofG
import csv
import math

import pytest
import torch

pytest.importorskip('matplotlib')
from scripts.export_predictions import gcc_phat
from scripts.plot_results import (
    METHODS, PREVIEW_FILES, SNR, T60, aggregate, comparison_tables,
    read_comparison, read_summary, save_figure,
)


def test_weighted_rmse():
    rows = [dict(n=1, mae=1., rmse=1., acc_10cm=1.),
            dict(n=3, mae=3., rmse=3., acc_10cm=0.)]
    result = aggregate(rows)
    assert result == dict(n=4, mae=2.5, rmse=math.sqrt(7), acc_10cm=.25)


def test_reject_duplicate_condition(tmp_path):
    path = tmp_path / 'summary.csv'
    rows = [dict(snr=snr, t60=t60, n=1, mae=1, rmse=2, acc_10cm=.5)
            for snr in SNR for t60 in T60]
    rows[-1] = rows[0]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='exactly once'):
        read_summary(path)


def test_gcc_delay_sign():
    x1, x2 = torch.zeros(2, 128), torch.zeros(2, 128)
    x1[:, 20] = 1
    x2[:, 25] = 1
    assert gcc_phat(x1, x2, 23).tolist() == [-5, -5]


def test_decimal_figure_name(tmp_path):
    import matplotlib.pyplot as plt

    figure, _ = plt.subplots(figsize=(1, 1))
    save_figure(figure, tmp_path / 'scatter_t60_0.2')
    assert (tmp_path / 'scatter_t60_0.2.png').exists()
    assert (tmp_path / 'scatter_t60_0.2.pdf').exists()


def test_four_comparison_methods():
    assert [method[0] for method in METHODS] == [
        'gcc_samples', 'pgcc_samples', 'ngcc_samples', 'qkgcc_samples']


def test_three_requested_examples():
    assert PREVIEW_FILES == ('scatter_snr0_t60_0.2.csv', 'scatter_snr12_t60_0.4.csv',
                             'scatter_snr12_t60_0.6.csv')


def test_full_comparison_coverage_and_best_values(tmp_path):
    rows, summary = [], []
    for snr in SNR:
        for t60 in T60:
            summary.append(dict(snr=snr, t60=t60, n=3195, mae=2., rmse=3., acc_10cm=.7))
            for _, name, _ in METHODS:
                metric = dict(mae=1., rmse=2., acc_10cm=.8) if name == 'NGCC-PHAT' else dict(mae=2., rmse=3., acc_10cm=.7)
                rows.append(dict(snr=snr, t60=t60, method=name, seed=1234, n=3195, **metric))
    path = tmp_path / 'comparison.csv'

    def save(values):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    save(rows)
    parsed = read_comparison(path, summary)
    lines, totals = comparison_tables(parsed, tmp_path)
    assert totals['NGCC-PHAT']['n'] == 95850
    assert '## Complete SNR × T60 comparison' in lines
    assert sum(line.startswith('| 0 | 0.2 |') for line in lines) == 1
    assert '**1.0000** | **2.0000** | **80.00**' in '\n'.join(lines)
    with (tmp_path / 'librispeech_four_methods_seed1234_wide.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 30
    save(rows[:-1])
    with pytest.raises(ValueError, match='all four methods exactly once'):
        read_comparison(path, summary)
