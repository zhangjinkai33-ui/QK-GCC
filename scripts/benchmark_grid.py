# Copyright (c) 2026 jinkz33@XMU&UofG
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import conditions
from scripts.export_predictions import PredictionDataset, gcc_phat, load_ngcc, load_pgcc

METHODS = ('GCC-PHAT', 'PGCC-PHAT (Cls.)', 'NGCC-PHAT', 'QK-GCC')
COLUMNS = ('gcc_samples', 'pgcc_samples', 'ngcc_samples', 'qkgcc_samples')
GRID = {(snr, t60) for snr in (0, 6, 12, 18, 24, 30) for t60 in (.2, .4, .6, .8, 1.)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finish(n, absolute, squared, correct):
    return dict(n=n, mae=absolute / n, rmse=math.sqrt(squared / n), acc_10cm=correct / n)


def cached_metrics(path, dataset, hashes):
    metadata_path = path.with_suffix('.json')
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if any(metadata.get(key) != value for key, value in hashes.items()):
        raise ValueError(f'Checkpoint/source mismatch in {path.name}')
    if not metadata.get('all_samples_included'):
        raise ValueError(f'Incomplete cached comparison: {path.name}')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(dataset) or [row['sample_id'] for row in rows] != [item.name for item in dataset.files]:
        raise ValueError(f'Cached samples differ: {path.name}')
    if not all(column in rows[0] for column in COLUMNS):
        raise ValueError(f'Missing cached method: {path.name}')
    result = {}
    for method, column in zip(METHODS, COLUMNS):
        errors = [abs(int(row[column]) - int(row['target_samples'])) for row in rows]
        result[method] = finish(len(rows), sum(errors), sum(error ** 2 for error in errors),
                                sum(error * 343 / 160 < 10 for error in errors))
    return result


def main():
    parser = argparse.ArgumentParser(description='Complete matched four-method test-grid evaluation')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--qkgcc-summary', type=Path, default=Path('benchmarks/librispeech_seed1234.csv'))
    parser.add_argument('--qkgcc-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--pgcc-checkpoint', type=Path, required=True)
    parser.add_argument('--ngcc-checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cache-dir', type=Path, default=Path('benchmarks'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/librispeech_four_methods_seed1234.csv'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.qkgcc_summary.open(newline='', encoding='utf-8-sig') as stream:
        summary_rows = list(csv.DictReader(stream))
    qk = {(int(row['snr']), float(row['t60'])): {
        key: int(row[key]) if key == 'n' else float(row[key])
        for key in ('n', 'mae', 'rmse', 'acc_10cm')} for row in summary_rows}
    found = conditions(args.data_root)
    if len(summary_rows) != 30 or set(qk) != GRID or len(found) != 30 or {(snr, t60) for snr, t60, _ in found} != GRID:
        raise ValueError('Both dataset and QK summary must contain the full 30-condition grid')
    if sum(row['n'] for row in qk.values()) != 95850:
        raise ValueError('Expected 95,850 QK test pairs')
    checkpoint = torch.load(args.qkgcc_checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('seed') != args.seed:
        raise ValueError('Released QK checkpoint seed differs')
    max_tau = checkpoint['model_config']['max_tau']
    hashes = dict(qkgcc_checkpoint_sha256=digest(args.qkgcc_checkpoint),
                  pgcc_checkpoint_sha256=digest(args.pgcc_checkpoint),
                  ngcc_checkpoint_sha256=digest(args.ngcc_checkpoint),
                  pgcc_source_sha256=digest(args.baseline_source),
                  ngcc_source_sha256=digest(args.baseline_source))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    pgcc = load_pgcc(args.baseline_source, args.pgcc_checkpoint, max_tau, device)
    ngcc = load_ngcc(args.baseline_source, args.ngcc_checkpoint, max_tau, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_suffix('.json')
    receipt = dict(owner='jinkz33@XMU&UofG', status='running', seed=args.seed,
                   split='full LibriSpeech V2 test grid', methods=list(METHODS),
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   checkpoint_and_source_hashes=hashes, qkgcc_summary_sha256=digest(args.qkgcc_summary),
                   qkgcc_origin='existing complete V4S evaluation; five cached conditions independently checked',
                   sample_rate=16000, sound_speed=343, max_tau=max_tau,
                   reference='rounded delay grid', tolerance='absolute error <10cm',
                   audio_distributed=False, conditions=[])
    rows = []
    partial = args.output.with_suffix('.partial.csv')
    fields = ['snr', 't60', 'method', 'seed', 'n', 'mae', 'rmse', 'acc_10cm']
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    for number, (snr, t60, directory) in enumerate(found, 1):
        started = time.perf_counter()
        dataset = PredictionDataset(directory, max_tau)
        if len(dataset) != qk[snr, t60]['n']:
            raise ValueError(f'Data count differs from QK summary at {snr}/{t60}')
        cached = args.cache_dir / f'scatter_snr{snr}_t60_{t60:.1f}.csv'
        values = cached_metrics(cached, dataset, hashes)
        origin = 'matched cached predictions' if values else 'fresh baseline inference'
        if values:
            if any(abs(values['QK-GCC'][key] - qk[snr, t60][key]) > 1e-6 for key in ('n', 'mae', 'rmse', 'acc_10cm')):
                raise ValueError('Cached QK predictions and full summary disagree')
        else:
            meters = {name: dict(n=0, absolute=0, squared=0, correct=0) for name in METHODS[:3]}
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=device.type == 'cuda')
            with torch.inference_mode():
                for x1, x2, target, _ in loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    predictions = [gcc_phat(x1, x2, max_tau).cpu(),
                        pgcc(x1[:, None], x2[:, None]).reshape(len(x1), -1).argmax(-1).cpu() - max_tau,
                        ngcc(x1[:, None], x2[:, None]).argmax(-1).cpu() - max_tau]
                    for name, prediction in zip(METHODS[:3], predictions):
                        error = (prediction - target).abs()
                        meter = meters[name]
                        meter['n'] += len(target)
                        meter['absolute'] += int(error.sum())
                        meter['squared'] += int(error.square().sum())
                        meter['correct'] += int((error.double() * 343 / 160 < 10).sum())
            values = {name: finish(**meter) for name, meter in meters.items()}
            values['QK-GCC'] = qk[snr, t60]
        for name in METHODS:
            rows.append(dict(snr=snr, t60=t60, method=name, seed=args.seed, **values[name]))
        with partial.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        receipt['conditions'].append(dict(snr=snr, t60=t60, n=len(dataset), origin=origin,
            sample_ids_sha256=hashlib.sha256('\n'.join(path.name for path in dataset.files).encode()).hexdigest(),
            seconds=time.perf_counter() - started))
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(f'{number}/30 SNR={snr} T60={t60:.1f} n={len(dataset)} {origin}: ' +
              ' | '.join(f'{name} Acc={100 * values[name]["acc_10cm"]:.2f}%' for name in METHODS), flush=True)
    if len(rows) != 120 or any(sum(row['n'] for row in rows if row['method'] == name) != 95850 for name in METHODS):
        raise ValueError('Incomplete final grid')
    partial.replace(args.output)
    receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                   rows=120, n_per_method=95850, csv_sha256=digest(args.output))
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Complete: 30 conditions x four methods, 95,850 pairs per method', flush=True)


if __name__ == '__main__':
    main()
