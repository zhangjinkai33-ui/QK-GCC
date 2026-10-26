# Copyright (c) 2026 jinkz33@XMU&UofG
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import conditions, load_model


class PredictionDataset(Dataset):
    def __init__(self, directory: Path, max_tau: int):
        self.files = sorted(directory.glob('*.pt'))
        self.max_tau = max_tau

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = torch.load(self.files[index], map_location='cpu', weights_only=False)
        target = round(float(sample['delay'])) - self.max_tau
        raw = float(sample.get('raw_delay', target))
        return sample['x1'].float().reshape(-1), sample['x2'].float().reshape(-1), target, raw


def gcc_phat(x1: torch.Tensor, x2: torch.Tensor, max_tau: int):
    size = 2 * x1.shape[-1]
    cross = torch.fft.rfft(x1, n=size) * torch.fft.rfft(x2, n=size).conj()
    response = torch.fft.irfft(cross / (cross.abs() + 1e-8), n=size)
    return torch.cat((response[:, -max_tau:], response[:, :max_tau + 1]), -1).argmax(-1) - max_tau


def load_pgcc(source: Path, checkpoint_path: Path, max_tau: int, device):
    sys.path.insert(0, str(source.resolve().parent))
    spec = importlib.util.spec_from_file_location('pgcc_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('head') not in ('classification', 'classifier'):
        raise ValueError('A classification PGCC checkpoint with head metadata is required')
    if checkpoint.get('max_tau_gcc') != max_tau:
        raise ValueError('PGCC and QK-GCC must use the same delay grid')
    model = module.PGCCPHAT(beta=checkpoint['beta'], max_tau=max_tau, head='classification')
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()


def load_ngcc(source, checkpoint_path, max_tau, device):
    sys.path.insert(0, str(source.resolve().parent))
    spec = importlib.util.spec_from_file_location('ngcc_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.NGCCPHAT(max_tau=max_tau, head='classifier', use_sinc=True,
                           sig_len=2048, num_channels=128, fs=16000)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model', checkpoint))
    return model.to(device).eval()


def append_ngcc(args, selected, max_tau, device):
    model = load_ngcc(args.ngcc_source, args.ngcc_checkpoint, max_tau, device)
    for snr, t60, directory in selected:
        destination = args.output_dir / f'scatter_snr{snr}_t60_{t60:.1f}.csv'
        with destination.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows or 'ngcc_samples' in rows[0]:
            raise ValueError('Expected existing predictions without an NGCC column')
        dataset = PredictionDataset(directory, max_tau)
        if [row['sample_id'] for row in rows] != [path.name for path in dataset.files]:
            raise ValueError('NGCC sample order differs from the existing comparison')
        offset = 0
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        with torch.inference_mode():
            for x1, x2, target, _ in tqdm(loader, desc=f'NGCC {directory.name}'):
                logits = model(x1[:, None].to(device), x2[:, None].to(device))
                predictions = logits.argmax(-1).cpu() - max_tau
                for index, prediction in enumerate(predictions.tolist()):
                    row = rows[offset + index]
                    if int(row['target_samples']) != int(target[index]):
                        raise ValueError('NGCC reference target differs from existing comparison')
                    row['ngcc_samples'] = prediction
                offset += len(x1)
        temporary = destination.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
        metadata_path = destination.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(ngcc_head='classifier', ngcc_seed=args.ngcc_seed,
                        ngcc_selection='minimum validation MAE checkpoint',
                        ngcc_checkpoint_sha256=hashlib.sha256(args.ngcc_checkpoint.read_bytes()).hexdigest(),
                        ngcc_source_sha256=hashlib.sha256(args.ngcc_source.read_bytes()).hexdigest())
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'Added NGCC to {destination}: {len(rows)} matched pairs', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Export aligned scatter predictions, without audio')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snr', type=int, nargs='+', required=True)
    parser.add_argument('--t60', type=float, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('benchmarks'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--pgcc-source', type=Path)
    parser.add_argument('--pgcc-checkpoint', type=Path)
    parser.add_argument('--pgcc-seed', type=int)
    parser.add_argument('--append-ngcc', action='store_true')
    parser.add_argument('--ngcc-source', type=Path)
    parser.add_argument('--ngcc-checkpoint', type=Path)
    parser.add_argument('--ngcc-seed', type=int)
    args = parser.parse_args()
    if bool(args.pgcc_source) != bool(args.pgcc_checkpoint):
        parser.error('Supply both --pgcc-source and --pgcc-checkpoint')
    if args.append_ngcc and not (args.ngcc_source and args.ngcc_checkpoint):
        parser.error('--append-ngcc requires --ngcc-source and --ngcc-checkpoint')
    if bool(args.ngcc_source) != bool(args.ngcc_checkpoint):
        parser.error('Supply both --ngcc-source and --ngcc-checkpoint')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    max_tau = checkpoint['model_config']['max_tau']
    selected = [(snr, t60, path) for snr, t60, path in conditions(args.data_root)
                if snr in args.snr and any(abs(t60 - value) < 1e-8 for value in args.t60)]
    if not selected:
        raise FileNotFoundError('No selected conditions found')
    if args.append_ngcc:
        append_ngcc(args, selected, max_tau, device)
        return
    model = load_model(args.checkpoint, device)
    pgcc = load_pgcc(args.pgcc_source, args.pgcc_checkpoint, max_tau, device) if args.pgcc_source else None
    ngcc = load_ngcc(args.ngcc_source, args.ngcc_checkpoint, max_tau, device) if args.ngcc_source else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for snr, t60, directory in selected:
        destination = args.output_dir / f'scatter_snr{snr}_t60_{t60:.1f}.csv'
        if destination.exists():
            raise FileExistsError(destination)
        dataset = PredictionDataset(directory, max_tau)
        if not len(dataset):
            raise ValueError(f'Empty condition: {directory.name}')
        rows = []
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        with torch.inference_mode():
            for x1, x2, target, raw in tqdm(loader, desc=directory.name):
                x1, x2 = x1.to(device), x2.to(device)
                qk = model(x1, x2).argmax(-1).cpu() - max_tau
                gcc = gcc_phat(x1, x2, max_tau).cpu()
                extra = pgcc(x1[:, None], x2[:, None]).reshape(len(x1), -1).argmax(-1).cpu() - max_tau if pgcc is not None else None
                neural = ngcc(x1[:, None], x2[:, None]).argmax(-1).cpu() - max_tau if ngcc is not None else None
                start = len(rows)
                for index in range(len(x1)):
                    row = dict(sample_id=dataset.files[start + index].name, snr=snr, t60=t60,
                               target_samples=int(target[index]), raw_target_samples=float(raw[index]),
                               gcc_samples=int(gcc[index]), qkgcc_samples=int(qk[index]))
                    if extra is not None:
                        row['pgcc_samples'] = int(extra[index])
                    if neural is not None:
                        row['ngcc_samples'] = int(neural[index])
                    rows.append(row)
        with destination.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = dict(owner='jinkz33@XMU&UofG', model='QK-GCC V4S', seed=checkpoint.get('seed'),
                        n=len(rows), snr=snr, t60=t60, sample_rate=16000, sound_speed=343,
                        reference='rounded delay class minus max_tau; raw reference supplied separately',
                        all_samples_included=True, audio_distributed=False,
                        qkgcc_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        sample_ids_sha256=hashlib.sha256('\n'.join(row['sample_id'] for row in rows).encode()).hexdigest())
        if pgcc is not None:
            metadata.update(pgcc_head='classification', pgcc_seed=args.pgcc_seed,
                            pgcc_checkpoint_sha256=hashlib.sha256(args.pgcc_checkpoint.read_bytes()).hexdigest(),
                            pgcc_source_sha256=hashlib.sha256(args.pgcc_source.read_bytes()).hexdigest())
        if ngcc is not None:
            metadata.update(ngcc_head='classifier', ngcc_seed=args.ngcc_seed,
                            ngcc_checkpoint_sha256=hashlib.sha256(args.ngcc_checkpoint.read_bytes()).hexdigest(),
                            ngcc_source_sha256=hashlib.sha256(args.ngcc_source.read_bytes()).hexdigest())
        destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'Saved {destination}: {len(rows)} aligned predictions', flush=True)


if __name__ == '__main__':
    main()
