"""
DeepCASE training script for BGL supercomputer log dataset.

Pipeline
--------
1. Parse BGL_benign.log + BGL_anomaly.log → combined time-ordered DataFrame
2. Build string→int vocabulary from the full dataset
3. Run DeepCASE Preprocessor to generate context sequences
4. Save preprocessed tensors (shared by eval script)
5. Split by --ratio (sequential from the start, time order)
6. Train ContextBuilder on benign-only training samples
7. Fit Interpreter on the same benign training samples (score=0)
8. Save models

Usage
-----
  cd DeepCASE
  python BGL_deepcase_train.py
  python BGL_deepcase_train.py --ratio 0.7 --epochs 10 --gpu 0
"""

import argparse
import os
import re
import sys

import numpy  as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

# DeepCASE lives in the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepcase.preprocessing.preprocessor    import Preprocessor
from deepcase.context_builder.context_builder import ContextBuilder
from deepcase.interpreter.interpreter        import Interpreter


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_log_key(line):
    """Normalise one BGL log line into a template string."""
    parts = line.strip().split(None, 9)
    if len(parts) < 9:
        return None
    log_type  = parts[6]
    component = parts[7]
    level     = parts[8]
    content   = parts[9] if len(parts) > 9 else ""
    content   = re.sub(r'0x[0-9a-fA-F]+',    '<HEX>', content)
    content   = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>',  content)
    content   = re.sub(r':\d+\b',             ':<*>',  content)
    content   = re.sub(r'\b\d+\b',            '<*>',   content)
    content   = re.sub(r'\s+',                ' ',     content).strip()
    return f"{log_type}|{component}|{level}|{content}"


def parse_bgl_log(log_path, label_value):
    """Parse a BGL log file → DataFrame(timestamp, event, machine, label)."""
    records = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'  Parsing {os.path.basename(log_path)}', leave=False):
            parts = line.strip().split(None, 9)
            if len(parts) < 9:
                continue
            key = extract_log_key(line)
            if key:
                records.append({
                    'timestamp': int(parts[1]),
                    'event'    : key,
                    'machine'  : parts[3],   # Node ID, e.g. R02-M1-N0-C:J12-U11
                    'label'    : label_value,
                })
    return pd.DataFrame(records)


def select_device(gpu_id=None):
    """Pick the GPU with the most free memory (or the requested GPU), else CPU."""
    if not torch.cuda.is_available():
        print('[Device] CPU (CUDA not available)')
        return 'cpu'
    n        = torch.cuda.device_count()
    chosen   = gpu_id if gpu_id is not None else \
               max(range(n), key=lambda i: torch.cuda.mem_get_info(i)[0])
    if gpu_id is None and n > 1:
        summary = '  |  '.join(
            f'GPU{i}: {torch.cuda.mem_get_info(i)[0]/1024**3:.1f} GB free'
            for i in range(n)
        )
        print(f'[Device] Auto-selected GPU {chosen}  [{summary}]')
    torch.cuda.set_device(chosen)
    torch.backends.cudnn.benchmark = True
    print(f'[Device] {torch.cuda.get_device_name(chosen)}  (cuda:{chosen})')
    return f'cuda:{chosen}'


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepCASE training on BGL dataset')
    parser.add_argument('--benign_log',  default='../Data/BGL/BGL_benign.log')
    parser.add_argument('--anomaly_log', default='../Data/BGL/BGL_anomaly.log')
    parser.add_argument('--model_dir',   default='bgl_model')
    parser.add_argument('--ratio',       default=0.8,   type=float,
                        help='Fraction of data used for training (0~1], time-sequential from start')
    parser.add_argument('--context',     default=10,    type=int,
                        help='Context window size (number of preceding events)')
    parser.add_argument('--timeout',     default=3600,  type=int,
                        help='Max seconds between context events (default: 1 hour)')
    parser.add_argument('--complexity',  default=128,   type=int,
                        help='Hidden dimension of ContextBuilder')
    parser.add_argument('--epochs',      default=10,    type=int)
    parser.add_argument('--batch_size',  default=128,   type=int)
    parser.add_argument('--gpu',         default=None,  type=int,
                        help='GPU index (default: auto-select by free memory)')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device_str = select_device(args.gpu)

    # ── 1. Parse logs ─────────────────────────────────────────────────────────
    print('\n[Step 1] Parsing BGL logs...')
    df_benign  = parse_bgl_log(args.benign_log,  label_value=0)
    df_anomaly = parse_bgl_log(args.anomaly_log, label_value=1)
    df_full    = pd.concat([df_benign, df_anomaly], ignore_index=True)
    df_full    = df_full.sort_values('timestamp').reset_index(drop=True)
    print(f'  Total  : {len(df_full):,}  '
          f'(benign={len(df_benign):,}, anomaly={len(df_anomaly):,})')

    # ── 2. Build string vocabulary ────────────────────────────────────────────
    print('\n[Step 2] Building vocabulary...')
    unique_keys      = sorted(df_full['event'].unique())
    vocab            = {k: i for i, k in enumerate(unique_keys)}
    df_full['event'] = df_full['event'].map(vocab)
    print(f'  Vocab size : {len(vocab):,}')

    # ── 3. Preprocess → context sequences ─────────────────────────────────────
    print('\n[Step 3] Building context sequences (Preprocessor)...')
    preprocessor = Preprocessor(context=args.context, timeout=args.timeout)
    events_all, context_all, labels_all, dc_mapping = preprocessor.sequence(
        df_full[['timestamp', 'event', 'machine', 'label']],
        verbose=True,
    )
    n_features = len(dc_mapping)   # includes the NO_EVENT slot
    print(f'  n_features : {n_features}  (event types + NO_EVENT)')
    print(f'  Total seqs : {len(events_all):,}')

    # ── 4. Save preprocessed tensors ──────────────────────────────────────────
    train_size   = int(len(events_all) * args.ratio)
    preproc_path = os.path.join(args.model_dir, 'bgl_preprocessed.pt')
    torch.save({
        'events_all'  : events_all,
        'context_all' : context_all,
        'labels_all'  : labels_all,
        'train_size'  : train_size,
        'n_features'  : n_features,
        'context_len' : args.context,
        'vocab'       : vocab,
        'dc_mapping'  : dc_mapping,
    }, preproc_path)
    print(f'\n  [Saved] {preproc_path}')
    print(f'  Train size : {train_size:,}  (ratio={args.ratio:.2f})')
    print(f'  Test  size : {len(events_all) - train_size:,}')

    # ── 5. Prepare benign-only training split ──────────────────────────────────
    print('\n[Step 4] Preparing training data (benign only)...')
    train_events  = events_all[:train_size]
    train_context = context_all[:train_size]
    train_labels  = labels_all[:train_size]

    benign_mask     = train_labels == 0
    train_events_b  = train_events[benign_mask]
    train_context_b = train_context[benign_mask]
    print(f'  Benign train samples : {benign_mask.sum().item():,}')

    # ── 6. Train ContextBuilder ───────────────────────────────────────────────
    print('\n[Step 5] Training ContextBuilder...')
    cb = ContextBuilder(
        input_size    = n_features,
        output_size   = n_features,
        max_length    = args.context,
        hidden_size   = args.complexity,
        num_layers    = 1,
        bidirectional = False,
        LSTM          = False,
    ).to(device_str)

    cb.fit(
        X             = train_context_b,
        y             = train_events_b.unsqueeze(1),  # shape: (N, 1)
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        learning_rate = 0.01,
        optimizer     = optim.Adam,
        teach_ratio   = 0.5,
        verbose       = True,
    )

    # ── 7. Fit Interpreter ────────────────────────────────────────────────────
    print('\n[Step 6] Fitting Interpreter...')
    interpreter = Interpreter(
        context_builder = cb,
        features        = n_features,
        eps             = 0.1,
        min_samples     = 5,
        threshold       = 0.2,
    )
    score = torch.zeros(len(train_events_b), dtype=torch.float)  # all benign → 0
    interpreter.fit(
        X          = train_context_b,
        y          = train_events_b.unsqueeze(1),
        score      = score,
        batch_size = args.batch_size,
        verbose    = True,
    )

    # ── 8. Save models ────────────────────────────────────────────────────────
    cb_path    = os.path.join(args.model_dir, 'bgl_context_builder.pt')
    interp_path = os.path.join(args.model_dir, 'bgl_interpreter.pkl')
    cb.save(cb_path)
    interpreter.save(interp_path)

    print(f'\n[Saved] ContextBuilder → {cb_path}')
    print(f'[Saved] Interpreter    → {interp_path}')
    print('Finished Training')
