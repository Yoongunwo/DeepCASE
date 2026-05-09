"""
DeepCASE evaluation script for BGL supercomputer log dataset.

Requires BGL_deepcase_train.py to have been run first (produces bgl_model/).

Prediction score semantics (from DeepCASE)
-------------------------------------------
  > 0  : anomaly (maliciousness score)
  = 0  : known benign
  -1   : confidence below threshold
  -2   : event not seen during training
  -3   : nearest cluster exceeds epsilon

The --uncertain flag controls how codes -1 / -2 / -3 are classified.

Usage
-----
  cd DeepCASE
  python BGL_deepcase_eval.py
  python BGL_deepcase_eval.py --uncertain anomaly --k 1
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepcase.context_builder.context_builder import ContextBuilder
from deepcase.interpreter.interpreter          import Interpreter


# ── Helper ────────────────────────────────────────────────────────────────────

def select_device(gpu_id=None):
    if not torch.cuda.is_available():
        print('[Device] CPU (CUDA not available)')
        return 'cpu'
    n      = torch.cuda.device_count()
    chosen = gpu_id if gpu_id is not None else \
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
    parser = argparse.ArgumentParser(description='DeepCASE evaluation on BGL dataset')
    parser.add_argument('--model_dir',    default='bgl_model')
    parser.add_argument('--k',            default=1,          type=int,
                        help='Number of nearest clusters to consider')
    parser.add_argument('--batch_size',   default=1024,       type=int)
    parser.add_argument('--chunk_size',   default=50_000,     type=int,
                        help='Test samples processed per interpreter.predict() call. '
                             'interpreter.predict() accumulates CUDA allocator pool '
                             'across batches; chunking + empty_cache() between chunks '
                             'prevents OOM on large test sets. Set 0 to disable.')
    parser.add_argument('--uncertain',    default='anomaly',
                        choices=['anomaly', 'benign'],
                        help='How to classify uncertain scores (<0): '
                             '"anomaly" (conservative) or "benign" (strict)')
    parser.add_argument('--gpu',          default=None,       type=int)
    args = parser.parse_args()

    device_str = select_device(args.gpu)

    # ── Load preprocessed data ────────────────────────────────────────────────
    preproc_path = os.path.join(args.model_dir, 'bgl_preprocessed.pt')
    print(f'\nLoading preprocessed data : {preproc_path}')
    data = torch.load(preproc_path, map_location='cpu', weights_only=False)

    events_all  = data['events_all']
    context_all = data['context_all']
    labels_all  = data['labels_all']
    train_size  = data['train_size']
    n_features  = data['n_features']

    test_events  = events_all [train_size:]
    test_context = context_all[train_size:]
    test_labels  = labels_all [train_size:].numpy()

    n_benign  = (test_labels == 0).sum()
    n_anomaly = (test_labels == 1).sum()
    print(f'  Test samples : {len(test_events):,}  '
          f'(benign={n_benign:,}, anomaly={n_anomaly:,})')

    # ── Load models ───────────────────────────────────────────────────────────
    cb_path     = os.path.join(args.model_dir, 'bgl_context_builder.pt')
    interp_path = os.path.join(args.model_dir, 'bgl_interpreter.pkl')
    print(f'Loading ContextBuilder : {cb_path}')
    cb = ContextBuilder.load(cb_path, device=device_str)
    cb.eval()
    print(f'Loading Interpreter    : {interp_path}')
    interpreter = Interpreter.load(interp_path, context_builder=cb)

    # ── Predict ───────────────────────────────────────────────────────────────
    # interpreter.predict() calls context_builder.query() which runs 100
    # attention-optimisation iterations per batch.  PyTorch's CUDA allocator
    # pool grows monotonically across all batches and never shrinks mid-call.
    # Processing the test set in chunks with empty_cache() between them prevents
    # the pool from accumulating to OOM.
    print('\nPredicting...')
    t0 = time.time()
    n_test = len(test_events)

    chunk = args.chunk_size if args.chunk_size > 0 else n_test
    chunks = range(0, n_test, chunk)
    parts  = []

    for start in tqdm(chunks, desc='Chunks', leave=False, disable=(chunk >= n_test)):
        end   = min(start + chunk, n_test)
        preds = interpreter.predict(
            X          = test_context[start:end].to(device_str),
            y          = test_events [start:end].unsqueeze(1).to(device_str),
            k          = args.k,
            batch_size = args.batch_size,
            verbose    = (chunk >= n_test),   # inner progress only when no outer bar
        )
        parts.append(np.asarray(preds).reshape(-1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.concatenate(parts)
    elapsed = time.time() - t0

    # ── Decode predictions ────────────────────────────────────────────────────
    # > 0 : anomaly  | = 0 : benign  | < 0 : uncertain
    if args.uncertain == 'anomaly':
        pred_binary = (predictions != 0).astype(int)   # anything non-zero → anomaly
    else:
        pred_binary = (predictions > 0).astype(int)    # only positive → anomaly

    # ── Confusion matrix ──────────────────────────────────────────────────────
    tp = int(((pred_binary == 1) & (test_labels == 1)).sum())
    fp = int(((pred_binary == 1) & (test_labels == 0)).sum())
    tn = int(((pred_binary == 0) & (test_labels == 0)).sum())
    fn = int(((pred_binary == 0) & (test_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Score distribution ────────────────────────────────────────────────────
    n_pos  = int((predictions >  0).sum())
    n_zero = int((predictions == 0).sum())
    n_neg1 = int((predictions == -1).sum())
    n_neg2 = int((predictions == -2).sum())
    n_neg3 = int((predictions == -3).sum())

    print(f"\n{'=' * 60}")
    print(f"  Results  (uncertain → '{args.uncertain}',  k={args.k})")
    print(f"{'=' * 60}")
    print(f"  Prediction score breakdown:")
    print(f"    > 0  (anomaly)              : {n_pos:>10,}")
    print(f"    = 0  (benign)               : {n_zero:>10,}")
    print(f"    -1   (low confidence)       : {n_neg1:>10,}")
    print(f"    -2   (unseen event)         : {n_neg2:>10,}")
    print(f"    -3   (cluster too distant)  : {n_neg3:>10,}")
    print(f"{'─' * 60}")
    print(f"  TP={tp:,}   FP={fp:,}   TN={tn:,}   FN={fn:,}")
    print(f"{'─' * 60}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print('Finished Predicting')
