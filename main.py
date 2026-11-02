"""
Top-level CLI to:
 - train all ablation models with K-Fold cross-validation
 - train a single model
 - run benchmarks
"""

import os
import argparse
import json
from sklearn.model_selection import KFold

from data_loader import BladderPairsDataset
from train import train_all_models, train_fold
from test import evaluate_model
from benchmark import run_bench

def build_pairs_from_dir(root_dir):
    ds = BladderPairsDataset(root_dir=root_dir)
    return ds.pairs

def split_pairs_kfold(all_pairs, n_splits=10, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pairs = all_pairs
    folds = []
    indices = list(range(len(pairs)))
    for train_idx, val_idx in kf.split(indices):
        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]
        folds.append((train_pairs, val_pairs))
    return folds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./dataset', help='images root')
    parser.add_argument('--out_dir', default='./checkpoints', help='where to save checkpoints and summaries')
    parser.add_argument('--mode', choices=['train_all', 'train_one', 'bench', 'eval_one'], default='bench')
    parser.add_argument('--model_name', default='base', help='when training/eval single model')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None, help='path to model for eval_one')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == 'bench':
        print("Running benchmarks...")
        run_bench(output_csv=os.path.join(args.out_dir, 'benchmarks.csv'), device=args.device)
        return

    all_pairs = build_pairs_from_dir(args.data_dir)
    print(f"Found {len(all_pairs)} paired samples in {args.data_dir}")
    if len(all_pairs) == 0:
        raise RuntimeError("No paired samples found. Check filenames and parser.")

    folds = split_pairs_kfold(all_pairs, n_splits=args.n_splits)
    if args.mode == 'train_all':
        print("Training all ablation models with cross-validation...")
        train_all_models(
            all_folds=folds,
            out_root=args.out_dir,
            image_size=224,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            device=args.device
        )
        print("Done. Summary saved to", os.path.join(args.out_dir, 'global_summary.json'))
    elif args.mode == 'train_one':
        print("Training single model:", args.model_name)
        model_out = os.path.join(args.out_dir, args.model_name)
        os.makedirs(model_out, exist_ok=True)
        per_fold_metrics = []
        for fold_idx, fold in enumerate(folds):
            fold_out = os.path.join(model_out, f'fold_{fold_idx}')  # type: ignore
            os.makedirs(fold_out, exist_ok=True)
            best_path, history, info = train_fold(
                pairs_list=fold,
                fold_idx=fold_idx,
                output_dir=fold_out,
                model_name=args.model_name,
                image_size=224,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                lr=args.lr,
                num_workers=args.num_workers,
                device=args.device
            )
            per_fold_metrics.append(info)
        with open(os.path.join(model_out, 'summary.json'), 'w') as f:  # type: ignore
            json.dump(per_fold_metrics, f, indent=2)
        print("Saved per-fold results to", model_out)
    elif args.mode == 'eval_one':
        if args.model_path is None:
            raise ValueError("Provide --model_path for eval_one")
        metrics, preds, targets = evaluate_model(args.model_path, all_pairs, batch_size=args.batch_size, num_workers=args.num_workers, model_name=args.model_name, device=args.device)
        print("Evaluation:", metrics)
    else:
        raise ValueError("Unknown mode:", args.mode)

if __name__ == '__main__':
    main()
