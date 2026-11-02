import os
import math
import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_loader import BladderPairsDataset
from model import create_model, count_parameters

def get_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for img_t, img_s, vols in tqdm(loader, desc="Train batches", leave=False):
        img_t = img_t.to(device)
        img_s = img_s.to(device)
        vols = vols.to(device)

        optimizer.zero_grad()
        preds = model(img_t, img_s)
        loss = criterion(preds, vols)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img_t.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    eps = 1e-6
    with torch.no_grad():
        for img_t, img_s, vols in loader:
            img_t = img_t.to(device)
            img_s = img_s.to(device)
            vols = vols.to(device)
            preds = model(img_t, img_s)
            loss = criterion(preds, vols)
            running_loss += loss.item() * img_t.size(0)
            all_preds.append(preds.detach().cpu())
            all_targets.append(vols.detach().cpu())

    val_loss = running_loss / len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    mse = ((preds - targets) ** 2).mean()
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    mape = np.mean(np.abs((preds - targets) / (targets + eps))) * 100.0
    return val_loss, rmse, mae, mape


def train_fold(pairs_list, fold_idx, output_dir,
               model_name='base',
               image_size=224, batch_size=8, num_epochs=30,
               lr=1e-4, weight_decay=1e-5, num_workers=0, device=None,
               save_history=True):
    """
    Train on one fold using supplied pairs_list = (train_pairs, val_pairs).
    Returns path to best checkpoint and a metrics dict.
    """
    device = get_device(device)
    train_pairs, val_pairs = pairs_list

    train_dataset = BladderPairsDataset(pairs_list=train_pairs, image_size=image_size)
    val_dataset = BladderPairsDataset(pairs_list=val_pairs, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = create_model(model_name, in_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f'best_{model_name}_fold{fold_idx}.pt')

    history = {'train_mse': [], 'val_mse': [], 'val_rmse': [], 'val_mae': [], 'val_mape': []}
    total_params, trainable = count_parameters(model)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_mae, val_mape = validate(model, val_loader, criterion, device)
        print(f"[{model_name}] Fold {fold_idx} Epoch {epoch}/{num_epochs} "
              f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | Val MAPE: {val_mape:.2f}%")

        history['train_mse'].append(float(train_loss))
        history['val_mse'].append(float(val_loss))
        history['val_rmse'].append(float(val_rmse))
        history['val_mae'].append(float(val_mae))
        history['val_mape'].append(float(val_mape))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_rmse': val_rmse,
                'model_name': model_name
            }, best_model_path)

    if save_history:
        with open(os.path.join(output_dir, f'history_{model_name}_fold{fold_idx}.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # save model info
        info = {
            'model_name': model_name,
            'params_total': total_params,
            'params_trainable': trainable,
            'best_val_rmse': best_val_rmse,
            'best_model_path': best_model_path
        }
        with open(os.path.join(output_dir, f'info_{model_name}_fold{fold_idx}.json'), 'w') as f:
            json.dump(info, f, indent=2)

    return best_model_path, history, {'params_total': total_params, 'params_trainable': trainable, 'best_val_rmse': best_val_rmse}


def train_all_models(all_folds, out_root='./checkpoints', model_names=None,
                     image_size=224, batch_size=8, epochs=30, lr=1e-4, weight_decay=1e-5,
                     num_workers=0, device=None):
    """
    all_folds: list of (train_pairs, val_pairs) for each fold (from main's KFold)
    model_names: list of model name strings; if None uses all 5 models
    Returns summary dictionary with per-model aggregated metrics.
    """
    if model_names is None:
        model_names = ["base", "no_coord", "no_spatial", "no_channel", "no_spatial_no_channel"]

    device = get_device(device)
    os.makedirs(out_root, exist_ok=True)
    summary = {}

    for model_name in model_names:
        print("========================================")
        print(f"Training model variant: {model_name}")
        model_out = os.path.join(out_root, model_name)
        os.makedirs(model_out, exist_ok=True)

        per_fold_metrics = []
        per_fold_paths = []
        for fold_idx, folds in enumerate(all_folds):
            print(f"--- {model_name} fold {fold_idx} ---")
            fold_out = os.path.join(model_out, f'fold_{fold_idx}')
            os.makedirs(fold_out, exist_ok=True)
            best_model_path, history, info = train_fold(
                pairs_list=folds,
                fold_idx=fold_idx,
                output_dir=fold_out,
                model_name=model_name,
                image_size=image_size,
                batch_size=batch_size,
                num_epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                num_workers=num_workers,
                device=device,
                save_history=True
            )
            per_fold_metrics.append(info)
            per_fold_paths.append(best_model_path)

        # aggregate
        vals = [m['best_val_rmse'] for m in per_fold_metrics]
        mean_rmse = float(np.mean(vals))
        std_rmse = float(np.std(vals))
        summary[model_name] = {
            'per_fold': per_fold_metrics,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'paths': per_fold_paths
        }
        with open(os.path.join(model_out, 'summary.json'), 'w') as f:
            json.dump(summary[model_name], f, indent=2)
    with open(os.path.join(out_root, 'global_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
