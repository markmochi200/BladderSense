import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import BladderPairsDataset
from model import create_model

def evaluate_model(model_path, test_pairs, model_name='base', batch_size=8, num_workers=0, device=None):
    """
    test_pairs: list of (t_path, s_path, vol) tuples
    model_path: checkpoint saved by train_fold
    model_name: name used to create the model
    """
    device = torch.device(device) if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    dataset = BladderPairsDataset(pairs_list=test_pairs, image_size=224)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = create_model(model_name, in_channels=1).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for img_t, img_s, vols in loader:
            img_t = img_t.to(device)
            img_s = img_s.to(device)
            vols = vols.numpy()
            preds = model(img_t, img_s).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(vols)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mse = ((preds - targets) ** 2).mean()
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    n = len(targets)
    metrics = {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'n': int(n)}
    return metrics, preds, targets


def inference_on_input(model_path, s_image_path, t_image_path, model_name='base', device=None):
    """
    Simple cv2-based inference helper. Uses grayscale resize to 224.
    """
    import cv2
    device = torch.device(device) if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def load_and_preprocess(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = (img - 0.5) / 0.5  # normalize as dataset does
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return torch.tensor(img, dtype=torch.float32)

    img_s = load_and_preprocess(s_image_path).unsqueeze(0).to(device)
    img_t = load_and_preprocess(t_image_path).unsqueeze(0).to(device)

    model = create_model(model_name, in_channels=1).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        pred = model(img_t, img_s)
    return float(pred.cpu().item())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--s', required=True)
    parser.add_argument('--t', required=True)
    parser.add_argument('--model_name', default='base')
    args = parser.parse_args()
    print("Prediction:", inference_on_input(args.model_path, args.s, args.t, model_name=args.model_name))
