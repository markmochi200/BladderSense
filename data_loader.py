import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def parse_volume_from_filename(fname):
    """
    parse filenames like 's 99.2.jpg' or 't 76.2.jpg' or 's_99.2.jpg' etc.
    returns (view, float_volume) or None if not matched.
    """
    base = os.path.basename(fname)
    # allow s or t, optional separators, capture float (start of name)
    m = re.match(r'^\s*([stST])[_\s-]*([0-9]+(?:\.[0-9]+)?)', base)
    if not m:
        return None
    view = m.group(1).lower()
    vol = float(m.group(2))
    return view, vol


class BladderPairsDataset(Dataset):
    """
    Dataset that returns paired images (transverse, sagittal) and the float volume.
    Only pairs where both s and t exist for the same numeric volume are used.

    Either provide `pairs_list` directly (list of (t_path, s_path, vol)) or `root_dir` to scan.
    """
    def __init__(self, root_dir=None, pairs_list=None, image_size=224, transform=None):
        assert root_dir is not None or pairs_list is not None, "Provide root_dir or pairs_list"
        self.root_dir = root_dir

        if pairs_list is not None:
            self.pairs = pairs_list
        else:
            files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            mapping = {}  # volume -> {'s': path, 't': path}
            for f in files:
                parsed = parse_volume_from_filename(f)
                if parsed is None:
                    continue
                view, vol = parsed
                vol_key = float(vol)
                if vol_key not in mapping:
                    mapping[vol_key] = {}
                mapping[vol_key][view] = os.path.join(root_dir, f)
            self.pairs = []
            for vol_key, d in mapping.items():
                if 's' in d and 't' in d:
                    self.pairs.append((d['t'], d['s'], float(vol_key)))
            self.pairs.sort(key=lambda x: x[2])

        # transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),  # (C=1,H,W) with values in [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_t, path_s, vol = self.pairs[idx]
        img_t = Image.open(path_t).convert('L')
        img_s = Image.open(path_s).convert('L')
        img_t = self.transform(img_t)
        img_s = self.transform(img_s)
        return img_t, img_s, torch.tensor(vol, dtype=torch.float32)
