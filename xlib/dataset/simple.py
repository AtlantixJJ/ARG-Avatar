"""Simple datasets and dataset utilities."""
import os
import torch
from PIL import Image
import numpy as np
from xlib.utils import construct_class_by_name


class ConcatDataset(torch.utils.data.Dataset):
    """Concatenate items of multiple datasets with interleaving.

    Builds an index mapping so that global index i maps to a specific
    dataset and its local index. Datasets are interleaved in chunks of
    size ``sync_freq``: we take ``sync_freq`` consecutive samples from
    one dataset, then switch to the next dataset, cycling through all
    datasets.

    If datasets have different lengths, local indices wrap around using
    modulo so the iterator remains well-defined.
    """
    def __init__(self, ds_list, ds_keys, ds_sync=None):
        self.ds_list = []
        for ds_key, dic in zip(ds_keys, ds_list):
            print(f"Constructing dataset: {ds_key}")
            ds = construct_class_by_name(**dic)
            self.ds_list.append(ds)
        
        # Normalize OmegaConf ListConfig inputs to plain Python lists so worker
        # processes do not hit lazy ListConfig indexing during __getitem__.
        self.ds_keys = list(ds_keys)
        self.ds_sync = list(ds_sync) if ds_sync is not None else [True] * len(ds_list)

        # Only sync datasets contribute to cumulative length
        sync_lens = [len(ds) if self.ds_sync[i] else 0
                     for i, ds in enumerate(self.ds_list)]
        other_lens = [len(ds) if not self.ds_sync[i] else 0
                      for i, ds in enumerate(self.ds_list)]
        self.cumsum_sync_lens = np.cumsum(sync_lens)
        self.cumsum_other_lens = np.cumsum(other_lens)
        # Build index arrays (dataset ids and local indices) without loops over items
        sync_ds = []
        sync_loc = []
        other_ds = []
        other_loc = []
        for i, ds in enumerate(self.ds_list):
            n = len(ds)
            if n == 0:
                continue
            ds_ids = np.full(n, i, dtype=np.int64)
            loc_ids = np.arange(n, dtype=np.int64)
            if self.ds_sync[i]:
                sync_ds.append(ds_ids)
                sync_loc.append(loc_ids)
            else:
                other_ds.append(ds_ids)
                other_loc.append(loc_ids)
        sync_ds = np.concatenate(sync_ds, axis=0) if len(sync_ds) > 0 else np.empty((0,), dtype=np.int64)
        sync_loc = np.concatenate(sync_loc, axis=0) if len(sync_loc) > 0 else np.empty((0,), dtype=np.int64)
        other_ds = np.concatenate(other_ds, axis=0) if len(other_ds) > 0 else np.empty((0,), dtype=np.int64)
        other_loc = np.concatenate(other_loc, axis=0) if len(other_loc) > 0 else np.empty((0,), dtype=np.int64)

        S = sync_ds.shape[0]
        O = other_ds.shape[0]

        assert S != 0 or O != 0, "All datasets are empty"

        if S == 0:
            self._ds_arr = other_ds
            self._local_arr = other_loc
        elif O == 0:
            self._ds_arr = sync_ds
            self._local_arr = sync_loc
        else:
            self.sync_freq = self.cumsum_sync_lens[-1] // self.cumsum_other_lens[-1] \
                if self.cumsum_other_lens[-1] > 0 else 0
            k = self.sync_freq
            # reshape sync into rows of length k with padding
            num_chunks = int((S + k - 1) // k)
            pad = num_chunks * k - S
            if pad > 0:
                sync_ds_pad = np.full(pad, -1, dtype=np.int64)
                sync_loc_pad = np.full(pad, -1, dtype=np.int64)
                sync_ds_grid = np.concatenate([sync_ds, sync_ds_pad], axis=0).reshape(num_chunks, k)
                sync_loc_grid = np.concatenate([sync_loc, sync_loc_pad], axis=0).reshape(num_chunks, k)
            else:
                sync_ds_grid = sync_ds.reshape(num_chunks, k)
                sync_loc_grid = sync_loc.reshape(num_chunks, k)

            # prepare one 'other' per row for the first m rows
            m = int(min(O, num_chunks))
            other_col_ds = np.full((num_chunks, 1), -1, dtype=np.int64)
            other_col_loc = np.full((num_chunks, 1), -1, dtype=np.int64)
            if m > 0:
                other_col_ds[:m, 0] = other_ds[:m]
                other_col_loc[:m, 0] = other_loc[:m]

            # concatenate per-row: k sync then 1 other (if present)
            final_ds_grid = np.concatenate([sync_ds_grid, other_col_ds], axis=1)
            final_loc_grid = np.concatenate([sync_loc_grid, other_col_loc], axis=1)
            flat_ds = final_ds_grid.ravel()
            flat_loc = final_loc_grid.ravel()
            # drop padding
            valid_mask = flat_ds >= 0
            flat_ds = flat_ds[valid_mask]
            flat_loc = flat_loc[valid_mask]
            if O > m:
                flat_ds = np.concatenate([flat_ds, other_ds[m:]], axis=0)
                flat_loc = np.concatenate([flat_loc, other_loc[m:]], axis=0)
            self._ds_arr = flat_ds
            self._local_arr = flat_loc

        # Positions where an 'other' dataset sample is used (helpful for debugging/analysis)
        if self._ds_arr.shape[0] > 0 and O > 0:
            ds_sync_arr = np.array(self.ds_sync, dtype=bool)
            self.sync_indices = np.nonzero(~ds_sync_arr[self._ds_arr])[0].tolist()
        else:
            self.sync_indices = []

    def __len__(self):
        return int(self._ds_arr.shape[0])

    def __repr__(self):
        strs = [f'{ds_name}={len(ds)}{"" if self.ds_sync[i] else "(async)"}'
                for i, (ds_name, ds) in enumerate(zip(self.ds_keys, self.ds_list))]
        return ' '.join(strs)

    def __getitem__(self, idx):
        ds_idx = int(self._ds_arr[idx])
        local_idx = int(self._local_arr[idx])
        return self.ds_keys[ds_idx], self.ds_list[ds_idx][local_idx]


class UnionDataset(torch.utils.data.Dataset):
    """Return items of multiple datasets in a dictionary."""
    def __init__(self, ds_list, ds_keys):
        self.ds_list = [construct_class_by_name(**dic) for dic in ds_list]
        self.ds_keys = list(ds_keys)

    def __len__(self):
        return max([len(ds) for ds in self.ds_list])

    def __getitem__(self, idx):
        return {self.ds_keys[i]: ds[idx % len(ds)]
                for i, ds in enumerate(self.ds_list)}


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image-only datasets.
    """
    def __init__(self, data_path, size=None, transform=None):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        image_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    file_path = os.path.join(root, file)
                    #if 'masks' in file_path:
                    #    continue
                    image_files.append(file_path)
        image_files.sort()
        self.files = image_files

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(fpath, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size:
                img = img.resize(self.size, Image.Resampling.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'image_path': fpath
        }

    def __len__(self):
        return len(self.files)
