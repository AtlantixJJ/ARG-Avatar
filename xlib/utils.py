"""All utility functions."""
import os
import sys
import time
import threading
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf, DictConfig
from typing import Any, Callable, Optional, Tuple


class DictRecorder(dict):
    """A dictionary that can record data and stack them."""

    def __init__(self):
        super().__init__()

    def add(self, key, val):
        if key not in self:
            self[key] = []
        if isinstance(val, torch.Tensor):
            self[key].append(val.detach().cpu())
        elif isinstance(val, list):
            for i, v in enumerate(val):
                if len(self[key]) <= i:
                    self[key].append([])
                self[key][i].append(v)
        else:
            self[key].append(val)
    
    def add_dict(self, dic):
        for key, val in dic.items():
            self.add(key, val)

    def stack(self):
        for key in list(self.keys()):
            if len(self[key]) > 0:
                if isinstance(self[key][0], torch.Tensor):
                    self[key] = torch.stack(self[key]).squeeze()
                elif isinstance(self[key][0], list):
                    dic = {}
                    for i in range(len(self[key])):
                        dic[str(i)] = torch.stack(self[key][i])
                    self[key] = dic
            else:
                del key
                
    def clear(self):
        """Clear all recorded data from the dictionary."""
        super().clear()
    
    def summarize(self, prefix=''):
        self.stack()
        sum_dic = {}
        for key in list(self.keys()):
            if isinstance(self[key], torch.Tensor):
                sum_dic[prefix + key] = self[key].mean()
            elif isinstance(self[key], dict):
                for k, v in self[key].items():
                    sum_dic[prefix + key + '/' + k] = v.mean()
        # clean up
        self.clear()
        return sum_dic


# Config loading and by-name construction now live in xlib/config.py;
# re-exported here so existing `from xlib.utils import ...` call sites keep working.
from xlib.config import (  # noqa: F401
    _load_config_with_bases,
    load_config,
    get_module_from_obj_name,
    get_obj_from_module,
    get_obj_by_name,
    call_func_by_name,
    construct_class_by_name,
    get_module_dir_by_obj_name,
)


########## Image Reading ##########


def imread_tensor(fpath: str) -> torch.Tensor:
    """Read to 0-1 tensor in [3, H, W]."""
    x = torch.from_numpy(imread(fpath)).float()
    if x.max() > 5:
        x /= 255.
    return x.permute(2, 0, 1)


def imread_pil(fpath: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    img = Image.open(open(fpath, "rb")).convert("RGB")
    if size is not None:
        img = img.resize(size, resample=Image.Resampling.BILINEAR)
    return img


def imread(fpath: str) -> np.ndarray:
    img = Image.open(open(fpath, "rb")).convert("RGB")
    return np.asarray(img).copy()


def imwrite(fpath: str, arr: np.ndarray):
    with open(fpath, "wb") as f:
        Image.fromarray(arr).save(f)


    
########## Other ###########


class GeneralThread(threading.Thread):
    """A general thread for running a function."""
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args, self.kwargs = args, kwargs

    def run(self):
        try:
            self.res = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.res = None
            self.exception = e


class AsyncExecutor:
    """Record the merging process as a video.
    """
    def __init__(self):
        self.thread = None

    def __call__(self, func: Callable, *args):
        if self.thread is not None:
            self.thread.join()
        self.thread = GeneralThread(func, *args)
        self.thread.start()
    
    def clean_sync(self):
        if self.thread is not None:
            self.thread.join()
            if hasattr(self.thread, 'exception'):
                raise self.thread.exception
            res = self.thread.res
            self.thread = None
            return res
        
        print("AsyncExecutor: No thread to join")





########## Dictionary ##########


def dict_append(dic: dict, val: Any, key1: str, key2: Optional[str] = None, key3: Optional[str] = None):
    """Create a list or append to it with at most 3 levels."""
    if key1 and not key2 and not key3:
        if key1 not in dic:
            dic[key1] = []
        dic[key1].append(val)
        return

    if key1 not in dic:
        dic[key1] = {}
    if key1 and key2 and not key3:
        if key2 not in dic:
            dic[key1][key2] = []
        dic[key1][key2].append(val)
        return

    if key2 not in dic[key1]:
        dic[key1][key2] = {}
    if key3 not in dic[key1][key2]:
        dic[key1][key2][key3] = []
    dic[key1][key2][key3].append(val)


def cat_dict(dics: list[dict], cat_tensor=True, is_stack=True, device='cpu') -> dict:
    """Concatenate a dictionary of tensors."""
    keys = dics[0].keys()
    res = {}
    for k in keys:
        l = [d[k] for d in dics]
        if isinstance(l[0], torch.Tensor) and cat_tensor:
            res[k] = torch.stack(l) if is_stack else torch.cat(l)
            device = res[k].device
        elif isinstance(dics[0][k], list):
            res[k] = []
            for d in l:
                res[k].extend(d)
        else:
            res[k] = l

        """
        if isinstance(dics[0][k], torch.Tensor):
            res[k] = torch.stack(l) if is_stack else torch.cat(l)
        """
    for k in keys:
        l = [d[k] for d in dics]
        if (isinstance(l[0], float) or isinstance(l[0], int)) and cat_tensor:
            res[k] = torch.tensor(l, device=device)

    return res
