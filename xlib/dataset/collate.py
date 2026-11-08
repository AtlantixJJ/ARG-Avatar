"""Custom collate functions for datasets with PIL images."""
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data.dataloader import default_collate


def _is_pil_image_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], Image.Image)


def collate_with_pil_images(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that keeps PIL images as nested lists.

    Tensor-like fields are collated with default_collate; PIL image fields are
    kept as List[List[PIL.Image]] with shape [B][N].
    """
    if not batch:
        return {}

    # Support ConcatDataset which returns (dataset_key, sample_dict) tuples.
    # Keep the dataset keys as a separate list to match existing consumers.
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2 and isinstance(batch[0][1], dict):
        dataset_keys = [item[0] for item in batch]
        samples = [item[1] for item in batch]
        return [dataset_keys, collate_with_pil_images(samples)]

    sample = batch[0]
    result: Dict[str, Any] = {}
    for key, value in sample.items():
        field_values = [s[key] for s in batch]
        if _is_pil_image_list(value):
            result[key] = field_values
            continue
        try:
            result[key] = default_collate(field_values)
        except TypeError:
            result[key] = field_values

    return result
