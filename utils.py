import os
import json
import logging
import random
import time
import sys
import pdb
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def collate_batch(features):
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    # 将features的label属性转换为labels, 以匹配模型的输入参数名称
    if hasattr(first, "label") and first.label is not None:
        if type(first.label) is int:
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        batch = {"labels": labels}
    elif hasattr(first, "label_ids") and first.label_ids is not None:
        if type(first.label_ids[0]) is int:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
    return batch
