import sys
import torch
from torch import Tensor
from typing import Tuple

import os
import numpy as np
import pandas as pd
import json
import pickle
import random
import hashlib
import logging
from datetime import datetime
import umap.plot
import matplotlib.pyplot as plt


metric_list = ['pc', 'rc', 'f1', 'prc', 'roc', 'apc', 'acc', 'mcc', 'trt', 'tst']

def dump_final_results(params, eval_results):
    benchmark_results = []
    timestamp = ['time']
    config_list = ['framework', 'datasource', 'dataset', 'session_size', 'window_size', 'stride', 'without_duplicate',
                   'semi_supervised', 'no_validation', 'embedding_type', 'encoder_type', 'kernel_sizes', 'pooling_mode',
                   'reconstruction_mode', 'masking_by', 'masking_ratio', 'loss_ablation', 'embedding_dim', 'hidden_size',
                   'alpha', 'beta', 'gamma', 'epoches', 'batch_size', 'learning_rate', 'patience']
    df_title = timestamp + config_list + metric_list

    benchmark_results.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    for k in config_list:
        if k in params.keys():
            benchmark_results.extend([params[k]])
        else:
            benchmark_results.extend(['-'])
    benchmark_results.extend([eval_results[k] for k in metric_list if k in eval_results.keys()])

    os.makedirs('./experiment_records', exist_ok=True)
    result_file_name = './experiment_records/logsd_benchmark_result.csv'
    if os.path.exists(result_file_name):
        df = pd.read_csv(result_file_name, encoding='utf-8-sig')
        df.loc[len(df)] = benchmark_results
        df.to_csv(result_file_name, index=False)
    else:
        pd.DataFrame([benchmark_results], columns=df_title).to_csv(result_file_name, index=False, encoding='utf-8-sig')

def dump_params(params):
    hash_id = hashlib.md5(
        str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")
    ).hexdigest()[0:8]
    params["hash_id"] = hash_id
    save_dir = os.path.join("./experiment_records", hash_id)
    os.makedirs(save_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(save_dir, "params.json"))

    log_file = os.path.join(save_dir, hash_id + ".log")
    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info(json.dumps(params, indent=4))
    return save_dir


def decision(probability):
    return random.random() < probability


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def tensor2flatten_arr(tensor):
    return tensor.data.cpu().numpy().reshape(-1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(gpu=-1):
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)

def new_mask_feature(x: Tensor,
                     p=-1.0,
                     mode: str = 'col',
                     fill_value: float = 0.,
                     mask_by: str = 'probability') -> Tuple[Tensor, Tensor]:
    assert mode in ['row', 'col', 'all']
    assert mask_by in ['probability', 'frequency']

    if p == -1.0:
        p_value = np.random.choice([0.05, 0.1, 0.15, 0.2], 1)[0]
    else:
        if p == 0.0:
            return x, torch.ones_like(x, dtype=torch.bool)
        else:
            p_value = p

    if mode == 'row':
        mask = torch.zeros(x.size(-2), dtype=torch.bool, device=x.device)
        mask[:int(p_value*x.size(-2))] = 1
        idx = torch.randperm(x.size(-2))
        mask = mask[idx]
        mask = mask.view(-1, 1).repeat(1, x.size(-1))
    elif mode == 'col':
        mask = torch.zeros(x.size(-1), dtype=torch.bool, device=x.device)
        mask[:int(p_value * x.size(-1))] = 1
        idx = torch.randperm(x.size(-1))
        mask = mask[idx]
        mask = mask.view(1, -1).repeat(x.size(-2), 1)
    else:
        mask = torch.zeros(x.size(-1) * x.size(-2), dtype=torch.bool, device=x.device)
        mask[:int(p_value * x.size(-1) * x.size(-2))] = 1
        idx = torch.randperm(x.size(-1) * x.size(-2))
        mask = torch.reshape(mask[idx], (x.size(-2), x.size(-1)))
    if fill_value == 0.0:
        x = x.masked_fill(~mask, fill_value)
    else:
        rnd_noise = torch.randn_like(x)
        rnd_noise = rnd_noise.masked_fill(~mask, 0)
        x = x + rnd_noise
    mask = mask.unsqueeze(0).repeat(x.size(0), 1, 1)
    return x, mask

def mask_vectors_in_batch_by_duplicate_node(x, p=-1.0, fill_value=0.0):
    # Find the duplicate vectors along the last dimension
    reshape_x = x.view(-1, x.size(-1))

    _, inv, counts = torch.unique(reshape_x, dim=-2, return_inverse=True, return_counts=True)

    if p == -1.0:
        p_value = int(np.random.choice([0.05, 0.1, 0.15, 0.2], 1)[0] * len(counts))
    else:
        if p == 0.0:
            return x, torch.ones_like(x, dtype=torch.bool)
        else:
            p_value = int(p * len(counts))

    p_value = p_value if p_value > 1 else 1

    idx = [torch.where(inv == i)[0] for i, c, in enumerate(counts) if counts[i] in torch.topk(counts, k=p_value, largest=False).values]
    indices = torch.concat(idx, dim=0)

    # Create a mask based on whether the vector's index is in the top-k indices
    mask = torch.ones_like(x, dtype=torch.bool)
    reshape_mask = mask.view(-1, mask.size(-1))
    reshape_mask[indices] = 0
    mask = reshape_mask.view(x.size(0), x.size(1), x.size(2))

    # Apply the mask to the tensor and set masked vectors to zero
    if fill_value == 0.0:
        masked_x = torch.where(~mask, torch.zeros_like(x), x)
    else:
        noise = torch.randn_like(x)
        masked_x = torch.where(~mask, x+noise, x)

    return masked_x, mask
