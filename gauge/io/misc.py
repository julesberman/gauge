import pickle
from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def save_pickle(output_path, data, verbose=True):
    try:
        with open(output_path, "wb") as outfile:
            pickle.dump(data, outfile)
            if verbose:
                print(f'result saved to {output_path.absolute()}')
    except Exception as e:
        print(e)
        if verbose:
            print(f"ERROR could not save to {output_path}")


def take_n(n, arr, axis=0):
    L = arr.shape[axis]
    if n >= L:
        return arr
    indices = jnp.linspace(0, L, n, dtype=jnp.int32)
    res = jnp.take(arr, indices, axis=axis)
    return res


def flatten_dataclass(dc, sep):
    flat = pd.json_normalize(asdict(dc), sep=sep)
    flat = flat.to_dict(orient='records')[0]
    return flat


def flatten_config(cfg, sep):
    d_cfg = OmegaConf.to_container(cfg)
    flat = pd.json_normalize(d_cfg, sep=sep)
    flat = flat.to_dict(orient='records')[0]
    return flat


def convert_list_to_numpy(dic: dict):
    for key, value in dic.items():
        if isinstance(value, list) and key != 'aux':
            dic[key] = np.array(value)
    return dic


def convert_jax_to_numpy(dic: dict):
    for key, value in dic.items():
        if isinstance(value, jnp.ndarray):
            dic[key] = np.array(value)
    return dic
