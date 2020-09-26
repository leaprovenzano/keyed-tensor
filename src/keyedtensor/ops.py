from typing import Callable, Sequence, Optional

from functools import partial
from operator import itemgetter

import torch

from keyedtensor import KeyedTensor


def _many_to_one(
    keyedtensors: Sequence[KeyedTensor], op: Callable, dim: Optional[int] = None
) -> KeyedTensor:
    keys = list(keyedtensors[0])
    getter = itemgetter(*keys)
    if dim is not None:
        op = partial(op, dim=dim)
    return KeyedTensor(zip(keys, map(op, zip(*map(getter, keyedtensors)))))


def cat(keyedtensors: Sequence[KeyedTensor], dim: Optional[int] = None) -> KeyedTensor:
    return _many_to_one(keyedtensors, torch.cat, dim=dim)


def stack(keyedtensors: Sequence[KeyedTensor], dim: Optional[int] = None) -> KeyedTensor:
    return _many_to_one(keyedtensors, torch.stack, dim=dim)
