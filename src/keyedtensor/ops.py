from typing import Callable, Sequence

from functools import partial
from operator import itemgetter

import torch

from keyedtensor import KeyedTensor


def _many_to_one(keyedtensors: Sequence[KeyedTensor], op: Callable, dim: int) -> KeyedTensor:
    keys = list(keyedtensors[0])
    getter = itemgetter(*keys)
    op = partial(op, dim=dim)
    return KeyedTensor(zip(keys, map(op, zip(*map(getter, keyedtensors)))))


def cat(keyedtensors: Sequence[KeyedTensor], dim: int = 0) -> KeyedTensor:
    """like torch.cat but for KeyedTensors, concatenates a sequence of KeyedTensor along existing\
    dimension.

    Args:
        keyedtensors: a sequence of KeyedTensors. should all have the same keys (though they may be\
         differently ordered) and shapes should be alignable just as they
         would need to be with `torch.cat`.
        dim: integer dimension to concatenate on dim should. Defaults to 0.

    Example:
        >>> import torch
        >>>
        >>> import keyedtensor as kt
        >>> from keyedtensor import KeyedTensor
        >>>
        >>> x1 = KeyedTensor(a=torch.ones(2, 3), b=torch.ones(2))
        >>> x2 = KeyedTensor(a=torch.ones(3, 3), b=torch.ones(3)) * 2
        >>> x3 = KeyedTensor(b=torch.ones(1), a=torch.ones(1, 3)) * 3
        >>>
        >>> kt.cat((x1, x2, x3), dim=0).to(torch.int64)
        KeyedTensor(a=tensor([[1, 1, 1],
                              [1, 1, 1],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [3, 3, 3]]),
                    b=tensor([1, 1, 2, 2, 2, 3]))
    """
    return _many_to_one(keyedtensors, torch.cat, dim=dim)


def stack(keyedtensors: Sequence[KeyedTensor], dim: int = 0) -> KeyedTensor:
    """like torch.stack but for KeyedTensors, stacks a sequence of KeyedTensor along existing\
    dimension.

    Args:
        keyedtensors: a sequence of KeyedTensors. should all have the same keys (though they may be\
         differently ordered) and shapes should be the same just as they
         would need to be with `torch.stack`.
        dim: integer dimension to stack on dim should. Defaults to 0.

    Example:
        >>> import torch
        >>>
        >>> import keyedtensor as kt
        >>> from keyedtensor import KeyedTensor
        >>>
        >>> x1 = KeyedTensor(a=torch.ones(3), b=torch.ones(2))
        >>> x2 = KeyedTensor(a=torch.ones(3), b=torch.ones(2)) * 2
        >>> x3 = KeyedTensor(b=torch.ones(2), a=torch.ones(3)) * 3
        >>>
        >>> kt.stack((x1, x2, x3), dim=0).to(torch.int64)
        KeyedTensor(a=tensor([[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]]),
                    b=tensor([[1, 1],
                              [2, 2],
                              [3, 3]]))
    """
    return _many_to_one(keyedtensors, torch.stack, dim=dim)
