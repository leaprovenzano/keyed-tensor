from typing import Union

from functools import cached_property


class KeyedSize:
    """A simple keyed representation of size for KeyedTensors

    Note:
        KeyedSize objects should not be instantiated directly use the `size` method
        or `shape` property of `KeyedTensor` if you want one.

    Example:
        >>> from keyedtensor import KeyedTensor
        >>> import torch
        >>>
        >>> x = KeyedTensor(a=torch.rand(3, 4, 2), b=torch.rand(3, 1, 3), c=torch.rand(3, 1))
        >>> x.shape
        KeyedSize(a=torch.Size([3, 4, 2]), b=torch.Size([3, 1, 3]), c=torch.Size([3, 1]))

        you can index in a couple different ways -- for instance you can get the shape
        of a particular key with attribute access or use a string:

        >>> x.shape.a
        torch.Size([3, 4, 2])

        >>> x.shape['b']
        torch.Size([3, 1, 3])

        you can also index with ints or slices to get values across all keys:

        >>> x.shape[0]
        KeyedSize(a=3, b=3, c=3)

        >>> x.shape[:2]
        KeyedSize(a=torch.Size([3, 4]), b=torch.Size([3, 1]), c=torch.Size([3, 1]))
    """

    def __init__(self, keys, values):
        self._keys = tuple(keys)
        self._values = tuple(values)

    @cached_property
    def _key2idx(self):
        return {k: i for i, k in enumerate(self._keys)}

    def __getitem__(self, k: Union[str, int]):
        if isinstance(k, str):
            return self._values[self._key2idx[k]]
        return KeyedSize(self._keys, (v[k] for v in self._values))

    def __getattr__(self, k):
        if k in self._keys:
            return self._values[self._key2idx[k]]
        raise AttributeError(k)

    def keys(self):
        yield from self._keys

    def values(self):
        yield from self._values

    def items(self):
        yield from zip(self._keys, self._values)

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in self.items())})'

    def numel(self) -> 'KeyedSize':
        """like torch.Size.numel but for KeyedSizes, retuns one numel result per key.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> x = KeyedTensor(a=torch.rand(2, 3), b=torch.rand(2, 10))
            >>> x.shape
            KeyedSize(a=torch.Size([2, 3]), b=torch.Size([2, 10]))

            >>> x.shape.numel()
            KeyedSize(a=6, b=20)
        """

        return self.__class__(self.keys(), map(lambda x: x.numel(), self.values()))
