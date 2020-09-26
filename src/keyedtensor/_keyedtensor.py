import sys
from typing import List, Union, Optional, Callable

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

import numpy as np

import torch

from collectionish import AttyDict

from keyedtensor._registry import TorchFuncRegistry
from keyedtensor._repr_utils import keyedtensor_str
from keyedtensor._keyedsize import KeyedSize

DimT = Union[Literal['keys'], int]
ValidOtherT = Union['KeyedTensor', int, float, bool, torch.Tensor, np.ndarray]


# patterns


def _self_apply_with_args(kt: 'KeyedTensor', op, *args, **kwargs):
    return kt._apply_out_of_place(lambda x: op(x, *args, **kwargs))


def _one_to_many(kt: 'KeyedTensor', op, *args, **kwargs) -> List['KeyedTensor']:
    return [
        kt.__class__(zip(kt.keys(), values))
        for values in zip(*map(lambda x: op(x, *args, **kwargs), kt.values()))
    ]


def _apply_with_other(kt: 'KeyedTensor', op, other: ValidOtherT, **kwargs) -> 'KeyedTensor':
    if isinstance(other, KeyedTensor):
        if sorted(kt) == sorted(other):
            return kt.__class__((k, op(v, other[k], **kwargs)) for k, v in kt.items())
        else:
            raise RuntimeError('cannot compare equality on KeyedTensors with different keys')
    elif isinstance(other, (float, torch.Tensor, np.ndarray, int, bool)):
        return kt.__class__(zip(kt.keys(), map(lambda x: op(x, other, **kwargs), kt.values())))
    return NotImplemented


def _r_apply_with_other(kt: 'KeyedTensor', op, other: ValidOtherT) -> 'KeyedTensor':
    if isinstance(other, (float, torch.Tensor, np.ndarray, int, bool)):
        return kt.__class__(zip(kt.keys(), map(lambda x: op(other, x), kt.values())))
    return NotImplemented


class KeyedTensor(AttyDict):

    torchfunc_registry: TorchFuncRegistry = TorchFuncRegistry()

    def _set_key(self, key: str, value):
        if key not in self:
            self._validate_key(key)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        dict.__setitem__(self, key, value)

    def __setitem__(self, key: Union[str, int, torch.Tensor, np.ndarray, slice], value):
        if isinstance(key, str):
            self._set_key(key, value)
        else:
            for v in self.values():
                v[key] = value

    def __getitem__(self, key: Union[str, int, torch.Tensor, np.ndarray, slice]):
        if isinstance(key, str):
            return dict.__getitem__(self, key)
        return self._apply_out_of_place(lambda x: x[key])

    def _apply_out_of_place(self, op):
        return self.__class__(zip(self.keys(), map(op, self.values())))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func not in self.torchfunc_registry:
            return NotImplemented
        kwargs = kwargs if kwargs is not None else {}
        return self.torchfunc_registry[func](*args, **kwargs)

    def _self_reduction(self, op: Callable, dim: Optional[DimT] = None, **kwargs):
        if dim is None:
            return op(torch.cat(list(map(torch.flatten, self.values()))), **kwargs)
        elif dim == 'key':
            return _self_apply_with_args(self, op, **kwargs)
        return _self_apply_with_args(self, op, dim=dim, **kwargs)

    def cuda(self, *args, **kwargs) -> 'KeyedTensor':
        return self._apply_out_of_place(lambda x: x.cuda(*args, **kwargs))

    @property
    def data(self) -> 'KeyedTensor':
        return self._apply_out_of_place(lambda x: x.data)

    def dim(self):
        return self._apply_out_of_place(lambda x: x.dim)

    def size(self) -> KeyedSize:
        return KeyedSize(self.keys(), map(lambda x: x.shape, self.values()))

    @property
    def shape(self) -> KeyedSize:
        """return a KeyedSize for this KeyedTensor

        see `keyedtensor.KeyedSize` docs for more details on working with `KeyedSize`.

        Example:
            >>> from keyedtensor import KeyedTensor
            >>> import torch
            >>>
            >>> x = KeyedTensor(a=torch.rand(3, 4), b=torch.rand(3, 1), c=torch.rand(3))
            >>> x.size()
            KeyedSize(a=torch.Size([3, 4]), b=torch.Size([3, 1]), c=torch.Size([3]))
        """
        return self.size()

    # self magics

    def __repr__(self) -> str:
        return keyedtensor_str(self)

    def __abs__(self):
        return self.abs()

    def __all__(self):
        return self.all()

    def __any__(self):
        return self.any()

    def __neg__(self) -> 'KeyedTensor':
        return self.neg()

    # --------------------------self redictions -------------------------------------
    @torchfunc_registry.register(torch.all)
    def all(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.all but for keyed tensor. dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.
        """
        return self._self_reduction(torch.all, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.any)
    def any(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.any but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.
        """
        return self._self_reduction(torch.any, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.mean)
    def mean(self, dim: Optional[DimT] = None, **kwargs):
        """Like `torch.mean` but for keyed tensor.

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt
            KeyedTensor(a=tensor([[0.4963, 0.7682, 0.0885],
                                  [0.1320, 0.3074, 0.6341],
                                  [0.4901, 0.8964, 0.4556]]),
                        b=tensor([0.6323, 0.3489, 0.4017]))

            with dim unspecified we apply `mean` over all values in the `KeyedTensor`:

            >>> kt.mean()
            tensor(0.4710)

            specify a numeric dim to apply `mean` along each keyed tensor:

            >>> kt.mean(dim=-1)
            KeyedTensor(a=tensor([0.4510, 0.3578, 0.6141]), b=tensor(0.4610))

            or specify `key` to apply `mean` per keyed tensor:

            >>> kt.mean(dim='key')
            KeyedTensor(a=tensor(0.4743), b=tensor(0.4610))

            calling torch.mean on a KeyedTensor works as expected:

            >>> torch.mean(kt, dim='key')
            KeyedTensor(a=tensor(0.4743), b=tensor(0.4610))

        """
        return self._self_reduction(torch.mean, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.sum)
    def sum(self, dim: Optional[DimT] = None, **kwargs):
        """Like `torch.sum` but for keyed tensor.

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt
            KeyedTensor(a=tensor([[0.4963, 0.7682, 0.0885],
                                  [0.1320, 0.3074, 0.6341],
                                  [0.4901, 0.8964, 0.4556]]),
                        b=tensor([0.6323, 0.3489, 0.4017]))

            with dim unspecified we apply `sum` over all values in the `KeyedTensor`:

            >>> kt.sum()
            tensor(5.6516)

            specify a numeric dim to apply `sum` along each keyed tensor:

            >>> kt.sum(dim=-1)
            KeyedTensor(a=tensor([1.3530, 1.0735, 1.8422]), b=tensor(1.3829))

            or specify `key` to apply `sum` per keyed tensor:

            >>> kt.sum(dim='key')
            KeyedTensor(a=tensor(4.2687), b=tensor(1.3829))

            calling torch.sum on a KeyedTensor works as expected:

            >>> torch.sum(kt, dim='key')
            KeyedTensor(a=tensor(4.2687), b=tensor(1.3829))
        """
        return self._self_reduction(torch.sum, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.var)
    def var(self, dim: Optional[DimT] = None, **kwargs):
        """Like `torch.var` but for keyed tensor.

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt
            KeyedTensor(a=tensor([[0.4963, 0.7682, 0.0885],
                                  [0.1320, 0.3074, 0.6341],
                                  [0.4901, 0.8964, 0.4556]]),
                        b=tensor([0.6323, 0.3489, 0.4017]))

            with dim unspecified we apply `var` over all values in the `KeyedTensor`:

            >>> kt.var()
            tensor(0.0574)

            specify a numeric dim to apply `var` along each keyed tensor:

            >>> kt.var(dim=-1)
            KeyedTensor(a=tensor([0.1171, 0.0649, 0.0601]), b=tensor(0.0227))

            or specify `key` to apply `var` per keyed tensor:

            >>> kt.var(dim='key')
            KeyedTensor(a=tensor(0.0731), b=tensor(0.0227))

            calling torch.var on a KeyedTensor works as expected:

            >>> torch.var(kt, dim='key')
            KeyedTensor(a=tensor(0.0731), b=tensor(0.0227))
        """
        return self._self_reduction(torch.var, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.argmax)
    def argmax(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.argmax but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.argmax(dim=-1)
            KeyedTensor(a=tensor([1, 2, 1]), b=tensor(0))

            >>> kt.argmax(dim='key')
            KeyedTensor(a=tensor(7), b=tensor(0))
        """
        return self._self_reduction(torch.argmax, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.argmin)
    def argmin(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.argmin but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.argmin(dim=-1)
            KeyedTensor(a=tensor([2, 0, 2]), b=tensor(1))

            >>> kt.argmin(dim='key')
            KeyedTensor(a=tensor(2), b=tensor(1))
        """
        return self._self_reduction(torch.argmin, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.std)
    def std(self, dim: Optional[DimT] = None, **kwargs):
        """Like `torch.std` but for keyed tensor.

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt
            KeyedTensor(a=tensor([[0.4963, 0.7682, 0.0885],
                                  [0.1320, 0.3074, 0.6341],
                                  [0.4901, 0.8964, 0.4556]]),
                        b=tensor([0.6323, 0.3489, 0.4017]))

            with dim unspecified we apply `std` over all values in the `KeyedTensor`:

            >>> kt.std()
            tensor(0.2395)

            specify a numeric dim to apply `std` along each keyed tensor:

            >>> kt.std(dim=-1)
            KeyedTensor(a=tensor([0.3421, 0.2548, 0.2452]), b=tensor(0.1507))

            or specify `key` to apply `std` per keyed tensor:

            >>> kt.std(dim='key')
            KeyedTensor(a=tensor(0.2704), b=tensor(0.1507))

            calling torch.std on a KeyedTensor works as expected:

            >>> torch.std(kt, dim='key')
            KeyedTensor(a=tensor(0.2704), b=tensor(0.1507))
        """
        return self._self_reduction(torch.std, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.norm)
    def norm(self, p='fro', dim: Optional[DimT] = None, **kwargs):
        """Like torch.norm but for keyed tensor, dim may optionally be a keyed

        Args:
            p: norm type (see torch.norm for full details. Defaults to 'fro'.
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>> _ = torch.manual_seed(0)
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.norm()
            tensor(1.8145)

            >>> kt.norm(dim=-1)
            KeyedTensor(a=tensor([0.9188, 0.7169, 1.1187]), b=tensor(0.8264))

            >>> kt.norm(dim='key')
            KeyedTensor(a=tensor(1.6154), b=tensor(0.8264))

            >>> kt.norm(p=1, dim='key')
            KeyedTensor(a=tensor(4.2687), b=tensor(1.3829))
        """
        return self._self_reduction(torch.norm, dim=dim, **kwargs, p=p)

    @torchfunc_registry.register(torch.prod)
    def prod(self, dim: Optional[DimT] = None, **kwargs):
        """Like `torch.prod` but for keyed tensor.

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt
            KeyedTensor(a=tensor([[0.4963, 0.7682, 0.0885],
                                  [0.1320, 0.3074, 0.6341],
                                  [0.4901, 0.8964, 0.4556]]),
                        b=tensor([0.6323, 0.3489, 0.4017]))

            with dim unspecified we apply `prod` over all values in the `KeyedTensor`:

            >>> kt.prod()
            tensor(1.5400e-05)

            specify a numeric dim to apply `prod` along each keyed tensor:

            >>> kt.prod(dim=-1)
            KeyedTensor(a=tensor([0.0337, 0.0257, 0.2002]), b=tensor(0.0886))

            or specify `key` to apply `prod` per keyed tensor:

            >>> kt.prod(dim='key')
            KeyedTensor(a=tensor(0.0002), b=tensor(0.0886))

            calling torch.prod on a KeyedTensor works as expected:

            >>> torch.prod(kt, dim='key')
            KeyedTensor(a=tensor(0.0002), b=tensor(0.0886))
        """
        return self._self_reduction(torch.prod, dim=dim, **kwargs)

    # --------------------------one to many -------------------------------------

    @torchfunc_registry.register(torch.unbind)
    def unbind(self) -> List['KeyedTensor']:
        return _one_to_many(self, torch.unbind)

    @torchfunc_registry.register(torch.chunk)
    def chunk(self, chunks: int, dim=0) -> List['KeyedTensor']:
        return _one_to_many(self, torch.chunk, chunks, dim=dim)

    @torchfunc_registry.register(torch.split)
    def split(self, split_size_or_sections, dim=0) -> List['KeyedTensor']:
        return _one_to_many(self, torch.split, split_size_or_sections, dim=dim)

    # -------------------------- simple self apply ops -------------------------------------
    @torchfunc_registry.register(torch.abs)
    def abs(self) -> 'KeyedTensor':
        """Like `torch.abs` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.abs()
            KeyedTensor(a=tensor([0.0075, 0.5364, 0.8230]), b=tensor([0.7359, 0.3852]))

            >>> torch.abs(kt)
            KeyedTensor(a=tensor([0.0075, 0.5364, 0.8230]), b=tensor([0.7359, 0.3852]))
        """
        return self._apply_out_of_place(torch.abs)

    @torchfunc_registry.register(torch.acos)
    def acos(self) -> 'KeyedTensor':
        """Like `torch.acos` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.acos()
            KeyedTensor(a=tensor([1.5783, 1.0046, 2.5375]), b=tensor([2.3978, 1.9662]))

            >>> torch.acos(kt)
            KeyedTensor(a=tensor([1.5783, 1.0046, 2.5375]), b=tensor([2.3978, 1.9662]))
        """
        return self._apply_out_of_place(torch.acos)

    @torchfunc_registry.register(torch.asin)
    def asin(self) -> 'KeyedTensor':
        """Like `torch.asin` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.asin()
            KeyedTensor(a=tensor([-0.0075,  0.5662, -0.9668]), b=tensor([-0.8271, -0.3954]))

            >>> torch.asin(kt)
            KeyedTensor(a=tensor([-0.0075,  0.5662, -0.9668]), b=tensor([-0.8271, -0.3954]))
        """
        return self._apply_out_of_place(torch.asin)

    @torchfunc_registry.register(torch.atan)
    def atan(self) -> 'KeyedTensor':
        """Like `torch.atan` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.atan()
            KeyedTensor(a=tensor([-0.0075,  0.4924, -0.6886]), b=tensor([-0.6344, -0.3676]))

            >>> torch.atan(kt)
            KeyedTensor(a=tensor([-0.0075,  0.4924, -0.6886]), b=tensor([-0.6344, -0.3676]))
        """
        return self._apply_out_of_place(torch.atan)

    @torchfunc_registry.register(torch.ceil)
    def ceil(self) -> 'KeyedTensor':
        """Like `torch.ceil` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.ceil()
            KeyedTensor(a=tensor([-0., 1., -0.]), b=tensor([-0., -0.]))

            >>> torch.ceil(kt)
            KeyedTensor(a=tensor([-0., 1., -0.]), b=tensor([-0., -0.]))
        """
        return self._apply_out_of_place(torch.ceil)

    @torchfunc_registry.register(torch.cos)
    def cos(self) -> 'KeyedTensor':
        """Like `torch.cos` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.cos()
            KeyedTensor(a=tensor([1.0000, 0.8595, 0.6800]), b=tensor([0.7412, 0.9267]))

            >>> torch.cos(kt)
            KeyedTensor(a=tensor([1.0000, 0.8595, 0.6800]), b=tensor([0.7412, 0.9267]))
        """
        return self._apply_out_of_place(torch.cos)

    @torchfunc_registry.register(torch.cosh)
    def cosh(self) -> 'KeyedTensor':
        """Like `torch.cosh` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.cosh()
            KeyedTensor(a=tensor([1.0000, 1.1474, 1.3583]), b=tensor([1.2832, 1.0751]))

            >>> torch.cosh(kt)
            KeyedTensor(a=tensor([1.0000, 1.1474, 1.3583]), b=tensor([1.2832, 1.0751]))
        """
        return self._apply_out_of_place(torch.cosh)

    @torchfunc_registry.register(torch.detach)
    def detach(self) -> 'KeyedTensor':
        """Like `torch.detach` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.detach()
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> torch.detach(kt)
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))
        """
        return self._apply_out_of_place(torch.detach)

    @torchfunc_registry.register(torch.digamma)
    def digamma(self) -> 'KeyedTensor':
        """Like `torch.digamma` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.digamma()
            KeyedTensor(a=tensor([132.9785,  -1.7941,  -4.7548]), b=tensor([-2.6389,  1.1087]))

            >>> torch.digamma(kt)
            KeyedTensor(a=tensor([132.9785,  -1.7941,  -4.7548]), b=tensor([-2.6389,  1.1087]))
        """
        return self._apply_out_of_place(torch.digamma)

    @torchfunc_registry.register(torch.erf)
    def erf(self) -> 'KeyedTensor':
        """Like `torch.erf` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.erf()
            KeyedTensor(a=tensor([-0.0084,  0.5519, -0.7556]), b=tensor([-0.7020, -0.4140]))

            >>> torch.erf(kt)
            KeyedTensor(a=tensor([-0.0084,  0.5519, -0.7556]), b=tensor([-0.7020, -0.4140]))
        """
        return self._apply_out_of_place(torch.erf)

    @torchfunc_registry.register(torch.erfc)
    def erfc(self) -> 'KeyedTensor':
        """Like `torch.erfc` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.erfc()
            KeyedTensor(a=tensor([1.0084, 0.4481, 1.7556]), b=tensor([1.7020, 1.4140]))

            >>> torch.erfc(kt)
            KeyedTensor(a=tensor([1.0084, 0.4481, 1.7556]), b=tensor([1.7020, 1.4140]))
        """
        return self._apply_out_of_place(torch.erfc)

    @torchfunc_registry.register(torch.erfinv)
    def erfinv(self) -> 'KeyedTensor':
        """Like `torch.erfinv` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.erfinv()
            KeyedTensor(a=tensor([-0.0066,  0.5183, -0.9547]), b=tensor([-0.7897, -0.3558]))

            >>> torch.erfinv(kt)
            KeyedTensor(a=tensor([-0.0066,  0.5183, -0.9547]), b=tensor([-0.7897, -0.3558]))
        """
        return self._apply_out_of_place(torch.erfinv)

    @torchfunc_registry.register(torch.exp)
    def exp(self) -> 'KeyedTensor':
        """Like `torch.exp` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.exp()
            KeyedTensor(a=tensor([0.9925, 1.7099, 0.4391]), b=tensor([0.4791, 0.6803]))

            >>> torch.exp(kt)
            KeyedTensor(a=tensor([0.9925, 1.7099, 0.4391]), b=tensor([0.4791, 0.6803]))
        """
        return self._apply_out_of_place(torch.exp)

    @torchfunc_registry.register(torch.expm1)
    def expm1(self) -> 'KeyedTensor':
        """Like `torch.expm1` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.expm1()
            KeyedTensor(a=tensor([-0.0075,  0.7099, -0.5609]), b=tensor([-0.5209, -0.3197]))

            >>> torch.expm1(kt)
            KeyedTensor(a=tensor([-0.0075,  0.7099, -0.5609]), b=tensor([-0.5209, -0.3197]))
        """
        return self._apply_out_of_place(torch.expm1)

    @torchfunc_registry.register(torch.floor)
    def floor(self) -> 'KeyedTensor':
        """Like `torch.floor` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.floor()
            KeyedTensor(a=tensor([-1.,  0., -1.]), b=tensor([-1., -1.]))

            >>> torch.floor(kt)
            KeyedTensor(a=tensor([-1.,  0., -1.]), b=tensor([-1., -1.]))
        """
        return self._apply_out_of_place(torch.floor)

    @torchfunc_registry.register(torch.frac)
    def frac(self) -> 'KeyedTensor':
        """Like `torch.frac` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.frac()
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> torch.frac(kt)
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))
        """
        return self._apply_out_of_place(torch.frac)

    @torchfunc_registry.register(torch.log)
    def log(self) -> 'KeyedTensor':
        """Like `torch.log` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2))
            >>> kt
            KeyedTensor(a=tensor([0.4963, 0.7682, 0.0885]), b=tensor([0.1320, 0.3074]))

            >>> kt.log()
            KeyedTensor(a=tensor([-0.7007, -0.2637, -2.4250]), b=tensor([-2.0247, -1.1795]))

            >>> torch.log(kt)
            KeyedTensor(a=tensor([-0.7007, -0.2637, -2.4250]), b=tensor([-2.0247, -1.1795]))
        """
        return self._apply_out_of_place(torch.log)

    @torchfunc_registry.register(torch.isnan)
    def isnan(self) -> 'KeyedTensor':
        """Like `torch.isnan` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = torch.sqrt(KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1)
            >>> kt
            KeyedTensor(a=tensor([   nan, 0.7324,    nan]), b=tensor([nan, nan]))

            >>> kt.isnan()
            KeyedTensor(a=tensor([ True, False,  True]), b=tensor([True, True]))

            >>> torch.isnan(kt)
            KeyedTensor(a=tensor([ True, False,  True]), b=tensor([True, True]))
        """
        return self._apply_out_of_place(torch.isnan)

    @torchfunc_registry.register(torch.isfinite)
    def isfinite(self) -> 'KeyedTensor':
        """Like `torch.isfinite` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=[float('inf'),  0.5364, -0.8230], b=[-0.7359, -0.3852])
            >>> kt
            KeyedTensor(a=tensor([    inf,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.isinf()
            KeyedTensor(a=tensor([ True, False, False]), b=tensor([False, False]))

            >>> torch.isinf(kt)
            KeyedTensor(a=tensor([ True, False, False]), b=tensor([False, False]))
        """
        return self._apply_out_of_place(torch.isfinite)

    @torchfunc_registry.register(torch.isinf)
    def isinf(self) -> 'KeyedTensor':
        """Like `torch.isinf` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=[float('inf'),  0.5364, -0.8230], b=[-0.7359, -0.3852])
            >>> kt
            KeyedTensor(a=tensor([    inf,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.isinf()
            KeyedTensor(a=tensor([ True, False, False]), b=tensor([False, False]))

            >>> torch.isinf(kt)
            KeyedTensor(a=tensor([ True, False, False]), b=tensor([False, False]))
        """
        return self._apply_out_of_place(torch.isinf)

    @torchfunc_registry.register(torch.lgamma)
    def lgamma(self) -> 'KeyedTensor':
        """Like `torch.lgamma` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.lgamma()
            KeyedTensor(a=tensor([4.8990, 0.5040, 1.8482]), b=tensor([1.5368, 1.3299]))

            >>> torch.lgamma(kt)
            KeyedTensor(a=tensor([4.8990, 0.5040, 1.8482]), b=tensor([1.5368, 1.3299]))
        """
        return self._apply_out_of_place(torch.lgamma)

    @torchfunc_registry.register(torch.log10)
    def log10(self) -> 'KeyedTensor':
        """Like `torch.log10` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2))
            >>> kt
            KeyedTensor(a=tensor([0.4963, 0.7682, 0.0885]), b=tensor([0.1320, 0.3074]))

            >>> kt.log10()
            KeyedTensor(a=tensor([-0.3043, -0.1145, -1.0532]), b=tensor([-0.8793, -0.5123]))

            >>> torch.log10(kt)
            KeyedTensor(a=tensor([-0.3043, -0.1145, -1.0532]), b=tensor([-0.8793, -0.5123]))
        """
        return self._apply_out_of_place(torch.log10)

    @torchfunc_registry.register(torch.neg)
    def neg(self) -> 'KeyedTensor':
        """Like `torch.neg` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.neg()
            KeyedTensor(a=tensor([ 0.0075, -0.5364,  0.8230]), b=tensor([0.7359, 0.3852]))

            >>> torch.neg(kt)
            KeyedTensor(a=tensor([ 0.0075, -0.5364,  0.8230]), b=tensor([0.7359, 0.3852]))
        """
        return self._apply_out_of_place(torch.neg)

    @torchfunc_registry.register(torch.numel)
    def numel(self) -> 'KeyedTensor':
        """Like `torch.numel` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.numel()
            KeyedTensor(a=tensor(3), b=tensor(2))

            >>> torch.numel(kt)
            KeyedTensor(a=tensor(3), b=tensor(2))
        """
        return self._apply_out_of_place(torch.numel)

    @torchfunc_registry.register(torch.reciprocal)
    def reciprocal(self) -> 'KeyedTensor':
        """Like `torch.reciprocal` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.reciprocal()
            KeyedTensor(a=tensor([-133.5681,    1.8641,   -1.2150]), b=tensor([-1.3588, -2.5964]))

            >>> torch.reciprocal(kt)
            KeyedTensor(a=tensor([-133.5681,    1.8641,   -1.2150]), b=tensor([-1.3588, -2.5964]))
        """
        return self._apply_out_of_place(torch.reciprocal)

    @torchfunc_registry.register(torch.relu)
    def relu(self) -> 'KeyedTensor':
        """Like `torch.relu` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.relu()
            KeyedTensor(a=tensor([0.0000, 0.5364, 0.0000]), b=tensor([0., 0.]))

            >>> torch.relu(kt)
            KeyedTensor(a=tensor([0.0000, 0.5364, 0.0000]), b=tensor([0., 0.]))
        """
        return self._apply_out_of_place(torch.relu)

    @torchfunc_registry.register(torch.round)
    def round(self) -> 'KeyedTensor':
        """Like `torch.round` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.round()
            KeyedTensor(a=tensor([-0.,  1., -1.]), b=tensor([-1., -0.]))

            >>> torch.round(kt)
            KeyedTensor(a=tensor([-0.,  1., -1.]), b=tensor([-1., -0.]))
        """
        return self._apply_out_of_place(torch.round)

    @torchfunc_registry.register(torch.rsqrt)
    def rsqrt(self) -> 'KeyedTensor':
        """Like `torch.rsqrt` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2))
            >>> kt
            KeyedTensor(a=tensor([0.4963, 0.7682, 0.0885]), b=tensor([0.1320, 0.3074]))

            >>> kt.rsqrt()
            KeyedTensor(a=tensor([1.4195, 1.1409, 3.3619]), b=tensor([2.7521, 1.8036]))

            >>> torch.rsqrt(kt)
            KeyedTensor(a=tensor([1.4195, 1.1409, 3.3619]), b=tensor([2.7521, 1.8036]))
        """
        return self._apply_out_of_place(torch.rsqrt)

    @torchfunc_registry.register(torch.sigmoid)
    def sigmoid(self) -> 'KeyedTensor':
        """Like `torch.sigmoid` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.sigmoid()
            KeyedTensor(a=tensor([0.4981, 0.6310, 0.3051]), b=tensor([0.3239, 0.4049]))

            >>> torch.sigmoid(kt)
            KeyedTensor(a=tensor([0.4981, 0.6310, 0.3051]), b=tensor([0.3239, 0.4049]))
        """
        return self._apply_out_of_place(torch.sigmoid)

    @torchfunc_registry.register(torch.sign)
    def sign(self) -> 'KeyedTensor':
        """Like `torch.sign` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.sign()
            KeyedTensor(a=tensor([-1.,  1., -1.]), b=tensor([-1., -1.]))

            >>> torch.sign(kt)
            KeyedTensor(a=tensor([-1.,  1., -1.]), b=tensor([-1., -1.]))
        """
        return self._apply_out_of_place(torch.sign)

    @torchfunc_registry.register(torch.sin)
    def sin(self) -> 'KeyedTensor':
        """Like `torch.sin` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.sin()
            KeyedTensor(a=tensor([-0.0075,  0.5111, -0.7332]), b=tensor([-0.6713, -0.3757]))

            >>> torch.sin(kt)
            KeyedTensor(a=tensor([-0.0075,  0.5111, -0.7332]), b=tensor([-0.6713, -0.3757]))
        """
        return self._apply_out_of_place(torch.sin)

    @torchfunc_registry.register(torch.sinh)
    def sinh(self) -> 'KeyedTensor':
        """Like `torch.sinh` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.sinh()
            KeyedTensor(a=tensor([-0.0075,  0.5625, -0.9192]), b=tensor([-0.8042, -0.3947]))

            >>> torch.sinh(kt)
            KeyedTensor(a=tensor([-0.0075,  0.5625, -0.9192]), b=tensor([-0.8042, -0.3947]))
        """
        return self._apply_out_of_place(torch.sinh)

    @torchfunc_registry.register(torch.sqrt)
    def sqrt(self) -> 'KeyedTensor':
        """Like `torch.sqrt` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2))
            >>> kt
            KeyedTensor(a=tensor([0.4963, 0.7682, 0.0885]), b=tensor([0.1320, 0.3074]))

            >>> kt.sqrt()
            KeyedTensor(a=tensor([0.7045, 0.8765, 0.2975]), b=tensor([0.3634, 0.5545]))

            >>> torch.sqrt(kt)
            KeyedTensor(a=tensor([0.7045, 0.8765, 0.2975]), b=tensor([0.3634, 0.5545]))
        """
        return self._apply_out_of_place(torch.sqrt)

    @torchfunc_registry.register(torch.square)
    def square(self) -> 'KeyedTensor':
        """Like `torch.square` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.square()
            KeyedTensor(a=tensor([5.6052e-05, 2.8777e-01, 6.7740e-01]), b=tensor([0.5416, 0.1483]))

            >>> torch.square(kt)
            KeyedTensor(a=tensor([5.6052e-05, 2.8777e-01, 6.7740e-01]), b=tensor([0.5416, 0.1483]))
        """
        return self._apply_out_of_place(torch.square)

    @torchfunc_registry.register(torch.trunc)
    def trunc(self) -> 'KeyedTensor':
        """Like `torch.trunc` but for keyed tensor.

        Example:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> kt = KeyedTensor(a=torch.rand(3), b=torch.rand(2)) * 2 - 1
            >>> kt
            KeyedTensor(a=tensor([-0.0075,  0.5364, -0.8230]), b=tensor([-0.7359, -0.3852]))

            >>> kt.trunc()
            KeyedTensor(a=tensor([-0., 0., -0.]), b=tensor([-0., -0.]))

            >>> torch.trunc(kt)
            KeyedTensor(a=tensor([-0., 0., -0.]), b=tensor([-0., -0.]))
        """
        return self._apply_out_of_place(torch.trunc)

    # -------------------------- self apply ops w/ args -------------------------------------

    @torchfunc_registry.register(torch.hardshrink)
    def hardshrink(self, *args, **kwargs) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.hardshrink, *args, **kwargs)

    @torchfunc_registry.register(torch.polygamma)
    def polygamma(self, *args, **kwargs):
        return _self_apply_with_args(self, torch.polygamma, *args, **kwargs)

    @torchfunc_registry.register(torch.t)
    def t(self, *args, **kwargs):
        return self._self_apply_with_args(torch.t, *args, **kwargs)

    @torchfunc_registry.register(torch.tan)
    def tan(self, *args, **kwargs):
        return self._apply_out_of_place(torch.tan, *args, **kwargs)

    @torchfunc_registry.register(torch.tanh)
    def tanh(self, *args, **kwargs):
        return self._apply_out_of_place(torch.tanh, *args, **kwargs)

    def to(self, *args, **kwargs):
        return self._apply_out_of_place(lambda x: x.to(*args, **kwargs))

    @torchfunc_registry.register(torch.squeeze)
    def squeeze(self, *args, **kwargs) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.squeeze, *args, **kwargs)

    @torchfunc_registry.register(torch.transpose)
    def transpose(self, dim0, dim1) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.transpose, dim0, dim1)

    @torchfunc_registry.register(torch.unsqueeze)
    def unsqueeze(self, dim) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.unsqueeze, dim)

    # -------------------------- self/ & comparison other ops  -------------------------------------

    @torchfunc_registry.register(torch.add)
    def add(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.add, other)

    @torchfunc_registry.register(torch.sub)
    def sub(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.sub, other)

    @torchfunc_registry.register(torch.mul)
    def mul(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.mul, other)

    @torchfunc_registry.register(torch.div)
    def div(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.div, other)

    @torchfunc_registry.register(torch.true_divide)
    def true_divide(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.true_divide, other)

    @torchfunc_registry.register(torch.pow)
    def pow(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.pow, other)

    @torchfunc_registry.register(torch.eq)
    def eq(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.eq, other)

    @torchfunc_registry.register(torch.ge)
    def ge(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.ge, other)

    @torchfunc_registry.register(torch.eq)
    def lt(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.lt, other)

    @torchfunc_registry.register(torch.ge)
    def le(self, other: ValidOtherT) -> 'KeyedTensor':
        return _apply_with_other(self, torch.le, other)

    # -------------------------- self/ & comparison other magics -----------------------------------

    def __pow__(self, other):
        return _r_apply_with_other()

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.true_divide(other)

    def __sub__(self, other):
        return self.sub(other)

    def __ipow__(self, other):
        return self.pow(other)

    def __iadd__(self, other):
        return self.add(other)

    def __imul__(self, other):
        return self.mul(other)

    def __itruediv__(self, other):
        return self.true_divide(other)

    def __isub__(self, other):
        return self.sub(other)

    def __rpow__(self, other):
        return _r_apply_with_other(self, torch.pow, other)

    def __radd__(self, other):
        return _r_apply_with_other(self, torch.add, other)

    def __rmul__(self, other):
        return _r_apply_with_other(self, torch.mul, other)

    def __rtruediv__(self, other):
        return _r_apply_with_other(self, torch.true_divide, other)

    def __rsub__(self, other):
        return _r_apply_with_other(self, torch.sub, other)

    def __eq__(self, other):
        return _apply_with_other(self, torch.eq, other)

    def __req__(self, other):
        return _r_apply_with_other(self, torch.eq, other)

    def __lt__(self, other):
        return _apply_with_other(self, torch.lt, other)

    def __rlt__(self, other):
        return _r_apply_with_other(self, torch.lt, other)

    def __le__(self, other):
        return _apply_with_other(self, torch.le, other)

    def __rle__(self, other):
        return _r_apply_with_other(self, torch.le, other)

    def __gt__(self, other):
        return _apply_with_other(self, torch.gt, other)

    def __rgt__(self, other):
        return _r_apply_with_other(self, torch.gt, other)

    def __ge__(self, other):
        return _apply_with_other(self, torch.ge, other)

    def __rge__(self, other):
        return _r_apply_with_other(self, torch.ge, other)
