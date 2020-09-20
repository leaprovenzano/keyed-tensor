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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({", ".join(f"{k}={t!r}" for k, t in self.items())})'

    def __abs__(self):
        return self.abs()

    def __all__(self):
        return self.all()

    def __any__(self):
        return self.any()

    def __neg__(self) -> 'KeyedTensor':
        return self.neg()

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
        """Like torch.mean but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.mean()
            tensor(0.4710)

            >>> print(kt.mean(dim=-1))
            {'a': tensor([0.4510, 0.3578, 0.6141]), 'b': tensor(0.4610)}

            >>> kt.mean(dim='key')
            {'a': tensor(0.4743), 'b': tensor(0.4610)}
        """
        return self._self_reduction(torch.mean, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.sum)
    def sum(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.sum but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.sum()
            tensor(5.6516)

            >>> kt.sum(dim=-1)
            {'a': tensor([1.3530, 1.0735, 1.8422]), 'b': tensor(1.3829)}

            >>> kt.sum(dim='key')
            {'a': tensor(4.2687), 'b': tensor(1.3829)}
        """
        return self._self_reduction(torch.sum, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.var)
    def var(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.var but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.var()
            tensor(0.0574)

            >>> kt.var(dim=-1)
            {'a': tensor([0.1171, 0.0649, 0.0601]), 'b': tensor(0.0227)}

            >>> kt.var(dim='key')
            {'a': tensor(0.0731), 'b': tensor(0.0227)}
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
            {'a': tensor([1, 2, 1]), 'b': tensor(0)}

            >>> kt.argmax(dim='key')
            {'a': tensor(7), 'b': tensor(0)}
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
            {'a': tensor([2, 0, 2]), 'b': tensor(1)}

            >>> kt.argmin(dim='key')
            {'a': tensor(2), 'b': tensor(1)}
        """
        return self._self_reduction(torch.argmin, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.std)
    def std(self, dim: Optional[DimT] = None, **kwargs):
        """Like torch.std but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.std()
            tensor(0.2395)

            >>> kt.std(dim=-1)
            {'a': tensor([0.3421, 0.2548, 0.2452]), 'b': tensor(0.1507)}

            >>> kt.std(dim='key')
            {'a': tensor(0.2704), 'b': tensor(0.1507)}
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
            {'a': tensor([0.9188, 0.7169, 1.1187]), 'b': tensor(0.8264)}

            >>> kt.norm(dim='key')
            {'a': tensor(1.6154), 'b': tensor(0.8264)}

            >>> kt.norm(p=1, dim='key')
            {'a': tensor(4.2687), 'b': tensor(1.3829)}
        """
        return self._self_reduction(torch.norm, dim=dim, **kwargs, p=p)

    @torchfunc_registry.register(torch.prod)
    def prod(self, dim: Optional[DimT] = None, **kwargs):
        return self._self_reduction(torch.prod, dim=dim, **kwargs)

    @torchfunc_registry.register(torch.unbind)
    def unbind(self) -> List['KeyedTensor']:
        return _one_to_many(self, torch.unbind)

    @torchfunc_registry.register(torch.abs)
    def abs(self):
        return self._apply_out_of_place(torch.abs)

    @torchfunc_registry.register(torch.acos)
    def acos(self):
        return self._apply_out_of_place(torch.acos)

    @torchfunc_registry.register(torch.asin)
    def asin(self):
        return self._apply_out_of_place(torch.asin)

    @torchfunc_registry.register(torch.atan)
    def atan(self):
        return self._apply_out_of_place(torch.atan)

    @torchfunc_registry.register(torch.ceil)
    def ceil(self):
        return self._apply_out_of_place(torch.ceil)

    @torchfunc_registry.register(torch.chunk)
    def chunk(self, chunks: int, dim=0) -> List['KeyedTensor']:
        return _one_to_many(self, torch.chunk, chunks, dim=dim)

    @torchfunc_registry.register(torch.cos)
    def cos(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.cos)

    @torchfunc_registry.register(torch.cosh)
    def cosh(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.cosh)

    def cuda(self, *args, **kwargs) -> 'KeyedTensor':
        return self._apply_out_of_place(lambda x: x.cuda(*args, **kwargs))

    @property
    def data(self) -> 'KeyedTensor':
        return self._apply_out_of_place(lambda x: x.data)

    @torchfunc_registry.register(torch.detach)
    def detach(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.detach)

    @torchfunc_registry.register(torch.digamma)
    def digamma(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.digamma)

    def dim(self):
        return self._apply_out_of_place(lambda x: x.dim)

    @torchfunc_registry.register(torch.erf)
    def erf(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.erf)

    @torchfunc_registry.register(torch.erfc)
    def erfc(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.erfc)

    @torchfunc_registry.register(torch.erfinv)
    def erfinv(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.erfinv)

    @torchfunc_registry.register(torch.exp)
    def exp(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.exp)

    @torchfunc_registry.register(torch.expm1)
    def expm1(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.expm1)

    @torchfunc_registry.register(torch.floor)
    def floor(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.floor)

    @torchfunc_registry.register(torch.frac)
    def frac(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.frac)

    @torchfunc_registry.register(torch.hardshrink)
    def hardshrink(self, *args, **kwargs) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.hardshrink, *args, **kwargs)

    @torchfunc_registry.register(torch.isnan)
    def isnan(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.isnan)

    @torchfunc_registry.register(torch.isfinite)
    def isfinite(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.isfinite)

    @torchfunc_registry.register(torch.isinf)
    def isinf(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.isinf)

    @torchfunc_registry.register(torch.lgamma)
    def lgamma(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.lgamma)

    @torchfunc_registry.register(torch.log)
    def log(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.log)

    @torchfunc_registry.register(torch.log10)
    def log10(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.log10)

    @torchfunc_registry.register(torch.neg)
    def neg(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.neg)

    @torchfunc_registry.register(torch.numel)
    def numel(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.numel)

    @torchfunc_registry.register(torch.polygamma)
    def polygamma(self, *args, **kwargs):
        return _self_apply_with_args(self, torch.polygamma, *args, **kwargs)

    @torchfunc_registry.register(torch.reciprocal)
    def reciprocal(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.reciprocal)

    @torchfunc_registry.register(torch.relu)
    def relu(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.relu)

    @torchfunc_registry.register(torch.round)
    def round(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.round)

    @torchfunc_registry.register(torch.rsqrt)
    def rsqrt(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.rsqrt)

    @torchfunc_registry.register(torch.sigmoid)
    def sigmoid(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.sigmoid)

    @torchfunc_registry.register(torch.sign)
    def sign(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.sign)

    @torchfunc_registry.register(torch.sin)
    def sin(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.sin)

    @torchfunc_registry.register(torch.sinh)
    def sinh(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.sinh)

    @torchfunc_registry.register(torch.split)
    def split(self, split_size_or_sections, dim=0) -> List['KeyedTensor']:
        return _one_to_many(self, torch.split, split_size_or_sections, dim=dim)

    @torchfunc_registry.register(torch.sqrt)
    def sqrt(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.sqrt)

    @torchfunc_registry.register(torch.square)
    def square(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.square)

    @torchfunc_registry.register(torch.squeeze)
    def squeeze(self, *args, **kwargs) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.squeeze, *args, **kwargs)

    @torchfunc_registry.register(torch.t)
    def t(self, *args, **kwargs):
        return self._apply_out_of_place(torch.t, *args, **kwargs)

    @torchfunc_registry.register(torch.tan)
    def tan(self, *args, **kwargs):
        return self._apply_out_of_place(torch.tan, *args, **kwargs)

    @torchfunc_registry.register(torch.tanh)
    def tanh(self, *args, **kwargs):
        return self._apply_out_of_place(torch.tanh, *args, **kwargs)

    def to(self, *args, **kwargs):
        return self._apply_out_of_place(lambda x: x.to(*args, **kwargs))

    @torchfunc_registry.register(torch.transpose)
    def transpose(self, dim0, dim1) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.transpose, dim0, dim1)

    @torchfunc_registry.register(torch.trunc)
    def trunc(self, *args, **kwargs):
        return self._apply_out_of_place(torch.trunc, *args, **kwargs)

    @torchfunc_registry.register(torch.unsqueeze)
    def unsqueeze(self, dim) -> 'KeyedTensor':
        return _self_apply_with_args(self, torch.unsqueeze, dim)

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
