from typing_extensions import Literal
from typing import List, Union, Optional
import numpy as np

import torch

from collectionish import AttyDict

from keyedtensor._registry import TorchFuncRegistry


DimT = Union[Literal['keys'], int]


# patterns


def self_reduction(
    kt: 'KeyedTensor', op, dim: Optional[DimT] = None, keepdim: bool = False, **kwargs
):
    if dim is None:
        return op(torch.stack(list(map(op, kt.values()))))
    elif dim == 'key':
        return kt._apply_out_of_place(op)
    return kt._apply_out_of_place(lambda x: op(x, dim=dim, keepdim=keepdim))


def self_apply_with_args(kt: 'KeyedTensor', op, *args, **kwargs):
    return kt._apply_out_of_place(lambda x: op(x, *args, **kwargs))


def one_to_many(kt: 'KeyedTensor', op, *args, **kwargs) -> List['KeyedTensor']:
    return [
        kt.__class__(zip(kt.keys(), values))
        for values in zip(*map(lambda x: op(x, *args, **kwargs), kt.values()))
    ]


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

    def __abs__(self):
        return self.abs()

    def __all__(self):
        return self.all()

    def __any__(self):
        return self.any()

    @torchfunc_registry.register(torch.all)
    def all(self, dim: Optional[DimT] = None, keepdim: bool = False):
        """Like torch.all but for keyed tensor. dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.
            keepdim: whether the output tensor has :attr:`dim` retained or not. Defaults to False.
        """
        return self_reduction(self, torch.any, dim=dim, keepdim=keepdim)

    @torchfunc_registry.register(torch.any)
    def any(self, dim: Optional[DimT] = None, keepdim: bool = False):
        """Like torch.any but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.
            keepdim: whether the output tensor has :attr:`dim` retained or not. Defaults to False.
        """
        return self_reduction(self, torch.any, dim=dim, keepdim=keepdim)

    @torchfunc_registry.register(torch.mean)
    def mean(self, dim: Optional[DimT] = None, keepdim: bool = False):
        """Like torch.mean but for keyed tensor, dim may optionally be a keyed

        Args:
            dim: the dimension to reduce -this may optionally be the string
                literal 'key' to reduce by key. Defaults to None.
            keepdim: whether the output tensor has :attr:`dim` retained or not. Defaults to False.

        Example:
            >>> import torch
            >>> from keyedtensor import KeyedTensor
            >>>
            >>> _ = torch.manual_seed(0)
            >>> kt = KeyedTensor(a=torch.rand(3, 3), b=torch.rand(3))
            >>> kt.mean()
            tensor(0.4676)

            >>> print(kt.mean(dim=-1))
            {'a': tensor([0.4510, 0.3578, 0.6141]), 'b': tensor(0.4610)}

            >>> kt.mean(dim='key')
            {'a': tensor(0.4743), 'b': tensor(0.4610)}
        """
        return self_reduction(self, torch.mean, dim=dim, keepdim=keepdim)

    @torchfunc_registry.register(torch.median)
    def median(self, *args, **kwargs):
        return self_reduction(self, torch.median, *args, **kwargs)

    @torchfunc_registry.register(torch.sum)
    def sum(self, *args, **kwargs):
        return self_reduction(self, torch.sum, *args, **kwargs)

    @torchfunc_registry.register(torch.var)
    def var(self, *args, **kwargs):
        return self_reduction(self, torch.var, *args, **kwargs)

    @torchfunc_registry.register(torch.argmax)
    def argmax(self, dim: Optional[DimT] = None, keepdim: bool = False):
        return self_reduction(self, torch.argmax, dim=dim, keepdim=keepdim)

    @torchfunc_registry.register(torch.argmin)
    def argmin(self, dim: Optional[DimT] = None, keepdim: bool = False):
        return self_reduction(self, torch.argmin, dim=dim, keepdim=keepdim)

    @torchfunc_registry.register(torch.std)
    def std(self, *args, **kwargs):
        return self_reduction(self, torch.std, *args, **kwargs)

    @torchfunc_registry.register(torch.norm)
    def norm(self, p='fro', *args, **kwargs):
        return self_reduction(self, torch.norm, *args, **kwargs, p=p)

    @torchfunc_registry.register(torch.unbind)
    def unbind(self) -> List['KeyedTensor']:
        return one_to_many(self, torch.unbind)

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

    @torchfunc_registry.register(torch.bernoulli)
    def bernoulli(self, *args, **kwargs):
        return self_apply_with_args(self, torch.bernoulli, *args, **kwargs)

    @torchfunc_registry.register(torch.ceil)
    def ceil(self):
        return self._apply_out_of_place(torch.ceil)

    @torchfunc_registry.register(torch.chunk)
    def chunk(self, chunks: int, dim=0) -> List['KeyedTensor']:
        return one_to_many(self, torch.chunk, chunks, dim=dim)

    @torchfunc_registry.register(torch.cos)
    def cos(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.cos)

    @torchfunc_registry.register(torch.cosh)
    def cosh(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.cosh)

    def cuda(self, *args, **kwargs) -> 'KeyedTensor':
        return self._apply_out_of_place(lambda x: x.cuda(*args, **kwargs))

    @torchfunc_registry.register(torch.cumprod)
    def cumprod(self, *args, **kwargs):
        return self_apply_with_args(self, torch.cumprod, *args, **kwargs)

    @torchfunc_registry.register(torch.cumsum)
    def cumsum(self, *args, **kwargs):
        return self_apply_with_args(self, torch.cumsum, *args, **kwargs)

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
        return self_apply_with_args(self, torch.hardshrink, *args, **kwargs)

    @torchfunc_registry.register(torch.inverse)
    def inverse(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.inverse)

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

    def __neg__(self) -> 'KeyedTensor':
        return self.neg()

    @torchfunc_registry.register(torch.numel)
    def numel(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.numel)

    @torchfunc_registry.register(torch.polygamma)
    def polygamma(self, *args, **kwargs):
        return self_apply_with_args(self, torch.polygamma, *args, **kwargs)

    @torchfunc_registry.register(torch.prod)
    def prod(self, *args, **kwargs):
        return self_reduction(self, torch.prod, *args, **kwargs)

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
        return one_to_many(self, torch.split, split_size_or_sections, dim=dim)

    @torchfunc_registry.register(torch.sqrt)
    def sqrt(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.sqrt)

    @torchfunc_registry.register(torch.square)
    def square(self) -> 'KeyedTensor':
        return self._apply_out_of_place(torch.square)

    @torchfunc_registry.register(torch.squeeze)
    def squeeze(self, *args, **kwargs) -> 'KeyedTensor':
        return self_apply_with_args(self, torch.squeeze, *args, **kwargs)

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
        return self_apply_with_args(self, torch.transpose, dim0, dim1)

    @torchfunc_registry.register(torch.trunc)
    def trunc(self, *args, **kwargs):
        return self._apply_out_of_place(torch.trunc, *args, **kwargs)

    @torchfunc_registry.register(torch.unsqueeze)
    def unsqueeze(self, dim) -> 'KeyedTensor':
        return self_apply_with_args(self, torch.unsqueeze, dim)
