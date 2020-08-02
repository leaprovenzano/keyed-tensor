import pytest
import torch
from keyedtensor import KeyedTensor

self_operator_funcs = [
    'abs',
    'acos',
    'asin',
    'atan',
    'ceil',
    'cos',
    'cosh',
    'digamma',
    'erf',
    'erfc',
    'erfinv',
    'exp',
    'expm1',
    'floor',
    'frac',
    'hardshrink',
    'isfinite',
    'isinf',
    'isnan',
    'lgamma',
    'log',
    'log10',
    'neg',
    'reciprocal',
    'relu',
    'round',
    'rsqrt',
    'sigmoid',
    'sign',
    'sin',
    'sinh',
    'sqrt',
    'square',
    'tan',
    'tanh',
    'trunc',
]


@pytest.mark.parametrize('funcname,', self_operator_funcs)
def test_all_reduce(funcname):

    kt = KeyedTensor(a=torch.rand(5, 4), b=torch.rand(5, 2), c=torch.rand(5))

    torchfunc = getattr(torch, funcname)

    # test as classmethod
    method = getattr(KeyedTensor, funcname)
    result = method(kt)

    assert (result == torchfunc(kt)).all()

    # test outputs
    assert (result.a == torchfunc(kt.a)).all()
    assert (result.b == torchfunc(kt.b)).all()
    assert (result.c == torchfunc(kt.c)).all()
