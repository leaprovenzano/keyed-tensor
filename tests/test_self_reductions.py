import pytest
import torch
from keyedtensor import KeyedTensor

bool_reductions = ['any', 'all']
self_reductions = ['argmin', 'argmax', 'mean', 'norm', 'prod', 'var', 'sum', 'std', 'prod']


@pytest.mark.parametrize('funcname,', self_reductions + bool_reductions)
def test_all_reduce(funcname):

    kt = KeyedTensor(a=torch.rand(5, 4), b=torch.rand(5, 2), c=torch.rand(5))
    if funcname in bool_reductions:
        kt.a = kt.a >= 0.5
        kt.b = kt.b >= 0.5
        kt.c = kt.c >= 0.5

    flatvalues = torch.cat(list(map(torch.flatten, kt.values())))

    torchfunc = getattr(torch, funcname)
    expected = torchfunc(flatvalues)

    # test as classmethod

    method = getattr(kt, funcname)

    return method() == torchfunc(kt) == expected


@pytest.mark.parametrize('funcname,', self_reductions + bool_reductions)
def test_keyed_reduce(funcname):

    kt = KeyedTensor(a=torch.rand(5, 4), b=torch.rand(5, 2), c=torch.rand(5))
    if funcname in bool_reductions:
        kt.a = kt.a >= 0.5
        kt.b = kt.b >= 0.5
        kt.c = kt.c >= 0.5

    torchfunc = getattr(torch, funcname)

    # test as classmethod

    method = getattr(KeyedTensor, funcname)
    result = method(kt, dim='key')

    assert result.a == torchfunc(kt.a)
    assert result.b == torchfunc(kt.b)
    assert result.c == torchfunc(kt.c)


@pytest.mark.parametrize('funcname,', self_reductions + bool_reductions)
def test_dim_reduce(funcname):

    kt = KeyedTensor(a=torch.rand(5, 4), b=torch.rand(5, 2), c=torch.rand(5))
    if funcname in bool_reductions:
        kt.a = kt.a >= 0.5
        kt.b = kt.b >= 0.5
        kt.c = kt.c >= 0.5

    torchfunc = getattr(torch, funcname)

    # test as classmethod
    method = getattr(KeyedTensor, funcname)
    result = method(kt, dim=-1)

    assert (result == torchfunc(kt, dim=-1)).all()

    # test outputs
    assert (result.a == torchfunc(kt.a, dim=-1)).all()
    assert (result.b == torchfunc(kt.b, dim=-1)).all()
    assert (result.c == torchfunc(kt.c, dim=-1)).all()
