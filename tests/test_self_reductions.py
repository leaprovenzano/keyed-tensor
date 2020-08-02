import pytest
import torch
from keyedtensor import KeyedTensor

bool_reductions = ['any', 'all']
self_reductions = ['argmin', 'argmax', 'mean', 'median', 'norm', 'prod', 'var', 'sum', 'std']


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
