from typing import Dict, Tuple, List
import pytest
import torch
from torch import nn

from keyedtensor import KeyedTensor, KeyedTensorOutputHook


class BaseModel(nn.Module):
    def __init__(self, in_feats=10, a_feats=3, b_feats=4):
        super().__init__()
        self.a_lin = nn.Linear(in_feats, a_feats)
        self.b_lin = nn.Linear(in_feats, b_feats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.a_lin(x)
        # we put sigmoid in here to differentiate grad fn
        b = torch.sigmoid(self.b_lin(x))
        return a, b


class TupleModel(BaseModel):
    pass


class DictModel(BaseModel):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        a, b = super().forward(x)
        return {'a': a, 'b': b}


class ListModel(BaseModel):
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(super().forward(x))


@pytest.mark.parametrize('model,', [TupleModel(), ListModel(), DictModel()])
def test_with_keys(model):
    hook = KeyedTensorOutputHook(keys=('boop', 'bop'))
    hook.register(model)

    # rand inputs
    x = torch.rand(5, 10)

    out = model(x)
    assert isinstance(out, KeyedTensor)
    # check keys are correct
    assert list(out) == ['boop', 'bop']


@pytest.mark.parametrize('model,', [DictModel()])
def test_with_null_on_mapping(model):
    hook = KeyedTensorOutputHook()
    hook.register(model)

    # rand inputs
    x = torch.rand(5, 10)

    out = model(x)
    assert isinstance(out, KeyedTensor)
    # check keys are correct
    assert list(out) == ['a', 'b']


@pytest.mark.parametrize('model,', [TupleModel(), ListModel()])
def test_with_null_keys_fails_on_nonmapping(model):
    expected_msg = 'KeyedTensorOutputHook expected outputs to be a mapping type but got .*'

    hook = KeyedTensorOutputHook()
    hook.register(model)

    with pytest.raises(TypeError, match=expected_msg):
        model(torch.rand(5, 10))


@pytest.mark.parametrize('model,', [TupleModel(), ListModel(), DictModel()])
def test_output_values(model):
    hook = KeyedTensorOutputHook(keys=('a', 'b'))

    # rand inputs
    x = torch.rand(5, 10)

    with torch.no_grad():
        orig_out = model(x)
    # standardize for dict models
    if isinstance(model, DictModel):
        orig_out = tuple(orig_out.values())

    # register the hook
    hook.register(model)

    with torch.no_grad():
        hooked_out = model(x)

    assert isinstance(hooked_out, KeyedTensor)
    # check keys are correct
    assert list(hooked_out) == ['a', 'b']

    assert not any(map(lambda x: x.requires_grad, hooked_out.values()))
    assert (hooked_out.a == orig_out[0]).all()
    assert (hooked_out.b == orig_out[1]).all()


@pytest.mark.parametrize('model,', [TupleModel(), ListModel(), DictModel()])
def test_grad(model):
    hook = KeyedTensorOutputHook(keys=('a', 'b'))
    # register the hook
    hook.register(model)
    out = model(torch.rand(5, 10))

    assert out.a.requires_grad
    assert out.b.requires_grad

    assert out.a.grad_fn.name() == 'AddmmBackward'
    assert out.b.grad_fn.name() == 'SigmoidBackward'
