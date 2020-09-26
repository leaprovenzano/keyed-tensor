import pytest
import torch
from keyedtensor import KeyedTensor


def test_invalid_key_raises_attribute_err():
    x = KeyedTensor(a=torch.rand(3, 4), b=torch.rand(3, 3), c=torch.rand(3))

    with pytest.raises(AttributeError, match='boop'):
        x.shape.boop
