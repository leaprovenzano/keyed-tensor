import torch
from keyedtensor import KeyedTensor


def test_neg():
    kt = KeyedTensor(a=torch.rand(5, 4), b=torch.rand(5))

    result = -kt

    # test outputs
    assert (result.a == -kt.a).all()
    assert (result.b == -kt.b).all()


def test_abs():
    kt = KeyedTensor(a=torch.rand(5, 4) * 2 - 1, b=torch.rand(5) * 2 - 1)

    result = abs(kt)

    # test outputs
    assert (result.a == abs(kt.a)).all()
    assert (result.b == abs(kt.b)).all()
