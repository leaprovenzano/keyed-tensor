import torch
from torch import nn
from typing import Optional, Sequence, Mapping, Union
from keyedtensor import KeyedTensor


class KeyedTensorOutputHook:
    """A hook for turning model outputs into KeyedTensors.

    This hook is useful for multi output models where you don't want to
    invoke KeyedTensor in the forward method (perhaps because you expect)
    to do scripting or other stuff.

    Args:
        keys (optional): a sequence of keys -- **NOTE** this option must be
         provided unless the model you're hooking into outputs a dictionary
         (or other mapping). Defaults to None.

    Example:
        >>> from typing import Tuple
        >>> import torch
        >>> from torch import nn
        >>> from keyedtensor.hooks import KeyedTensorOutputHook
        >>> _ =torch.manual_seed(0)
        >>>
        >>>
        >>> class ABModule(nn.Module):
        ...
        ...     def __init__(self, in_feats, a_feats, b_feats):
        ...         super().__init__()
        ...         self.a_lin = nn.Linear(in_feats, a_feats)
        ...         self.b_lin = nn.Linear(in_feats, b_feats)
        ...
        ...     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...         a = self.a_lin(x)
        ...         b = self.b_lin(x)
        ...         return a, b
        ...
        >>> # create a new hook that will turn the models outputs into
        >>> # a KeyedTensor with keys ('a', 'b')
        >>> hook = KeyedTensorOutputHook(keys=('a', 'b'))
        >>> # instantiate our model
        >>> model = ABModule(in_feats=10, a_feats=3, b_feats=2)
        >>> # register this hook with the model
        >>> handle = hook.register(model)
        >>> # now our model should output a tensordict
        >>> model(torch.rand(3, 10))
        KeyedTensor(a=tensor([[ 0.0373,  0.0077, -0.0353],
                              [-0.2695, -0.4360, -0.2835],
                              [ 0.0767,  0.1910,  0.1415]], grad_fn=<AddmmBackward>),
                    b=tensor([[ 0.0350,  0.3061],
                              [-0.2230,  0.7333],
                              [-0.1002,  0.1675]], grad_fn=<AddmmBackward>))


        If your model already outputs a regular dict or mapping you can use the hook without
        the keys argument to transform the models output mapping into a KeyedTensor:

        >>> from typing import Dict
        >>>
        >>> class ABDictModule(ABModule):
        ...
        ...     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...         a, b = super().forward(x)
        ...         return {'a': a, 'b': b}
        ...
        ...
        >>> # we dont need keys here since our model outputs a dict already
        >>> hook = KeyedTensorOutputHook()
        >>> model = ABDictModule(in_feats=10, a_feats=3, b_feats=2)
        >>> handle = hook.register(model)
        >>> model(torch.rand(3, 10))
        KeyedTensor(a=tensor([[-0.1848, -0.0749, -0.3038],
                              [-0.2702, -0.3355,  0.2280],
                              [-0.2691, -0.2188, -0.1089]], grad_fn=<AddmmBackward>),
                    b=tensor([[-0.0545,  0.0056],
                              [-0.3244,  0.2195],
                              [-0.1564, -0.0146]], grad_fn=<AddmmBackward>))

        you could also use the keys argument for the above case to map outputs to new keys:

        >>> # remove the old hook
        >>> _ = handle.remove()
        >>> # create a new hook with keys ('x', 'y')
        >>> hook = KeyedTensorOutputHook(keys=('x', 'y'))
        >>> handle = hook.register(model)
        >>> model(torch.rand(2, 10))
        KeyedTensor(x=tensor([[-0.0918, -0.1402, -0.1002],
                              [-0.5040, -0.3772, -0.1681]], grad_fn=<AddmmBackward>),
                    y=tensor([[ 0.1031,  0.0755],
                              [-0.0902, -0.0877]], grad_fn=<AddmmBackward>))
    """

    def __init__(self, keys: Optional[Sequence[str]] = None):
        self.keys = keys

    def register(self, model: nn.Module) -> torch.utils.hooks.RemovableHandle:
        """register this hook on a model and return a RemovableHandle."""
        return model.register_forward_hook(self)  # type: ignore

    def _handle(self, x: Union[Mapping[str, torch.Tensor], Sequence[torch.Tensor]]) -> KeyedTensor:
        if self.keys is None:
            if not isinstance(x, Mapping):
                raise TypeError(
                    (
                        f'{self.__class__.__name__} expected outputs to be a mapping type'
                        f' but got {type(x)}'
                    )
                )

            return KeyedTensor(x)

        elif isinstance(x, (Sequence, Mapping)):
            if len(x) != len(self.keys):
                raise RuntimeError(
                    (
                        f'{self.__class__.__name__} expected sequence of {len(self.keys)}'
                        f' to match keys but got output of len {len(x)}'
                    )
                )
            return KeyedTensor(zip(self.keys, x.values() if isinstance(x, Mapping) else x))

        raise TypeError(
            (
                f'{self.__class__.__name__} expected outputs to be a sequence of tensors'
                f' but got {type(x)}'
            )
        )

    def __call__(
        self, module, inp, out: Union[Mapping[str, torch.Tensor], Sequence[torch.Tensor]]
    ) -> KeyedTensor:
        return self._handle(out)
