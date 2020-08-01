from typing import Type, Callable, TypeVar, Generic, Hashable, Dict, Any
import itertools

import torch
from collectionish import AncestorChainMap


def _get_overridable_torchfuncs():
    return set(itertools.chain.from_iterable(torch._overrides.get_overridable_functions().values()))


OVERRIDABLE_TORCHFUNCS = _get_overridable_torchfuncs()

KT = TypeVar('KT', bound=Hashable)
BoundT = TypeVar('BoundT')


class MethodRegistryDescriptor(Generic[KT, BoundT]):

    """A generic method registry descriptor that handles inheretence properly."""

    name: str

    def __init__(self, **mapping: Dict[KT, Callable[[BoundT, Any], Any]]):
        self._registry = AncestorChainMap(mapping)

    def register(self, key: KT) -> Callable:
        def deco(f: Callable):
            self._registry[key] = f
            return f

        return deco

    def __get__(self, inst: BoundT, owner: Type[BoundT]) -> 'MethodRegistryDescriptor':
        return self

    def __contains__(self, key: KT) -> bool:
        return key in self._registry

    def __getitem__(self, key: KT) -> Callable[[BoundT, Any], Any]:
        return self._registry[key]

    def _update_registry_from_bases(self, owner: Type[BoundT]):
        # reverse so first base is first parent
        for parent in owner.__bases__[::-1]:
            if hasattr(parent, self.name):
                parent_descriptor = getattr(parent, self.name)
                if isinstance(parent_descriptor, self.__class__):
                    self._registry.add_parent(parent_descriptor._registry)

    def __set_name__(self, owner: Type[BoundT], name: str):
        self.name = name
        self._update_registry_from_bases(owner)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._registry})'


class TorchFuncRegistry(MethodRegistryDescriptor[Callable, BoundT]):

    """a classlevel registry for registring __torch_function__ overides.
    """

    def register(self, torchfunc: Callable) -> Callable:
        """this can be used as a decorator for registering torchfunc implementations.
        """
        # check the torchfunc can actually be overidden
        if torchfunc not in OVERRIDABLE_TORCHFUNCS:
            raise TypeError(
                'registered keys must be valid overridable torch functions'
                ' see torch._overrides.get_overridable_functions() for the breakdown'
                ' of what can and cannot be overridden.'
            )

        def deco(f: Callable):

            self._registry[torchfunc] = f
            return f

        return deco
