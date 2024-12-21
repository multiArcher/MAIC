from abc import ABC, abstractmethod
from typing import Callable


class MetaMaker(type):
    """Metaclass for Maker class"""
    @classmethod
    @abstractmethod
    def make_func(cls, target_type: str) -> Callable:
        """Abstract method to get the make function for the target type"""
        pass

    def __getitem__(cls, item) -> Callable:
        return cls.make_func(item)


class Maker(metaclass=MetaMaker):
    """Abstract class for making objects"""

    @classmethod
    def check_type(cls, key):
        """Check if the type is implemented in Maker class"""
        assert hasattr(cls, f"make_{key}"), f"Method 'make_{key}' not found in {cls.__name__}"

    @classmethod
    def make_func(cls, target_type: str) -> Callable:
        """Get the make function for the target type"""
        cls.check_type(target_type)
        return getattr(cls, f"make_{target_type}")

    @classmethod
    def make(cls, target_type: str, *args, **kwargs):
        """Make an object of the target type"""
        make_func = cls.make_func(target_type)
        return make_func(*args, **kwargs)
