from typing import Union, TypeVar

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .grc_controller import GRCMAC

CONTROLLER_TYPES = TypeVar("CONTROLLER_TYPES", BasicMAC, NonSharedMAC, MADDPGMAC, GRCMAC)

REGISTRY = {
    "basic_mac": BasicMAC,
    "non_shared_mac": NonSharedMAC,
    "maddpg_mac": MADDPGMAC,
    "grc_mac": GRCMAC
}


def get_controller(controller_name: str, scheme, groups, args) \
        -> Union[
            BasicMAC,
            NonSharedMAC,
            MADDPGMAC,
            GRCMAC
        ]:
    if controller_name in REGISTRY.keys():
        return REGISTRY[controller_name](scheme, groups, args)
    else:
        raise ValueError(f"Invalid controller name: {controller_name}.")
