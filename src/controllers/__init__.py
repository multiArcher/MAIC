from utils.maker import Maker
from .MAC import MAC

class MACMaker(Maker):
    """Factory class for creating MAC."""
    @staticmethod
    def make_basic_mac(*args, **kwargs) -> MAC:
        from .basic_controller import BasicMAC
        return BasicMAC(*args, **kwargs)

    @staticmethod
    def make_non_shared_mac(*args, **kwargs) -> MAC:
        from .non_shared_controller import NonSharedMAC
        return NonSharedMAC(*args, **kwargs)    # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_maddpg_mac(*args, **kwargs) -> MAC:
        from .maddpg_controller import MADDPGMAC
        return MADDPGMAC(*args, **kwargs)   # TODO migrate NonSharedMAC to MAC
