from .maker import Maker
from controllers.mac import MAC


class MACMaker(Maker):
    """Factory class for creating Controllers."""

    @staticmethod
    def make_basic_mac(*args, **kwargs) -> MAC:
        from controllers.basic_controller import BasicMAC
        return BasicMAC(*args, **kwargs)

    @staticmethod
    def make_non_shared_mac(*args, **kwargs) -> MAC:
        from controllers.non_shared_controller import NonSharedMAC
        return NonSharedMAC(*args, **kwargs)    # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_maddpg_mac(*args, **kwargs) -> MAC:
        from controllers.maddpg_controller import MADDPGMAC
        return MADDPGMAC(*args, **kwargs)   # TODO migrate NonSharedMAC to MAC

    @staticmethod
    def make_entity_mac(*args, **kwargs) -> MAC:
        from controllers.entity_controller import EntityMAC
        return EntityMAC(*args, **kwargs)

    @staticmethod
    def make_pymarl2_nmac_controller(*args, **kwargs) -> MAC:
        """MAC Implementation from PyMARL2"""
        from controllers.pymarl2_nmac_controller import NMAC
        return NMAC(*args, **kwargs)

    @staticmethod
    def make_entity_maic_mac(*args, **kwargs) -> MAC:
        from controllers.entity_maic_controller import EntityMAICMAC
        return EntityMAICMAC(*args, **kwargs)
