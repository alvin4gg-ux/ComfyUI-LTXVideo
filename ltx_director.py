# ComfyUI-LTXVideo/ltx_director.py
from comfy_api.latest import io

class GuideData:
    """
    A marker/data class used as an input type for batch guide data.
    """
    @staticmethod
    def Input(name: str, optional: bool = True, tooltip: str = "") -> io.Input:
        """
        Defines an input port for guide data (dictionary).
        """
        return io.Input(
            name=name,
            optional=optional,
            tooltip=tooltip,
            dtype=dict
        )