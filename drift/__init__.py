"""drift: endless, parameter-driven generative ambient music. numpy only.

    from drift import Engine
    eng = Engine("meadow", seed=1)
    audio = eng.render(44100)      # (44100, 2) float32
    eng.set(energy=0.8, tension=0.5)
"""

from .engine import Engine, PARAMS
from .output import PygamePlayer, load_config, save_config, stream_raw, write_wav
from .presets import PRESETS
from .voices import Arp, Bells, Drone, Layer, Melody, Pad, Swell, default_layers

__all__ = [
    "Engine", "PARAMS", "PRESETS", "PygamePlayer", "write_wav", "stream_raw", "save_config", "load_config",
    "Layer", "Drone", "Pad", "Swell", "Arp", "Bells", "Melody", "default_layers",
]
