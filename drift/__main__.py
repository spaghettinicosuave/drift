"""CLI.

    python -m drift wav out.wav --seconds 180 --preset night
    python -m drift wav out.wav --preset battle --meter 7/8
    python -m drift raw --preset city | pw-play --rate 44100 --channels 2 --format s16 -
    python -m drift wav out.wav --config configs/drift-4242-....json
"""

import argparse

from . import PRESETS, Engine, load_config, stream_raw, write_wav
from .theory import parse_meter


def meter_arg(text: str) -> str:
    try:
        return parse_meter(text).name
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def main() -> None:
    ap = argparse.ArgumentParser(prog="drift")
    ap.add_argument("mode", choices=["wav", "raw"])
    ap.add_argument("path", nargs="?", default="drift.wav")
    ap.add_argument("--preset", default="meadow", choices=sorted(PRESETS))
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--config", help="saved config (JSON) from demo.py or save_config(); overrides --preset")
    ap.add_argument("--meter", type=meter_arg, help="time signature, e.g. 7/8, 5/4, 12/8 or 3+3+2/8")
    ap.add_argument("--variant", action="append", default=[], metavar="LAYER=NAME",
                    help="pin a layer's variant, e.g. --variant drums=808 (repeatable)")
    a = ap.parse_args()

    variants = dict(v.split("=", 1) for v in a.variant)
    if a.config:
        eng = load_config(a.config, sr=a.sr)
        for layer, name in variants.items():
            eng.set_variant(layer, name)
    else:
        eng = Engine(a.preset, sr=a.sr, seed=a.seed, variants=variants)
    if a.meter:
        eng = Engine.from_config({**eng.config(), "meter": a.meter}, sr=a.sr)  # from bar one, not the next bar
    if a.mode == "wav":
        write_wav(eng, a.path, a.seconds, progress=True)
        print(f"wrote {a.path} (seed {eng.seed})")
        print("variants:", " ".join(f"{k}={v}" for k, v in eng.variants().items()))
    else:
        stream_raw(eng)


if __name__ == "__main__":
    main()
