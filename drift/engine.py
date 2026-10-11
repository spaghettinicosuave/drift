"""The Engine: clock + harmony + layers + effects. Pull-based: call render(n) for audio."""

from __future__ import annotations

import math
import random
import threading
from typing import Callable

import numpy as np

from .dsp import Echo, Reverb
from .presets import PRESETS
from .theory import Harmony, Meter, parse_meter
from .voices import Layer, default_layers

BLOCK = 1024

PARAMS = ("tempo", "energy", "drive", "tension", "brightness", "space", "volume")
# Common time signatures (any "N/4", "N/8", "N/2" or grouped "3+2+2/8" works too).
# Tempo counts beats: quarter notes in 4/4, dotted quarters in 6/8 (a jig at 110 has
# two beats per bar), quarter notes in uneven meters like 7/8.
METERS = ("4/4", "3/4", "6/8", "2/4", "5/4", "7/8", "5/8", "9/8", "12/8", "7/4")

DEFAULTS = {"tempo": 72, "energy": 0.45, "drive": 0.0, "tension": 0.2, "brightness": 0.5, "space": 0.6, "volume": 0.8}


class Engine:
    """Endless generative music.

    Parameters (all 0..1 except tempo in BPM) can be changed any time with set();
    they glide smoothly so the music never jumps:

        energy      how many layers play and how busy they are
        drive       drums, bass, stabs and risers on top (0 = pure ambient)
        tension     bright modes (lydian/major) -> dark modes (dorian/aeolian/phrygian)
        brightness  timbre: soft/dark -> bright/airy
        space       reverb and echo amount / length
        volume      master level
    """

    def __init__(
        self,
        preset: str | dict = "meadow",
        *,
        sr: int = 44100,
        seed: int | None = None,
        layers: list[Layer] | None = None,
        variants: dict[str, str] | None = None,
        glide: float = 6.0,
        crossfade: float = 6.0,
    ):
        cfg = dict(PRESETS[preset]) if isinstance(preset, str) else dict(preset)
        self.preset = preset if isinstance(preset, str) else None
        self.sr = sr
        self.seed = random.randrange(2**31) if seed is None else seed
        self.rng = random.Random(self.seed)
        self.nprng = np.random.default_rng(self.seed)
        self.layers = layers if layers is not None else default_layers()
        self.glide = glide
        self.crossfade = crossfade  # seconds; default for variant changes
        # Variant crossfades: outgoing copies [ghost, source, start sample, length, start gain],
        # fading-in layers -> (start sample, length), and layers that should enter right away.
        self._ghosts: list[list] = []
        self._incoming: dict[Layer, tuple[int, int]] = {}
        self._enter: set[Layer] = set()

        self.target = {k: float(cfg.get(k, DEFAULTS[k])) for k in PARAMS}
        self.p = dict(self.target)

        self.harmony = Harmony(self.rng, cfg.get("root"), "ionian")
        self.harmony.mode = self._mode_for(self.p["tension"])
        self.flags: dict[str, object] = {}
        self.listeners: list[Callable[[str, dict], None]] = []

        self.echo = Echo(sr, BLOCK)
        self.reverb = Reverb(sr, BLOCK, size=cfg.get("room", 1.0), seed=self.seed)

        self._apply_meter(parse_meter(cfg.get("meter", "4/4")))
        self._pending_meter: Meter | None = None
        self.pos = -1  # step within the bar
        self.bar = -1
        self.step = -1
        self.next_step_at = 0.0  # in samples, relative to start of current block
        self.chord_bars_left = 0
        self.chord_steps = 64
        self.samples = 0
        self.fade = 0.0  # fade-in on start
        self.level_target = 10 ** (-18 / 20)  # auto-gain aims for about -18 dB RMS
        self.ms = self.level_target**2
        self.agc = 1.0
        self._kicks: list[int] = []  # kick offsets in the current block (for sidechain)
        self._since_kick = 1e9  # samples since the last kick
        self._fifo = np.zeros((0, 2), dtype=np.float32)
        self._lock = threading.Lock()
        self.room = cfg.get("room", 1.0)
        for name, variant in {**cfg.get("variants", {}), **(variants or {})}.items():
            self.layer(name).set_variant(variant)
        for layer in self.layers:
            layer.level = max(0.0, min(1.0, (self.p[layer.param] - layer.threshold) * 2.5))

    # ------------------------------------------------------------------ public API

    def set(self, immediate: bool = False, **params: float) -> None:
        """Change parameters, e.g. engine.set(energy=0.7, tension=0.4)."""
        with self._lock:
            for k, v in params.items():
                if k not in self.target:
                    raise KeyError(f"unknown parameter {k!r}; expected one of {PARAMS}")
                self.target[k] = float(v)
                if immediate:
                    self.p[k] = float(v)

    def load(self, preset: str | dict, immediate: bool = False, fade: float | None = None) -> None:
        """Glide to another preset's parameters (the music keeps going). Presets that name
        variants crossfade to them over `fade` seconds and presets that name a meter switch
        to it at the next bar line; otherwise the current instruments and meter stay."""
        cfg = PRESETS[preset] if isinstance(preset, str) else preset
        keep = {"volume": self.target["volume"]}  # volume is the listener's, not the preset's
        self.set(immediate=immediate, **{k: cfg.get(k, keep.get(k, DEFAULTS[k])) for k in PARAMS})
        if "meter" in cfg:
            self.set_meter(cfg["meter"])
        for name, variant in cfg.get("variants", {}).items():
            self.set_variant(name, variant, fade=0.0 if immediate else fade)

    def set_meter(self, meter: str) -> None:
        """Time signature: '4/4', '3/4', '6/8', '7/8', '5/4', '12/8', or grouped like
        '3+3+2/8'. Takes effect at the next bar line. Raises ValueError if unreadable."""
        m = parse_meter(meter)
        with self._lock:
            self._pending_meter = None if m == self._meter else m

    def beat_at(self, pos: int) -> tuple[int, int, int]:
        """(beat index, step within the beat, beat length) for a step in the bar."""
        return self._beat_map[pos]

    def default_variants(self, fade: float | None = None) -> dict[str, str]:
        """Every layer back to its first (default) variant."""
        with self._lock:
            for layer in self.layers:
                if not layer.fixed_variant:
                    self._switch(layer, next(iter(layer.VARIANTS)), fade)
        return self.variants()

    def variants(self) -> dict[str, str]:
        return {x.name: x.variant for x in self.layers}

    def set_variant(self, layer: str, variant: str, fade: float | None = None) -> None:
        """Switch one layer's sound/style, crossfading over `fade` seconds (default
        self.crossfade). While it fades, the old variant keeps playing along with the new
        one; fade=0 switches at once and lets sounding notes ring out."""
        with self._lock:
            self._switch(self.layer(layer), variant, fade)

    def reroll_variants(self, rng: random.Random | None = None, keep: list[str] | tuple = (),
                        fade: float | None = None) -> dict[str, str]:
        """Pick a random variant for every layer except those named in `keep` and layers
        created with an explicit variant=. Independent of the music seed."""
        rng = rng or random.Random()
        with self._lock:
            for layer in self.layers:
                if layer.name not in keep and not layer.fixed_variant:
                    self._switch(layer, rng.choice(list(layer.VARIANTS)), fade)
        return self.variants()

    def layer(self, name: str) -> Layer:
        for x in self.layers:
            if x.name == name:
                return x
        raise KeyError(f"no layer named {name!r}; have {[x.name for x in self.layers]}")

    def config(self) -> dict:
        """Everything needed to recreate this engine's sound: seed, parameters, variants,
        disabled layers. JSON-serialisable. Engine.from_config(cfg) restores it; the music
        then starts from the beginning of that seed rather than where you left off."""
        return {
            "seed": self.seed,
            "params": {k: round(v, 4) for k, v in self.target.items()},
            "room": self.room,
            "meter": (self._pending_meter or self._meter).name,
            "variants": self.variants(),
            "disabled": [x.name for x in self.layers if not x.enabled],
        }

    @classmethod
    def from_config(cls, cfg: dict, **kw) -> Engine:
        eng = cls({**cfg.get("params", {}), "room": cfg.get("room", 1.0), "meter": cfg.get("meter", "4/4")},
                  seed=cfg.get("seed"), variants=cfg.get("variants"), **kw)
        for name in cfg.get("disabled", []):
            eng.layer(name).enabled = False
        return eng

    def on_event(self, fn: Callable[[str, dict], None]) -> None:
        """fn(kind, info) is called for 'chord', 'bar' and 'drift' events (from the audio thread)."""
        self.listeners.append(fn)

    def state(self) -> dict:
        return {
            "key": self.harmony.key_name(),
            "chord": self.harmony.chord_name(),
            "bar": max(0, self.bar),
            "meter": self.meter,
            "next_meter": self._pending_meter.name if self._pending_meter else None,
            "seconds": self.samples / self.sr,
            "params": dict(self.p),
            "layers": {x.name: round(x.level, 2) if x.enabled else None for x in self.layers},
            "variants": self.variants(),
            "notes": sum(len(x.notes) for x in self.layers + [g[0] for g in self._ghosts]),
        }

    def render(self, n: int) -> np.ndarray:
        """Return the next n frames as float32 stereo, shape (n, 2), roughly in [-1, 1]."""
        with self._lock:
            parts = [self._fifo]
            have = len(self._fifo)
            while have < n:
                b = self._block()
                parts.append(b)
                have += len(b)
            buf = np.concatenate(parts)
            self._fifo = buf[n:]
            return buf[:n]

    # ------------------------------------------------------------------ internals

    @property
    def step_samples(self) -> float:
        return self.sr * 60.0 / self.p["tempo"] / self.beat_steps

    @property
    def step_seconds(self) -> float:
        return 60.0 / self.p["tempo"] / self.beat_steps

    @property
    def chord_seconds(self) -> float:
        return self.chord_steps * self.step_seconds

    @staticmethod
    def _mode_for(tension: float) -> str:
        from .theory import MODE_ORDER
        return MODE_ORDER[min(len(MODE_ORDER) - 1, int(tension * len(MODE_ORDER)))]

    def _apply_meter(self, m: Meter) -> None:
        self._meter = m
        self.meter = m.name
        self.bar_steps, self.beat_steps = m.bar_steps, m.beat_steps
        self.beats, self.beat_lens = m.starts, m.groups  # beat start steps and lengths
        self._beat_map = m.beat_map()

    def _switch(self, layer: Layer, variant: str, fade: float | None) -> None:
        if variant not in layer.VARIANTS or variant == layer.variant:
            layer.set_variant(variant)  # raises KeyError for unknown names
            return
        fade = self.crossfade if fade is None else fade
        if fade > 0 and layer.enabled and (layer.notes or layer.active):
            length = max(1, int(fade * self.sr))
            gain = float(self._gain_in(layer, 1)[0]) if layer in self._incoming else 1.0
            mine = [g for g in self._ghosts if g[1] is layer]
            if len(mine) >= 2:  # switching fast: drop the quietest leftover
                self._ghosts.remove(mine[0])
            self._ghosts.append([layer.handoff(), layer, self.samples, length, gain])
            self._incoming[layer] = (self.samples, length)
            self._enter.add(layer)
        layer.set_variant(variant)

    def _progress(self, start: int, length: int, n: int) -> np.ndarray:
        return np.clip((self.samples + np.arange(n) - start) / length, 0.0, 1.0)

    def _gain_in(self, layer: Layer, n: int) -> np.ndarray:
        return np.sin(self._progress(*self._incoming[layer], n) * (math.pi / 2))

    def _voices(self) -> list[Layer]:
        """Layers that get hooks this step: the real ones, plus outgoing copies early in their fade."""
        if not self._ghosts:
            return self.layers
        return self.layers + [g for g, src, start, length, _ in self._ghosts if self.samples - start < 0.7 * length]

    def kick(self, offset: int) -> None:
        """Layers call this when a kick lands; ducking layers dip under it."""
        self._kicks.append(offset)

    def _duck_curve(self, n: int) -> np.ndarray:
        since = self._since_kick + np.arange(n, dtype=np.float64)
        for k in self._kicks:
            since[k:] = np.arange(n - k)
        self._since_kick = since[-1] + 1
        self._kicks.clear()
        tau = min(0.18, 0.45 * 60.0 / self.p["tempo"]) * self.sr
        return np.exp(-since / tau)

    def _emit(self, kind: str, **info) -> None:
        for fn in self.listeners:
            fn(kind, info)

    def _smooth(self) -> None:
        a = 1.0 - math.exp(-BLOCK / (self.sr * self.glide))
        for k, v in self.target.items():
            self.p[k] += (v - self.p[k]) * a

    def _tick(self, offset: int) -> None:
        self.step += 1
        self.pos += 1
        if self.pos >= self.bar_steps or self.bar < 0:
            if self._pending_meter:
                self._apply_meter(self._pending_meter)
                self._pending_meter = None
            self.pos = 0
            self.bar += 1
        step = self.step
        for ghost, src, *_ in self._ghosts:
            if ghost.enabled and not src.enabled:  # muted mid-fade
                ghost.enabled = False
                ghost.release_all(offset)
        voices = self._voices()
        chord_changed = False
        if self.pos == 0:
            bar = self.bar
            for layer in voices:
                layer.update_level(self)
            self.chord_bars_left -= 1
            if self.chord_bars_left <= 0:
                if bar and bar % 8 == 0:
                    change = self.harmony.drift(self.p["tension"])
                    if change:
                        self._emit("drift", change=change)
                self.harmony.next_chord(self.p["tension"])
                # Busy music changes chords faster; with drums, 1-bar chords get choppy, so favour 2-4.
                e, d = self.p["energy"], self.p["drive"]
                bars = self.rng.choices([1, 2, 4], [(1 + 2 * e) * (1 - d), 6, 3 - 2 * e + 3 * d])[0]
                bars *= -(-12 // self.bar_steps)  # very short bars (2/4, 3/8): keep chords from rushing
                self.chord_bars_left = bars
                self.chord_steps = bars * self.bar_steps
                chord_changed = True
                for layer in voices:
                    if layer.enabled:
                        layer.on_chord(self, offset)
                self._emit("chord", chord=self.harmony.chord_name(), key=self.harmony.key_name(), bars=bars)
            for layer in voices:
                if layer.enabled:
                    layer.on_bar(self, bar, offset)
            self._emit("bar", bar=bar)
        if self._enter:  # a new variant starts now instead of waiting for the next chord
            for layer in self.layers:
                if layer in self._enter and layer.enabled and not chord_changed:
                    layer.on_chord(self, offset)
            self._enter.clear()
        for layer in voices:
            if layer.enabled:
                layer.on_step(self, step, offset)

    def _block(self) -> np.ndarray:
        self._smooth()
        n = BLOCK
        while self.next_step_at < n:
            self._tick(int(self.next_step_at))
            self.next_step_at += self.step_samples
        self.next_step_at -= n

        p = self.p
        duck = self._duck_curve(n)[:, None] * min(1.0, p["drive"] * 1.5)
        dry = np.zeros((n, 2))
        rev_in = np.zeros((n, 2))
        echo_in = np.zeros((n, 2))
        tmp = np.zeros((n, 2))
        gains = {}
        for layer in self._incoming:
            gains[layer] = self._gain_in(layer, n)
        for ghost, _, start, length, g0 in self._ghosts:
            gains[ghost] = g0 * np.cos(self._progress(start, length, n) * (math.pi / 2))
        for layer in self.layers + [g[0] for g in self._ghosts]:
            if not layer.notes:
                continue
            tmp[:] = 0.0
            layer.render(n, tmp)
            if layer.saturate:
                tmp = np.tanh(tmp * layer.saturate) / layer.saturate * 1.5
            if layer.duck:
                tmp = tmp * (1.0 - 0.85 * layer.duck * duck)
            if layer in gains:
                tmp = tmp * gains[layer][:, None]
            dry += tmp
            if layer.reverb:
                rev_in += tmp * layer.reverb
            if layer.echo:
                echo_in += tmp * layer.echo

        space = p["space"]
        beat = 60.0 / p["tempo"]
        echo = self.echo.process(echo_in, beat * 0.75 * (4 / self.beat_steps), 0.3 + 0.35 * space, 0.6)
        rev_in += echo * 0.6
        wet = self.reverb.process(rev_in, 1.5 + 7.0 * space, 0.35 + 0.3 * (1.0 - p["brightness"]))

        mix = dry * (1.0 - 0.35 * space) + echo * (0.4 + 0.4 * space) + wet * (0.5 + 1.3 * space)

        # Slow auto-gain (+-8 dB): generative arrangements vary a lot in density.
        a = 1.0 - math.exp(-n / (self.sr * 6.0))
        self.ms += (float((mix**2).mean()) - self.ms) * a
        target = self.level_target * 10 ** (4.0 * p["drive"] / 20)  # driving music sits a bit hotter
        want = min(2.5, max(0.4, target / math.sqrt(self.ms + 1e-12)))
        self.agc += (want - self.agc) * a

        end = self.samples + n
        self._ghosts = [g for g in self._ghosts if end < g[2] + g[3]]
        self._incoming = {k: v for k, v in self._incoming.items() if end < v[0] + v[1]}

        self.fade = min(1.0, self.fade + n / (self.sr * 3.0))
        out = np.tanh(mix * (self.agc * p["volume"] * self.fade))
        self.samples += n
        return out.astype(np.float32)
