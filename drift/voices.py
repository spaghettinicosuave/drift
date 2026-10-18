"""Layers: independent musical parts that listen to the shared harmony and clock.

Each layer decides *what* to play; dsp.Note does the sound. Layers fade in and out
on their own (a slow random "wander" on top of the global `energy` parameter), so
the arrangement keeps changing without anyone scripting it.

Every layer has a few VARIANTS: named settings for both its sound and its playing
style. They are independent of the seed: pick them, or let reroll_variants() roll dice.

To make a new layer, subclass Layer and override any of on_bar / on_chord / on_step.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from . import dsp
from .dsp import Envelope, Note

if TYPE_CHECKING:
    from .engine import Engine


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def euclid(hits: int, steps: int = 16, rotate: int = 0) -> list[bool]:
    """Evenly spread `hits` onsets over `steps` (Euclidean rhythm)."""
    pat = [((i * hits) % steps) < hits for i in range(steps)]
    return pat[-rotate:] + pat[:-rotate] if rotate else pat


# Mix attributes a variant may override.
_MIX = ("reverb", "echo", "duck", "saturate", "threshold")

# Fixed resonances for acoustic-sounding variants: (centre Hz, width Hz, boost).
HURDY_BODY = ((420.0, 180.0, 1.5), (1100.0, 350.0, 2.5), (2400.0, 700.0, 1.5))
VIOL_BODY = ((280.0, 120.0, 1.2), (900.0, 300.0, 1.0), (2500.0, 800.0, 1.0))
FIDDLE_BODY = ((500.0, 250.0, 1.5), (1500.0, 500.0, 2.0), (3000.0, 1000.0, 1.5))
CHANTER_BODY = ((1000.0, 400.0, 3.0), (2500.0, 700.0, 2.0))
DULCIMER_BODY = ((400.0, 200.0, 0.6), (2000.0, 800.0, 0.5))


class Layer:
    name = "layer"
    # Default mix settings; override per instance via constructor kwargs.
    gain = 1.0
    reverb = 0.3  # send amount
    echo = 0.0  # send amount
    threshold = 0.0  # value of `param` at which this layer starts to appear
    param = "energy"  # which engine parameter brings this layer in
    wander_depth = 1.0  # how much the layer drifts in and out on its own
    duck = 0.0  # sidechain: how much this layer dips under the kick (scaled by `drive`)
    saturate = 0.0  # tanh drive on the layer's output; 0 = clean
    max_notes = 32

    # name -> settings read by the layer's code. "gain" and the _MIX keys are applied
    # automatically. The first entry is the default.
    VARIANTS: dict[str, dict] = {"default": {}}

    def __init__(self, *, gain=None, reverb=None, echo=None, threshold=None, variant=None, enabled=True):
        self._overrides = {k: v for k, v in (("reverb", reverb), ("echo", echo), ("threshold", threshold))
                           if v is not None}
        if gain is not None:
            self.gain = gain
        self.enabled = enabled
        self.notes: list[Note] = []
        self.level = 0.0
        self.wander = 0.0
        self.fixed_variant = variant is not None  # reroll_variants() leaves it alone
        self.variant = ""
        self.set_variant(variant or next(iter(self.VARIANTS)))

    # ---- variants

    @property
    def v(self) -> dict:
        return self.VARIANTS[self.variant]

    def set_variant(self, name: str) -> None:
        if name not in self.VARIANTS:
            raise KeyError(f"{self.name}: unknown variant {name!r}; expected one of {list(self.VARIANTS)}")
        self.variant = name
        for k in _MIX:
            value = self._overrides.get(k, self.v.get(k, getattr(type(self), k)))
            setattr(self, k, value)
        self.on_variant()

    def on_variant(self) -> None:
        """Called after the variant changes (e.g. to rebuild cached sounds)."""

    def handoff(self) -> Layer:
        """Split off a copy that keeps playing the current variant and takes the sounding
        notes with it, so the engine can fade it out while this layer fades in with a new one."""
        ghost = copy.copy(self)
        for k, val in vars(self).items():
            if isinstance(val, (list, dict, set)) and k != "_overrides":
                setattr(ghost, k, copy.copy(val))
        self.notes = []
        self.on_handoff()
        return ghost

    def on_handoff(self) -> None:
        """Forget references to notes that now belong to the outgoing copy (see handoff)."""

    # ---- activity

    @property
    def active(self) -> bool:
        return self.enabled and self.level > 0.05

    def update_level(self, ctx: Engine) -> None:
        """Called once per bar."""
        r = ctx.rng
        self.wander = clamp(self.wander * 0.92 + r.uniform(-0.12, 0.12), -0.5, 0.5)
        target = clamp((ctx.p[self.param] - self.threshold) * 2.5 + self.wander * self.wander_depth)
        self.level += (target - self.level) * 0.3

    # ---- hooks (offset = sample offset within the current block)

    def on_bar(self, ctx: Engine, bar: int, offset: int) -> None: ...

    def on_chord(self, ctx: Engine, offset: int) -> None: ...

    def on_step(self, ctx: Engine, step: int, offset: int) -> None: ...

    # ---- notes

    def play(self, ctx: Engine, midi: float, amp: float, env: Envelope, timbre: dsp.Wave | dsp.Partials, offset: int, **kw) -> Note:
        amp *= self.gain * self.v.get("gain", 1.0)
        note = Note(dsp.midi_to_hz(midi), amp, env, timbre, ctx.sr, delay=offset, rng=ctx.nprng, **kw)
        self.notes.append(note)
        if len(self.notes) > self.max_notes:
            self.notes[0].release(0)
        return note

    def play_sample(self, sample: dsp.Sample, amp: float, offset: int, pan: float = 0.0) -> dsp.SampleNote:
        note = dsp.SampleNote(sample, amp * self.gain * self.v.get("gain", 1.0), pan=pan, delay=offset)
        self.notes.append(note)
        return note

    def ks(self, ctx: Engine, midi: int, **kw) -> dsp.Sample:
        """Cached Karplus-Strong plucked string (lute, dulcimer...). kw go to dsp.ks_pluck."""
        cache = self.__dict__.setdefault("_ks", {})
        key = (midi, tuple(sorted(kw.items())))
        if key not in cache:
            if len(cache) > 96:
                cache.clear()
            cache[key] = dsp.ks_pluck(ctx.sr, dsp.midi_to_hz(midi), ctx.nprng, **kw)
        return cache[key]

    def release_all(self, offset: int = 0) -> None:
        for n in self.notes:
            n.release(offset)

    def render(self, n: int, out: np.ndarray) -> None:
        for note in self.notes:
            note.render(n, out)
        self.notes = [x for x in self.notes if not x.finished]


# =========================================================================== layers


class Drone(Layer):
    """Low, slow root note. Sometimes holds as a pedal tone through chord changes."""

    name = "drone"
    gain = 0.55
    duck = 0.6
    reverb = 0.25
    threshold = -1.0  # always present

    VARIANTS = {
        "warm": dict(
            center=38, attack=4.0, trem=(0.07, 0.25),
            timbre=lambda b: dsp.wave(6, 0.25 + 0.3 * b, unison=((0, 1.0), (5, 0.6))),
        ),
        "organ": dict(  # root + fifth, hollow odd harmonics, slow rotary wobble
            center=38, attack=2.5, trem=(0.8, 0.12), fifth=True, gain=1.28,
            timbre=lambda b: dsp.Wave([1, 0.5, 0.6, 0.2, 0.35, 0.05, 0.2, 0.0, 0.1 * b], unison=((0, 1.0), (3, 0.7))),
        ),
        "choir": dict(  # vowel-like formants, gentle vibrato, higher up
            center=45, attack=5.0, trem=(0.1, 0.2), vib=(0.35, 0.002), reverb=0.45, gain=2.75,
            timbre=lambda b: dsp.Wave(dsp.formant_amps(14, ((3.0, 1.2), (6.0 + 3 * b, 1.8)), tilt=1.3),
                                      unison=((-9, 1.0), (0, 0.8), (9, 1.0))),
        ),
        "sub": dict(  # deep and round, slow breathing pulse, a faint octave on top
            center=31, attack=3.0, trem=(0.22, 0.45), octave=True, gain=0.69,
            timbre=lambda b: dsp.Wave([1.0, 0.18 + 0.15 * b, 0.05]),
        ),
        "hurdy": dict(  # hurdy-gurdy drone strings: stay on the tonic, rasp, rhythmic trompette buzz
            gain=0.92, center=43, attack=0.8, trem=(0.6, 0.06), fifth=True, tonic=True, buzz=True, grit=0.12, reverb=0.2,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(28, 0.65 + 0.25 * b), unison=((0, 1.0), (3, 0.6)),
                                      formants=HURDY_BODY),
        ),
        "brasspad": dict(  # low synth-brass pad: spectrum opens on attack then sustains wide; italo sub-anchor
            center=36, attack=1.2, trem=(0.18, 0.06), fifth=True, gain=1.1, reverb=0.35,
            timbre=lambda b: dsp.Wave(
                dsp.harmonic_amps(4, 0.1),
                unison=((-12, 1.0), (-5, 0.8), (0, 0.9), (5, 0.8), (12, 1.0)),
                dark=dsp.harmonic_amps(22, 0.5 + 0.4 * b),
                morph=0.6,
            ),
        ),
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.current: list[Note] = []
        self.pitch: int | None = None

    def on_handoff(self):
        self.current = []
        self.pitch = None

    def on_chord(self, ctx, offset):
        h, v = ctx.harmony, self.v
        midi = h.near(0 if v.get("tonic") else h.degree, v["center"])
        if any(not n.finished for n in self.current):
            if midi == self.pitch:
                return
            in_chord = self.pitch is not None and self.pitch in h.chord_midis(self.pitch, self.pitch)
            if in_chord and ctx.rng.random() < 0.5:
                return  # pedal
            for n in self.current:
                n.release(offset)
        self.current = []
        if not self.active:
            return
        timbre = v["timbre"](ctx.p["brightness"])
        amp = 0.5 * self.level * (1.0 - 0.6 * ctx.p["drive"])
        env = Envelope(v["attack"], 1.0, 1.0, 7.0)
        pan = ctx.rng.uniform(-0.15, 0.15)
        extra = {"tremolo": v["trem"], "vibrato": v.get("vib", (0.0, 0.0)), "grit": v.get("grit", 0.0)}
        self.pitch = midi
        self.current.append(self.play(ctx, midi, amp, env, timbre, offset, pan=pan, **extra))
        if v.get("fifth"):
            self.current.append(self.play(ctx, midi + 7, amp * 0.5, env, timbre, offset, pan=-pan, **extra))
        if v.get("octave"):
            self.current.append(self.play(ctx, midi + 12, amp * 0.18, env, timbre, offset, pan=-pan, **extra))

    def on_step(self, ctx, step, offset):
        """Hurdy-gurdy trompette: the player's wrist strokes make the bridge buzz in rhythm."""
        if not self.v.get("buzz") or not self.active or self.pitch is None or not self.current:
            return
        r = ctx.rng
        _, k, n = ctx.beat_at(ctx.pos)
        if n == 6:  # dotted beat (6/8): chuff on its 3rd eighth, lighter on the 2nd
            weight = {4: 1.0, 2: 0.45}.get(k, 0.0)
        else:  # offbeat 8ths
            weight = 1.0 if k == n // 2 else 0.0
        crank = ctx.p["energy"]
        if weight == 0.0 or r.random() > weight * (0.2 + 0.9 * crank):
            return
        cache = self.__dict__.setdefault("_buzz", {})
        midi = self.pitch + 12
        if midi not in cache:
            cache[midi] = dsp.buzz(ctx.sr, dsp.midi_to_hz(midi), ctx.nprng)
        self.play_sample(cache[midi], 0.22 * self.level * weight * (0.5 + 0.5 * crank), offset, pan=0.1)

class Pad(Layer):
    """Sustained chords with long swells, voice-led around a slowly wandering centre."""

    name = "pad"
    gain = 0.5
    duck = 0.8
    reverb = 0.5
    echo = 0.05
    threshold = 0.05

    VARIANTS = {
        "strings": dict(
            attack=(1.5, 3.5), trem=0.3, voices=4, spread=4.5,
            timbre=lambda b, a: dsp.wave(8 + int(16 * b), 0.2 + 0.7 * b, unison=((-7, 1.0), (0, 0.8), (7, 1.0))),
        ),
        "glass": dict(  # hollow odd harmonics, quicker swell, an octave shimmer on top
            attack=(0.6, 1.5), trem=0.12, trem_rate=(2.5, 4.0), voices=4, spread=4.0, octave=True, gain=0.56,
            timbre=lambda b, a: dsp.Wave(dsp.harmonic_amps(5, 0.3 + 0.6 * b, odd_only=True), unison=((-3, 1.0), (3, 1.0))),
        ),
        "breath": dict(  # nearly pure tones, very slow, deep tremolo; wide and sparse
            attack=(3.0, 5.0), trem=0.5, voices=3, spread=6.0, gain=1.0,
            timbre=lambda b, a: dsp.Wave([1.0, 0.15 + 0.1 * b, 0.05], unison=((-4, 1.0), (4, 1.0))),
        ),
        "viols": dict(  # a bowed consort: nasal body, a little bow noise
            gain=0.7, attack=(1.0, 2.5), trem=0.12, voices=4, spread=4.5, grit=0.06,
            timbre=lambda b, a: dsp.Wave(dsp.harmonic_amps(20, 0.5 + 0.3 * b), unison=((-6, 1.0), (6, 1.0)),
                                         formants=VIOL_BODY),
        ),
        "bloom": dict(  # starts dark and opens up as it swells, like a filter sweep
            attack=(2.0, 4.0), trem=0.2, voices=4, spread=4.5, gain=1.1,
            timbre=lambda b, a: dsp.Wave(dsp.harmonic_amps(3, 0.1), unison=((-8, 1.0), (8, 1.0)),
                                         dark=dsp.harmonic_amps(10 + int(14 * b), 0.5 + 0.45 * b), morph=a * 0.8),
        ),
        "italo": dict(  # dense chorus supersaw: 7 detuned saws beating slowly against each other, hard stereo spread
            attack=(0.8, 1.8), trem=0.04, trem_rate=(0.08, 0.15), voices=4, spread=5.0, gain=0.72,
            timbre=lambda b, a: dsp.Wave(
                dsp.harmonic_amps(28, 0.45 + 0.45 * b),
                unison=((-22, 0.7), (-14, 1.0), (-7, 0.9), (0, 1.0), (7, 0.9), (14, 1.0), (22, 0.7)),
            ),
        ),
    }

    def __init__(self, center: float = 60, **kw):
        super().__init__(**kw)
        self.home = center
        self.center = center

    def on_chord(self, ctx, offset):
        self.release_all(offset)
        if not self.active:
            return
        h, r, p, v = ctx.harmony, ctx.rng, ctx.p, self.v
        self.center = clamp(self.center + r.uniform(-2.5, 2.5), self.home - 5, self.home + 6)
        degs = h.chord[: v["voices"]]
        mids = sorted({h.near(d, self.center + (i - (len(degs) - 1) / 2) * v["spread"]) for i, d in enumerate(degs)})
        lo, hi = v["attack"]
        attack = lo + (hi - lo) * (1.0 - p["energy"]) + r.uniform(0, 1.5)
        timbre = v["timbre"](p["brightness"], attack)
        amp = 0.32 * self.level / len(mids) ** 0.5
        env = Envelope(attack, 2.0, 0.8, 5.0)
        for i, m in enumerate(mids):
            trem = (r.uniform(*v.get("trem_rate", (0.08, 0.3))), v["trem"])
            pan = (i / max(1, len(mids) - 1) - 0.5) * 0.9
            self.play(ctx, m, amp, env, timbre, offset, pan=pan, tremolo=trem, grit=v.get("grit", 0.0))
        if v.get("octave"):
            self.play(ctx, mids[-1] + 12, amp * 0.35, env, timbre, offset, pan=r.uniform(-0.5, 0.5))


class Swell(Layer):
    """High, soft tones that rise across the whole chord and fade as the next one arrives."""

    name = "swell"
    gain = 0.45
    duck = 0.5
    reverb = 0.7
    echo = 0.15
    threshold = 0.25

    VARIANTS = {
        "flute": dict(
            timbre=lambda b, d: dsp.wave(5, 0.3 + 0.4 * b, odd_only=True, unison=((-4, 1.0), (4, 1.0))),
        ),
        "strings": dict(  # a rising string section
            gain=1.67, timbre=lambda b, d: dsp.wave(16, 0.4 + 0.4 * b, unison=((-9, 1.0), (0, 0.7), (9, 1.0))),
        ),
        "shimmer": dict(  # tone + octaves, with a fast flutter
            trem=(6.0, 0.25), gain=1.43,
            timbre=lambda b, d: dsp.Wave([1.0, 0.1], unison=((0, 1.0), (1200, 0.5), (2400, 0.2 + 0.2 * b))),
        ),
        "rise": dict(  # brassy: the spectrum opens over the whole swell
            gain=1.33, timbre=lambda b, d: dsp.Wave(dsp.harmonic_amps(3, 0.1), unison=((-6, 1.0), (6, 1.0)),
                                                    dark=dsp.harmonic_amps(20, 0.55 + 0.4 * b), morph=d * 0.45),
        ),
    }

    def on_chord(self, ctx, offset):
        self.release_all(offset)
        if not self.active or ctx.rng.random() > 0.15 + 0.5 * self.level:
            return
        h, r, v = ctx.harmony, ctx.rng, self.v
        tones = h.chord_midis(67, 86)
        if not tones:
            return
        picks = r.sample(tones, min(len(tones), r.choice([1, 2, 2])))
        dur = ctx.chord_seconds
        timbre = v["timbre"](ctx.p["brightness"], dur)
        for m in picks:
            self.play(
                ctx, m, 0.12 * self.level, Envelope(dur * 0.75, 1.0, 1.0, 3.5), timbre, offset,
                duration=dur * 0.9, pan=r.uniform(-0.7, 0.7), vibrato=(0.2, 0.0015),
                tremolo=v.get("trem", (0.0, 0.0)),
            )


class Arp(Layer):
    """Notes on a Euclidean rhythm, walking through chord tones."""

    name = "arp"
    gain = 0.5
    duck = 0.3
    reverb = 0.35
    echo = 0.35
    threshold = 0.3

    VARIANTS = {
        "pluck": dict(
            low=62, high=86, walk="random",
            timbre=lambda b, d: dsp.pluck(12, 0.3 + 0.6 * b, d),
        ),
        "marimba": dict(  # wooden bars, lower and drier
            low=55, high=79, walk="random", decay=0.6, reverb=0.25, gain=1.36,
            timbre=lambda b, d: dsp.Partials([1.0, 3.93, 9.2], [1.0, 0.25 + 0.2 * b, 0.08 * b], [0.0, 6.0, 14.0]),
        ),
        "glass": dict(  # high, glassy, long ringing
            low=72, high=96, walk="random", decay=1.6, reverb=0.5, gain=0.7,
            timbre=lambda b, d: dsp.struck([1.0, 2.0, 3.01, 4.2, 5.43], [1.0, 0.5, 0.3, 0.15, 0.08], 0.4 + 0.5 * b, 0.6),
        ),
        "lute": dict(  # plucked gut strings with a wooden body
            gain=3.16, low=55, high=79, walk="random", reverb=0.3, echo=0.15, ks=dict(decay=1.6, bright=0.45),
            timbre=None,
        ),
        "sequence": dict(  # gated saw arpeggiator, up and down in order
            low=57, high=81, walk="updown", gate=0.7, echo=0.45, gain=2.48,
            timbre=lambda b, d: dsp.Wave(dsp.harmonic_amps(20, 0.5 + 0.4 * b), unison=((-5, 1.0), (5, 1.0)),
                                         dark=dsp.harmonic_amps(5, 0.2), morph=0.15),
        ),
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.pattern = euclid(5, 16)
        self.bars_left = 0
        self.pos = 0.5  # position in the chord-tone pool, 0..1
        self.dir = 1

    def on_bar(self, ctx, bar, offset):
        self.bars_left -= 1
        if self.bars_left <= 0 or len(self.pattern) != ctx.bar_steps:
            r = ctx.rng
            hits = int(2 + round(ctx.p["energy"] * 6 * self.level)) + r.choice([-1, 0, 0, 1])
            if self.v["walk"] == "updown":
                hits += 3
            self.pattern = euclid(max(1, min(12, hits)), ctx.bar_steps, r.randrange(ctx.bar_steps))
            self.bars_left = r.choice([2, 4, 4, 8])

    def _next_index(self, r, n: int) -> int:
        if self.v["walk"] == "updown":
            i = int(self.pos * n) + self.dir
            if i >= n - 1 or i <= 0:
                self.dir = -self.dir
                i = max(0, min(n - 1, i))
            self.pos = (i + 0.5) / n
            return i
        self.pos = clamp(self.pos + r.choice([-0.25, -0.12, 0.12, 0.12, 0.25]) + (0.5 - self.pos) * 0.1)
        return min(n - 1, int(self.pos * n))

    def on_step(self, ctx, step, offset):
        if not self.active or not self.pattern[ctx.pos % len(self.pattern)]:
            return
        r, v = ctx.rng, self.v
        prob = 0.55 + 0.4 * self.level
        if ctx.flags.get("melody"):
            prob *= 0.5
        if r.random() > prob:
            return
        h, p = ctx.harmony, ctx.p
        pool = h.chord_midis(v["low"], v["high"])
        if not pool:
            return
        midi = pool[self._next_index(r, len(pool))]
        if v["walk"] == "random" and r.random() < 0.1:
            scale = h.scale_midis(midi - 2, midi + 2)
            midi = r.choice(scale) if scale else midi
        decay = (0.35 + 0.9 * (1.0 - p["energy"])) * v.get("decay", 1.0)
        accent = 1.0 if ctx.beat_at(ctx.pos)[1] == 0 else r.uniform(0.5, 0.85)
        if "ks" in v:
            bright = round(v["ks"]["bright"] + 0.4 * (p["brightness"] - 0.5), 1)
            sample = self.ks(ctx, midi, decay=v["ks"]["decay"], bright=bright)
            self.play_sample(sample, 0.16 * self.level * accent, offset, pan=r.uniform(-0.6, 0.6))
            return
        timbre = v["timbre"](p["brightness"], decay)
        if "gate" in v:
            env, kw = Envelope(0.004, 0.2, 0.6, 0.08), {"duration": ctx.step_seconds * v["gate"]}
        else:
            env, kw = Envelope(0.004, decay, 0.0, 0.3), {}
        self.play(ctx, midi, 0.16 * self.level * accent, env, timbre, offset, pan=r.uniform(-0.6, 0.6), **kw)


class Bells(Layer):
    """Struck, ringing tones. Mostly heard through the reverb."""

    name = "bells"
    gain = 0.5
    reverb = 0.8
    echo = 0.3
    threshold = 0.15

    VARIANTS = {
        "chime": dict(
            low=72, high=93, decay=1.0, prob=0.12, counts=[1, 1, 2, 3], style="scatter",
            timbre=lambda b: dsp.bell(b),
        ),
        "glock": dict(  # bright metal bars, little rising figures
            low=79, high=98, decay=0.45, prob=0.15, counts=[2, 3, 3, 4], style="run", gain=0.54,
            timbre=lambda b: dsp.struck([1.0, 2.76, 5.4, 8.93], [1.0, 0.45, 0.2, 0.1], 0.5 + 0.5 * b, 1.3),
        ),
        "gong": dict(  # rare, low, long and dark
            low=45, high=62, decay=2.2, prob=0.06, counts=[1], style="scatter", gain=1.32, reverb=0.6,
            timbre=lambda b: dsp.struck([1.0, 1.48, 1.93, 2.5, 2.97, 3.6], [1.0, 0.6, 0.5, 0.35, 0.25, 0.15], 0.3 * b, 0.35),
        ),
        "dulcimer": dict(  # hammered dulcimer: doubled strings, quick runs
            gain=3.05, low=67, high=91, decay=1.0, prob=0.2, counts=[3, 4, 5, 6], style="run", reverb=0.4,
            ks=dict(decay=2.2, bright=0.85, courses=2, detune=4.0, body=DULCIMER_BODY), timbre=None,
        ),
        "kalimba": dict(  # soft thumb-piano runs
            low=64, high=86, decay=0.35, prob=0.2, counts=[3, 4, 5], style="run", gain=0.53, reverb=0.5,
            timbre=lambda b: dsp.struck([1.0, 3.0, 5.2], [1.0, 0.2 + 0.2 * b, 0.08], 0.4 + 0.4 * b, 1.5),
        ),
    }

    def on_bar(self, ctx, bar, offset):
        r, v = ctx.rng, self.v
        if not self.active or r.random() > v["prob"] + 0.2 * self.level:
            return
        tones = ctx.harmony.chord_midis(v["low"], v["high"])
        if not tones:
            return
        n = r.choice(v["counts"])
        B = ctx.bar_steps
        if "ks" in v:
            ks = v["ks"]

            def strike(midi, amp, pos, pan):
                self.play_sample(self.ks(ctx, midi, **ks), amp, pos, pan=pan)
        else:
            timbre = v["timbre"](ctx.p["brightness"])
            env = Envelope(0.002, (2.5 + 2 * ctx.p["space"]) * v["decay"], 0.0, 1.0)

            def strike(midi, amp, pos, pan):
                self.play(ctx, midi, amp, env, timbre, pos, pan=pan)
        if v["style"] == "run":  # consecutive 8ths walking up (or down) the chord
            i = r.randrange(len(tones))
            step, at = r.choice([1, 1, -1]), r.randrange(0, max(1, B - 2 * n + 1), 2)
            for k in range(n):
                midi = tones[(i + step * k) % len(tones)]
                pos = offset + int((at + 2 * k) * ctx.step_samples)
                strike(midi, 0.13 * self.level * r.uniform(0.6, 1.0), pos, r.uniform(-0.5, 0.5))
            return
        for _ in range(n):
            pos = offset + int(r.randrange(0, B, 2) * ctx.step_samples)
            strike(r.choice(tones), 0.13 * self.level * r.uniform(0.6, 1.0), pos, r.uniform(-0.8, 0.8))


class Melody(Layer):
    """Short motifs that come back varied, then leave room. Pitches are scale degrees,
    resolved against the harmony at play time so they always fit the current chord."""

    name = "melody"
    gain = 0.5
    reverb = 0.45
    echo = 0.4
    threshold = 0.2

    VARIANTS = {
        "flute": dict(
            center=74, vib=(4.8, 0.003), attack=0.05, lens=([2, 3, 4, 6, 8], [2, 2, 4, 2, 2]),
            timbre=lambda b: dsp.Wave([1.0, 0.12 + 0.2 * b, 0.05 + 0.1 * b, 0.02 * b]),
        ),
        "whistle": dict(  # pure and high, wider vibrato, slower notes
            center=81, vib=(5.5, 0.006), attack=0.08, lens=([3, 4, 6, 8], [2, 4, 3, 3]), gain=0.83,
            timbre=lambda b: dsp.Wave([1.0, 0.03]),
        ),
        "reed": dict(  # clarinet-ish odd harmonics, lower, little vibrato
            center=69, vib=(4.5, 0.0015), attack=0.04, lens=([2, 3, 4, 6, 8], [2, 2, 4, 2, 2]), gain=0.8,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(6, 0.35 + 0.4 * b, odd_only=True)),
        ),
        "hurdy": dict(  # hurdy-gurdy chanter: nasal, raspy, quick key clicks and grace notes
            gain=1.84, center=72, vib=(0.0, 0.0), attack=0.012, lens=([2, 2, 3, 4, 6], [4, 3, 2, 3, 1]), ornament=0.45,
            grit=0.12, echo=0.15, reverb=0.3,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(24, 0.7 + 0.2 * b), unison=((0, 1.0), (4, 0.5)),
                                      formants=CHANTER_BODY),
        ),
        "fiddle": dict(  # bowed, singing vibrato, ornamented
            gain=1.16, center=74, vib=(5.5, 0.004), attack=0.05, lens=([2, 2, 3, 4, 6], [4, 3, 2, 3, 1]), ornament=0.35,
            grit=0.08, echo=0.2,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(24, 0.6 + 0.3 * b), formants=FIDDLE_BODY),
        ),
        "recorder": dict(  # breathy wooden flute, high
            gain=0.87, center=79, vib=(5.0, 0.002), attack=0.03, lens=([2, 3, 4, 6, 8], [3, 2, 4, 2, 1]), ornament=0.25,
            grit=0.05, echo=0.25,
            timbre=lambda b: dsp.Wave([1.0, 0.08, 0.12 + 0.05 * b, 0.02]),
        ),
        "lead": dict(  # bright synth lead, busier rhythms, lots of echo
            center=74, vib=(5.0, 0.002), attack=0.01, lens=([1, 2, 2, 3, 4], [2, 4, 3, 2, 2]), echo=0.55, gain=1.32,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(16, 0.5 + 0.4 * b), unison=((-6, 1.0), (6, 1.0))),
        ),
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.motif: list[tuple[int, int]] | None = None  # (degree offset, length in steps)
        self.queue: dict[int, tuple[int, int]] = {}  # abs step -> (degree, steps)

    def _new_motif(self, r, bar_steps: int = 16) -> list[tuple[int, int]]:
        degs = [0]
        for _ in range(r.randint(2, 5)):
            degs.append(degs[-1] + r.choice([-2, -1, -1, 1, 1, 2, 3, -3]))
        out, total = [], 0
        lens, weights = self.v["lens"]
        for d in degs:
            ln = r.choices(lens, weights)[0]
            if total + ln > bar_steps:
                break
            out.append((d, ln))
            total += ln
        return out or [(0, 8)]

    def _vary(self, r, motif):
        op = r.choice(["transpose", "invert", "retro", "tail", "same"])
        if op == "transpose":
            k = r.choice([-2, 2, 4])
            return [(d + k, ln) for d, ln in motif]
        if op == "invert":
            return [(-d, ln) for d, ln in motif]
        if op == "retro":
            degs = [d for d, _ in motif][::-1]
            return list(zip(degs, [ln for _, ln in motif]))
        if op == "tail":
            return motif[:-1] + [(motif[-1][0] + r.choice([-1, 1, 2]), motif[-1][1])]
        return list(motif)

    def on_variant(self):
        self.motif = None  # rhythms differ per variant

    def on_bar(self, ctx, bar, offset):
        # Drop notes whose step already passed (the layer was disabled mid-phrase);
        # otherwise the leftover queue would block new phrases forever.
        self.queue = {t: ev for t, ev in self.queue.items() if t >= ctx.step}
        ctx.flags["melody"] = bool(self.queue)
        if bar % 4 or self.queue or not self.active:
            return
        r = ctx.rng
        if r.random() > 0.25 + 0.5 * self.level:
            return
        if self.motif is None or r.random() < 0.3:
            self.motif = self._new_motif(r, ctx.bar_steps)
        else:
            self.motif = self._vary(r, self.motif)
        h = ctx.harmony
        anchor = r.choice(h.chord)
        start = ctx.step + r.choice([0, 0, 2, 4])
        for phrase, motif in enumerate([self.motif, self._vary(r, self.motif)]):
            t = start + phrase * 2 * ctx.bar_steps
            for d, ln in motif:
                self.queue[t] = (anchor + d, ln)
                t += ln

    def on_step(self, ctx, step, offset):
        ev = self.queue.pop(step, None)
        if ev is None:
            return
        deg, ln = ev
        h, r, v = ctx.harmony, ctx.rng, self.v
        midi = h.near(deg, v["center"])
        if ctx.beat_at(ctx.pos)[1] == 0:
            chord = h.chord_midis(midi - 2, midi + 2)
            if chord and midi not in chord:
                midi = min(chord, key=lambda m: abs(m - midi))
        dur = ln * ctx.step_seconds * 0.9
        timbre = v["timbre"](ctx.p["brightness"])
        amp = 0.13 * self.level * r.uniform(0.8, 1.0)
        pan = r.uniform(-0.3, 0.3)
        kw = {"vibrato": v["vib"], "grit": v.get("grit", 0.0)}
        if v.get("ornament") and ln >= 3 and r.random() < v["ornament"]:
            # Folk grace note: a flick from the scale tone above, just before the beat lands.
            above = h.scale_midis(midi + 1, midi + 3)
            if above:
                grace = min(0.07, ctx.step_seconds * 0.6)
                self.play(ctx, above[0], amp * 0.8, Envelope(0.005, 0.05, 0.8, 0.03), timbre, offset,
                          duration=grace, pan=pan, **kw)
                offset += int(grace * ctx.sr)
                dur -= grace
        self.play(ctx, midi, amp, Envelope(v["attack"], 0.4, 0.7, 0.5 + 0.1 * ln), timbre, offset,
                  duration=dur, pan=pan, **kw)


# =========================================================================== drive layers
# These follow `drive` instead of `energy`, so they can be blended over any preset.


class Drums(Layer):
    """Kick / snare / clap / hats. Sparse pulse -> boots-and-cats -> four-on-the-floor with fills."""

    name = "drums"
    param = "drive"
    threshold = 0.08
    wander_depth = 0.25
    gain = 0.28
    reverb = 0.1
    max_notes = 64

    # Kits: kick (punch, decay, drive), snare (tone, decay, brightness), hats (closed, open, brightness),
    # mix = per-voice level multipliers, swing = delay of off-16ths as a fraction of a step.
    # hat_kind: sound used for closed hits ("hat", "tambourine", "rim").
    # accent_kind: sound used for offbeat accent slot ("ohat", "cowbell", "rim", "tambourine").
    VARIANTS = {
        "punchy": dict(kick=(1.0, 0.35, 1.6), snare=(190, 0.16, 0.5), hats=(0.04, 0.22, 0.5),
                       mix=dict(snare=1.0, clap=1.0, hat=1.0, ohat=1.0), swing=0.0),
        "808": dict(kick=(0.5, 0.75, 1.0), snare=(230, 0.11, 0.35), hats=(0.03, 0.3, 0.75),
                    mix=dict(snare=0.6, clap=1.3, hat=1.0, ohat=0.9), swing=0.0, gain=1.06),
        "dusty": dict(kick=(0.6, 0.25, 1.2), snare=(170, 0.2, 0.1), hats=(0.035, 0.15, 0.15),
                      mix=dict(snare=1.1, clap=0.4, hat=0.8, ohat=0.6), swing=0.16, reverb=0.18, gain=1.25),
        "industrial": dict(kick=(1.6, 0.3, 4.0), snare=(240, 0.25, 0.9), hats=(0.05, 0.3, 1.0),
                           mix=dict(snare=1.0, clap=0.6, hat=1.0, ohat=1.0), swing=0.0, gain=0.69),
        # Frame drum (low "doum" + slapped "tek"), hand claps and tambourine. No sidechain pumping.
        "tavern": dict(gain=2.4, kit="tavern", sidechain=False, mix=dict(snare=1.1, clap=0.5, hat=1.3, ohat=1.0),
                       swing=0.0, reverb=0.2),
        # Italo: punchy kick, dry clap on 2+4, tight closed hats, rim on offbeats.
        "italo": dict(kick=(0.8, 0.55, 1.4), snare=(210, 0.10, 0.7), hats=(0.025, 0.18, 0.9),
                      mix=dict(snare=0.5, clap=1.5, hat=0.9, ohat=1.2), swing=0.0,
                      hat_kind="hat", accent_kind="rim", gain=1.1),
    }

    # Where things go, per (steps per bar, steps per beat). 4/4 is the original groove.
    GROOVES = {
        (16, 4): dict(extra=10, boots=(0, 8), back=(4, 12), syncop=[6, 10, 11, 14], beats=(0, 4, 8, 12),
                      ghosts=[3, 7, 9, 11, 15], fills=[[12, 13, 14, 15], [10, 12, 13, 14, 15], [14, 15]]),
        (12, 6): dict(extra=9, boots=(0,), back=(6,), syncop=[4, 10, 11], beats=(0, 6),  # 6/8 jig
                      ghosts=[2, 5, 8, 11], fills=[[8, 9, 10, 11], [6, 8, 10, 11], [10, 11]]),
        (12, 4): dict(extra=8, boots=(0,), back=(4, 8), syncop=[6, 10], beats=(0, 4, 8),  # 3/4 waltz
                      ghosts=[3, 7, 11], fills=[[8, 9, 10, 11], [6, 8, 10, 11], [10, 11]]),
    }

    def __init__(self, **kw):
        self.kit: dict | None = None
        self.kicks: dict[int, dsp.Sample] = {}
        super().__init__(**kw)
        self.pat: dict[str, list[float]] = {}

    def on_variant(self):
        self.kit = None
        self.kicks = {}

    # Builders for hat and accent slots — swap via hat_kind / accent_kind in a variant.
    _HAT_BUILDERS = {
        "hat":        lambda sr, g, v: dsp.hat(sr, g, v["hats"][0], v["hats"][2]),
        "tambourine": lambda sr, g, v: dsp.tambourine(sr, g, 0.05),
        "rim":        lambda sr, g, v: dsp.rim(sr, g),
    }
    _ACCENT_BUILDERS = {
        "ohat":       lambda sr, g, v: dsp.hat(sr, g, v["hats"][1], v["hats"][2]),
        "cowbell":    lambda sr, g, v: dsp.cowbell(sr, g, 0.18),
        "rim":        lambda sr, g, v: dsp.rim(sr, g),
        "tambourine": lambda sr, g, v: dsp.tambourine(sr, g, 0.18, shake=True),
    }

    def _build(self, ctx):
        sr, g, v = ctx.sr, ctx.nprng, self.v
        if v.get("kit") == "tavern":
            self.kit = {
                "snare": dsp.frame_drum(sr, g, 190.0, 0.07, 0.7),  # slapped rim: "tek"
                "clap": dsp.clap(sr, g, 0.1),
                "hat": dsp.tambourine(sr, g, 0.05),
                "ohat": dsp.tambourine(sr, g, 0.18, shake=True),
                "crash": dsp.tambourine(sr, g, 0.6),
            }
            return
        tone, sdecay, sbright = v["snare"]
        hat_build   = self._HAT_BUILDERS[v.get("hat_kind", "hat")]
        accent_build = self._ACCENT_BUILDERS[v.get("accent_kind", "ohat")]
        self.kit = {
            "snare": dsp.snare(sr, g, tone, sdecay, sbright),
            "clap":  dsp.clap(sr, g),
            "hat":   hat_build(sr, g, v),
            "ohat":  accent_build(sr, g, v),
            "crash": dsp.crash(sr, g),
        }

    def _kick(self, ctx) -> dsp.Sample:
        midi = ctx.harmony.near(0, 31)  # tuned to the key: ~44-62 Hz
        if midi not in self.kicks:
            if self.v.get("kit") == "tavern":  # frame drum "doum", an octave up
                self.kicks[midi] = dsp.frame_drum(ctx.sr, ctx.nprng, dsp.midi_to_hz(midi + 12), 0.25, 0.3)
            else:
                punch, decay, drive = self.v["kick"]
                self.kicks[midi] = dsp.kick(ctx.sr, dsp.midi_to_hz(midi), ctx.nprng, punch, decay, drive)
        return self.kicks[midi]

    def set_hat(self, kind: str) -> None:
        """Hot-swap the hat sound on the next bar. kind: 'hat', 'tambourine', 'rim'."""
        if kind not in self._HAT_BUILDERS:
            raise ValueError(f"hat_kind must be one of {list(self._HAT_BUILDERS)}, got {kind!r}")
        self.v["hat_kind"] = kind
        self.kit = None  # triggers rebuild on next bar

    def set_accent(self, kind: str) -> None:
        """Hot-swap the offbeat accent on the next bar. kind: 'ohat', 'cowbell', 'rim', 'tambourine'."""
        if kind not in self._ACCENT_BUILDERS:
            raise ValueError(f"accent_kind must be one of {list(self._ACCENT_BUILDERS)}, got {kind!r}")
        self.v["accent_kind"] = kind
        self.kit = None

    def on_bar(self, ctx, bar, offset):
        if self.kit is None:
            self._build(ctx)
        r, lv = ctx.rng, self.level
        B = ctx.bar_steps
        g = self._groove(ctx)
        starts = set(ctx.beats)
        offbeats = [i + n // 2 for i, n in zip(ctx.beats, ctx.beat_lens)]
        z = [0.0] * B
        kick, snare, clap, hat, ohat = list(z), list(z), list(z), list(z), list(z)
        if lv < 0.4:  # heartbeat pulse
            kick[0] = 1.0
            if r.random() < 0.5:
                kick[g["extra"]] = 0.6
            for i in offbeats:
                hat[i] = 0.5
        elif lv < 0.75:  # boots and cats
            for i in g["boots"]:
                kick[i] = 1.0
            if r.random() < 0.4:
                kick[r.choice(g["syncop"])] = 0.7
            for i in g["back"]:
                snare[i] = 0.9
            for i in range(0, B, 2):
                hat[i] = 0.4 if i in starts else 0.65
        else:  # driving
            for i in g["beats"]:
                kick[i] = 1.0
            for i in g["back"]:
                clap[i] = 1.0
                snare[i] = 0.5
            for i in range(B):
                hat[i] = (0.6 if i % 2 else 0.35) * r.uniform(0.8, 1.0)
            for i in offbeats:
                ohat[i] = 0.55
                hat[i] = 0.0
            for i in r.sample(g["ghosts"], 2):
                snare[i] = max(snare[i], 0.22)  # ghost notes
        if bar % 8 == 7:  # fill into the next phrase
            fill = r.choice(g["fills"])
            for k, i in enumerate(fill):
                snare[i] = 0.5 + 0.5 * k / len(fill)
            for i in fill:
                ohat[i] = 0.0
        self.pat = {"kick": kick, "snare": snare, "clap": clap, "hat": hat, "ohat": ohat}
        if bar % 8 == 0 and ctx.flags.pop("riser", False):
            self.play_sample(self.kit["crash"], 0.45 * lv, offset, pan=r.uniform(-0.3, 0.3))

    @classmethod
    def _groove(cls, ctx) -> dict:
        """Hand-made grooves for even meters; others are built from the beat groups:
        kicks on every other beat, backbeats between, fills over the last beat."""
        key = (ctx.bar_steps, ctx.beat_steps)
        if len(set(ctx.beat_lens)) == 1 and key in cls.GROOVES:
            return cls.GROOVES[key]
        cache = cls.__dict__.get("_made")
        if cache is None:
            cache = cls._made = {}
        meter = (ctx.beats, ctx.beat_lens)
        if meter not in cache:
            B, starts = ctx.bar_steps, list(ctx.beats)
            offs = [i + n // 2 for i, n in zip(starts, ctx.beat_lens)]
            boots, back = tuple(starts[::2]), tuple(starts[1::2]) or (offs[0],)
            cache[meter] = dict(
                extra=offs[len(offs) // 2], boots=boots, back=back, beats=tuple(starts),
                syncop=sorted({o for o in offs[1:] if o not in back} | {B - 1}),
                ghosts=[i for i in range(1, B, 2) if i not in boots],
                fills=[list(range(B - 4, B)), [i for i in (B - 6, B - 4, B - 3, B - 2, B - 1) if i >= 0], [B - 2, B - 1]],
            )
        return cache[meter]

    def on_step(self, ctx, step, offset):
        if not self.active or not self.pat:
            return
        if self.kit is None:  # variant changed mid-bar
            self._build(ctx)
        if len(self.pat["kick"]) != ctx.bar_steps:  # meter changed; new pattern next bar
            return
        i, r, lv, v = ctx.pos, ctx.rng, self.level, self.v
        amp = 0.35 + 0.65 * lv
        if i % 2 and v["swing"]:
            offset += int(v["swing"] * ctx.step_samples)
        if self.pat["kick"][i]:
            self.play_sample(self._kick(ctx), 0.9 * amp * self.pat["kick"][i], offset)
            if v.get("sidechain", True):
                ctx.kick(offset)
        for name, pan, a in (("snare", 0.05, 0.32), ("clap", -0.05, 0.32), ("hat", 0.25, 0.14), ("ohat", -0.25, 0.12)):
            vel = self.pat[name][i]
            if vel:
                self.play_sample(self.kit[name], a * v["mix"][name] * amp * vel * r.uniform(0.85, 1.0), offset, pan=pan)


class Bass(Layer):
    """Bass on the chord root: offbeat pulse, then rolling 16ths as drive climbs."""

    name = "bass"
    param = "drive"
    threshold = 0.25
    wander_depth = 0.2
    gain = 0.9
    reverb = 0.05
    duck = 1.0
    saturate = 3.0

    VARIANTS = {
        "saw": dict(
            pattern="pulse", gate=0.85, env=(0.003, 0.12, 0.5, 0.06),
            timbre=lambda b, lv: dsp.Wave(dsp.harmonic_amps(30, 0.4 + 0.5 * b), unison=((-8, 1.0), (8, 1.0)),
                                          dark=dsp.harmonic_amps(4, 0.2), morph=0.05 + 0.08 * (1 - lv)),
        ),
        "sub": dict(  # round and clean, longer notes
            pattern="pulse", gate=1.6, env=(0.006, 0.3, 0.85, 0.08), saturate=0.8, gain=0.43,
            timbre=lambda b, lv: dsp.Wave([1.0, 0.3 + 0.2 * b, 0.08]),
        ),
        "acid": dict(  # squelchy resonant peak, wanders off the root
            pattern="pulse", gate=0.7, env=(0.003, 0.15, 0.4, 0.05), saturate=4.0, wander=True, gain=1.51,
            timbre=lambda b, lv: dsp.Wave(dsp.formant_amps(30, ((6 + 6 * b, 1.5),), tilt=0.7),
                                          dark=dsp.harmonic_amps(5, 0.2), morph=0.06 + 0.05 * lv),
        ),
        "folk": dict(  # plucked bass strings: root on the downbeat, fifth on the next beat
            gain=1.7, pattern="oompah", saturate=0.0, duck=0.0, threshold=0.4, reverb=0.15,
            ks=dict(decay=1.4, bright=0.3), timbre=None,
        ),
        "reese": dict(  # wide detuned saws, long notes that beat against each other
            pattern="long", gate=0.95, env=(0.02, 0.4, 0.8, 0.15), saturate=2.0, gain=0.92,
            timbre=lambda b, lv: dsp.Wave(dsp.harmonic_amps(24, 0.3 + 0.4 * b),
                                          unison=((-24, 1.0), (-8, 0.8), (8, 0.8), (24, 1.0))),
        ),
        "italo": dict(  # punchy gated 16ths: short gate, snappy attack, resonant peak tracks brightness
            pattern="pulse", gate=0.45, env=(0.003, 0.08, 0.0, 0.04), saturate=3.5, gain=1.05,
            timbre=lambda b, lv: dsp.Wave(
                dsp.formant_amps(30, ((4.0 + 8.0 * b, 1.2),), tilt=0.65),
                unison=((-8, 1.0), (8, 1.0)),
                dark=dsp.harmonic_amps(5, 0.15),
                morph=0.04,
            ),
        ),
    }

    def on_step(self, ctx, step, offset):
        if not self.active:
            return
        v, B = self.v, ctx.bar_steps
        if v["pattern"] == "oompah":
            self._oompah(ctx, offset)
            return
        if v["pattern"] == "long":
            if ctx.pos % (B // 2 if self.level > 0.6 else B):
                return
            length = B // 2 if self.level > 0.6 else B
        else:
            length = 1
            _, i, b = ctx.beat_at(ctx.pos)
            if self.level < 0.6:
                if i != b // 2:  # offbeat 8ths
                    return
            elif i == 0:  # rolling: everything except where the kick lands
                return
        h, r = ctx.harmony, ctx.rng
        root = h.near(h.degree, 40)
        midi = root + (12 if r.random() < 0.15 else 7 if r.random() < 0.08 else 0)
        if v.get("wander") and r.random() < 0.3:
            midi = r.choice(h.scale_midis(root - 2, root + 12) or [root])
        accent = r.uniform(0.7, 1.15) if v.get("wander") else 1.0
        self.play(ctx, midi, 0.3 * (0.6 + 0.4 * self.level) * accent, Envelope(*v["env"]),
                  v["timbre"](ctx.p["brightness"], self.level), offset, duration=ctx.step_seconds * length * v["gate"])


    def _oompah(self, ctx, offset):
        beat, k, _ = ctx.beat_at(ctx.pos)
        if k:
            return
        beats = len(ctx.beats)
        h, r = ctx.harmony, ctx.rng
        root = h.near(h.degree, 40)
        if beats == 3:  # waltz: root on one, sometimes the fifth on three
            if beat == 1 or (beat == 2 and r.random() > 0.35):
                return
        midi = root if beat % 2 == 0 else root - 5
        if self.level > 0.6 and beat % 2 == 0 and r.random() < 0.2:
            midi += 12
        ks = self.v["ks"]
        self.play_sample(self.ks(ctx, midi, **ks), 0.4 * (0.6 + 0.4 * self.level), offset, pan=-0.1)


class Stabs(Layer):
    """Chord hits on syncopated rhythms."""

    name = "stabs"
    param = "drive"
    threshold = 0.45
    wander_depth = 0.6
    gain = 1.5
    reverb = 0.3
    echo = 0.35
    duck = 0.3
    saturate = 1.5
    max_notes = 24

    VARIANTS = {
        "supersaw": dict(
            env=(0.004, 0.16, 0.25, 0.25), length=1.5,
            timbre=lambda b: dsp.wave(28, 0.55 + 0.4 * b, unison=((-16, 0.7), (-7, 1.0), (0, 1.0), (7, 1.0), (16, 0.7))),
        ),
        "organ": dict(  # house organ: drawbar-ish harmonics, short and percussive
            env=(0.003, 0.12, 0.35, 0.08), length=1.0, saturate=2.5, gain=1.05,
            timbre=lambda b: dsp.Wave([1.0, 0.8, 0.6, 0.45, 0.0, 0.3, 0.0, 0.25 * b + 0.05], unison=((-3, 1.0), (3, 1.0))),
        ),
        "pluck": dict(  # bright attack that snaps shut
            env=(0.002, 0.25, 0.0, 0.2), length=2.0, echo=0.5, gain=0.54,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(24, 0.6 + 0.35 * b), unison=((-6, 1.0), (6, 1.0)),
                                      dark=dsp.harmonic_amps(3, 0.1), morph=0.06),
        ),
        "strum": dict(  # strummed lute/cittern chords: down on the beat, lighter up-strums between
            gain=1.19, strum=True, saturate=0.0, duck=0.0, threshold=0.55, reverb=0.25, echo=0.05,
            ks=dict(decay=1.3, bright=0.5), timbre=None,
            rhythms={"4/4": [[0, 4, 6, 8, 12, 14], [0, 6, 8, 12, 14], [0, 4, 8, 10, 12]],
                     "6/8": [[0, 4, 6, 10], [0, 2, 4, 6, 8, 10], [0, 6, 10]],
                     "3/4": [[4, 8], [4, 8], [4, 6, 8]]},
        ),
        "brass": dict(  # "bwah": opens up on each hit
            env=(0.03, 0.3, 0.55, 0.2), length=2.0, saturate=2.0, gain=0.64,
            timbre=lambda b: dsp.Wave(dsp.harmonic_amps(4, 0.2), unison=((-7, 1.0), (0, 0.8), (7, 1.0)),
                                      dark=dsp.harmonic_amps(24, 0.6 + 0.35 * b), morph=0.08),
        ),
        "italo": dict(  # tight punchy chord stab: zero sustain, hard snap, wide supersaw; pure dancefloor
            env=(0.003, 0.1, 0.0, 0.08), length=0.7, saturate=2.5, gain=1.15, echo=0.4,
            timbre=lambda b: dsp.Wave(
                dsp.harmonic_amps(30, 0.5 + 0.4 * b),
                unison=((-18, 0.6), (-9, 1.0), (-3, 0.8), (0, 1.0), (3, 0.8), (9, 1.0), (18, 0.6)),
            ),
        ),
    }

    RHYTHMS = {
        "4/4": [[0, 3, 6, 10, 12], [2, 6, 10, 14], [0, 6, 12], [3, 6, 11, 14], [0, 3, 8, 11]],
        "6/8": [[0, 3, 6, 9], [0, 4, 6, 10], [2, 6, 8], [0, 3, 8]],
        "3/4": [[0, 4, 8], [4, 8], [0, 3, 6, 9], [0, 6, 8]],
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rhythm = self.RHYTHMS["4/4"][0]
        self.rhythm_key = ("4/4", "")

    def on_bar(self, ctx, bar, offset):
        key = (ctx.meter, self.variant)
        if bar % 4 == 0 or key != self.rhythm_key:
            table = self.v.get("rhythms", self.RHYTHMS)
            self.rhythm = ctx.rng.choice(table.get(ctx.meter) or self._made_rhythms(ctx))
            self.rhythm_key = key

    def _made_rhythms(self, ctx) -> list[list[int]]:
        """Rhythms for meters without hand-made ones, built from the beat groups."""
        starts = list(ctx.beats)
        offs = [i + n // 2 for i, n in zip(starts, ctx.beat_lens)]
        if self.v.get("strum"):  # down on the beats, up-strums between
            return [sorted(starts + offs), starts, sorted(starts + offs[-1:])]
        return [starts, offs, [0] + offs[1:], sorted({0, *offs[::2], *starts[1::2]})]

    def on_step(self, ctx, step, offset):
        if not self.active or ctx.pos not in self.rhythm or ctx.rng.random() > 0.5 + 0.5 * self.level:
            return
        h, v = ctx.harmony, self.v
        if v.get("strum"):
            self._strum(ctx, offset)
            return
        tones = [h.near(d, 66 + 3 * i) for i, d in enumerate(h.chord[:3])]
        timbre = v["timbre"](ctx.p["brightness"])
        for m in sorted(set(tones)):
            self.play(ctx, m, 0.12 * self.level, Envelope(*v["env"]), timbre, offset,
                      duration=ctx.step_seconds * v["length"], pan=ctx.rng.uniform(-0.4, 0.4))


    def _strum(self, ctx, offset):
        h, r = ctx.harmony, ctx.rng
        strings = h.chord_midis(50, 74)[:6]
        if not strings:
            return
        down = ctx.beat_at(ctx.pos)[1] == 0
        if not down:  # up-strums catch only the top strings, lighter
            strings = strings[::-1][:4]
        gap = int(ctx.sr * r.uniform(0.010, 0.016) * (1.0 if down else 0.7))
        amp = 0.09 * self.level * (1.0 if down else 0.6)
        for k, m in enumerate(strings):
            self.play_sample(self.ks(ctx, m, **self.v["ks"]), amp * r.uniform(0.8, 1.0), offset + k * gap,
                             pan=-0.3 + 0.6 * k / max(1, len(strings) - 1))


class Riser(Layer):
    """A two-bar sweep into the next 8-bar phrase, landing on a crash from Drums."""

    name = "riser"
    param = "drive"
    threshold = 0.35
    wander_depth = 0.3
    gain = 0.8
    reverb = 0.5
    echo = 0.2

    VARIANTS = {
        "sweep": dict(noise=0.7, tone=0.25, octaves=1.5),
        "noise": dict(noise=1.0, tone=0.0, octaves=0.0, gain=1.21),
        "tonal": dict(noise=0.1, tone=0.5, octaves=2.0, gain=0.53),
        "reverse": dict(reverse=True, gain=1.13),  # a reversed cymbal sucking into the downbeat
        "roll": dict(gain=0.94, roll=True, threshold=0.5),  # frame-drum roll that speeds up into the downbeat
    }

    def on_bar(self, ctx, bar, offset):
        if bar % 8 != 6 or not self.active or ctx.rng.random() > 0.3 + 0.5 * self.level:
            return
        v = self.v
        secs = 2 * ctx.bar_steps * ctx.step_seconds
        if v.get("roll"):
            sample = dsp.drum_roll(ctx.sr, secs, ctx.nprng, dsp.midi_to_hz(ctx.harmony.near(0, 31) + 12))
        elif v.get("reverse"):
            sample = dsp.reverse_cymbal(ctx.sr, secs, ctx.nprng)
        else:
            base = dsp.midi_to_hz(ctx.harmony.near(0, 55))
            sample = dsp.riser(ctx.sr, secs, base, ctx.nprng, v["noise"], v["tone"], v["octaves"])
        self.play_sample(sample, 0.35 * self.level, offset)
        ctx.flags["riser"] = True


LAYER_TYPES = [Drone, Pad, Swell, Arp, Bells, Melody, Drums, Bass, Stabs, Riser]


def default_layers() -> list[Layer]:
    return [cls() for cls in LAYER_TYPES]
