"""Sound effects, synthesized from scratch: no samples, every call a slightly different take.

    from drift import sfx
    x = sfx.render("hit_bone")                 # mono float32, at the recipe's calibrated level
    x = sfx.render("footstep", surface="grass")
    x = sfx.render("heal", key=(62, sfx.MAJOR))  # musical ones can follow the music's key

Recipes are plain functions (sr, rng, **params) -> mono array, registered with @recipe.
Each output is scaled so its loudest 50 ms sits at the recipe's `level` (dB RMS), which
keeps effects balanced against each other and against the music (the Engine aims for
about -18 dB RMS). SfxPlayer (pygame) adds variation pools, panning and music ducking.

**Tuning.** Every recipe takes the knobs in `POST` (gain, pitch, low and high cut), and
some declare their own with `knobs()` (the layers of a footstep, say). Values that
differ from the defaults live in `sfx_tuning.json` next to this file, which `sfx_lab.py`
edits by ear; `load_tuning()` re-reads it (drift.server does so when it changes).
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Callable

import numpy as np

from .dsp import _band_noise, midi_to_hz

MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)
SURFACES = ("stone", "dirt", "grass", "wood", "water")

RECIPES: dict[str, tuple[Callable, float]] = {}

#: recipe -> knob -> (default, low, high, what it does). Declared with knobs().
KNOBS: dict[str, dict[str, tuple]] = {}
#: Knobs every recipe has, applied to the finished sound.
POST = {
    "gain_db": (0.0, -18.0, 12.0, "louder / quieter than the calibrated level, dB"),
    "pitch": (0.0, -12.0, 12.0, "semitones (also shortens / stretches it)"),
    "low_cut": (0.0, 0.0, 2000.0, "remove rumble below, Hz (0 = off)"),
    "high_cut": (20000.0, 500.0, 20000.0, "remove fizz above, Hz (20000 = off)"),
}
#: recipe -> knob -> value, only where it differs from the default.
TUNING: dict[str, dict[str, float]] = {}
TUNING_PATH = os.environ.get("DRIFT_SFX_TUNING") or os.path.join(os.path.dirname(__file__), "sfx_tuning.json")


def knobs(recipe_name: str, **spec: tuple) -> None:
    """Declare a recipe's tunable constants: name=(default, low, high, description)."""
    KNOBS[recipe_name] = spec


class _Knobs(dict):
    __getattr__ = dict.__getitem__


def tuned(recipe_name: str) -> _Knobs:
    """The recipe's knob values: tuning where set, defaults otherwise (k.name)."""
    t = TUNING.get(recipe_name, {})
    return _Knobs({k: t.get(k, d[0]) for k, d in KNOBS.get(recipe_name, {}).items()})


def load_tuning(path: str | None = None) -> dict:
    global TUNING
    try:
        with open(path or TUNING_PATH) as f:
            TUNING = {k: dict(v) for k, v in json.load(f).items()}
    except (OSError, ValueError):
        TUNING = {}
    return TUNING


def save_tuning(path: str | None = None) -> str:
    """Write the tuning, dropping values equal to their defaults."""
    out = {}
    for name, vals in sorted(TUNING.items()):
        spec = {**KNOBS.get(name, {}), **POST}
        keep = {k: v for k, v in sorted(vals.items()) if k in spec and v != spec[k][0]}
        if keep:
            out[name] = keep
    path = path or TUNING_PATH
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    return path


def _soft_filter(x, sr, low=0.0, high=20000.0):
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr) + 1e-9
    if low > 0:
        spec *= (f / np.sqrt(f**2 + low**2)) ** 2
    if high < 20000:
        spec /= 1.0 + (f / high) ** 4
    return np.fft.irfft(spec, len(x))


def _post(x, sr, t):
    """The POST knobs, skipped entirely at their defaults."""
    if t.get("pitch", 0.0):
        r = 2 ** (t["pitch"] / 12)
        x = np.interp(np.arange(0, len(x) - 1, r), np.arange(len(x)), x)
    if t.get("low_cut", 0.0) > 0 or t.get("high_cut", 20000.0) < 20000:
        x = _soft_filter(x, sr, t.get("low_cut", 0.0), t.get("high_cut", 20000.0))
    return x


def recipe(name: str, level: float):
    """Register fn(sr, rng, **params) as a sound named `name`, loudest 50 ms at `level` dB RMS."""
    def deco(fn):
        RECIPES[name] = (fn, level)
        return fn
    return deco


def render(name: str, sr: int = 44100, rng: np.random.Generator | None = None, **params) -> np.ndarray:
    fn, level = RECIPES[name]
    rng = rng if rng is not None else np.random.default_rng()
    x = np.asarray(fn(sr, rng, **params), dtype=np.float64)
    t = TUNING.get(name, {})
    x = _post(x, sr, t)
    fade = min(len(x) // 4, int(0.004 * sr))
    if fade:
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    level += t.get("gain_db", 0.0)
    return (x * (10 ** (level / 20) / max(loudness(x, sr), 1e-9))).astype(np.float32)


def loudness(x: np.ndarray, sr: int) -> float:
    """RMS of the loudest 50 ms window."""
    w = max(1, int(0.05 * sr))
    if len(x) <= w:
        return float(np.sqrt(np.mean(x**2)))
    c = np.concatenate([[0.0], np.cumsum(x**2)])
    return float(np.sqrt((c[w:] - c[:-w]).max() / w))


# ------------------------------------------------------------------ building blocks


def _t(sr, seconds):
    return np.arange(int(sr * seconds)) / sr


def _env(sr, seconds, attack=0.002, decay=0.1):
    t = _t(sr, seconds)
    return np.minimum(1.0, t / max(attack, 1e-6)) * np.exp(-np.maximum(0.0, t - attack) / decay)


def _noise(sr, seconds, lo, hi, rng):
    x = _band_noise(int(sr * seconds), sr, lo, hi, rng)
    return x / (np.abs(x).max() + 1e-12)


def _modes(sr, seconds, freqs, decays, amps, rng, jitter=0.03):
    """Struck object: a sum of decaying sine modes (wood, metal, bone...)."""
    t = _t(sr, seconds)
    out = np.zeros_like(t)
    for f, d, a in zip(freqs, decays, amps):
        f *= 1 + rng.uniform(-jitter, jitter)
        out += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    return out


def _mix(*parts):
    """Sum arrays of different lengths (padding the shorter ones with silence)."""
    out = np.zeros(max(len(p) for p in parts))
    for p in parts:
        out[: len(p)] += p
    return out


def _place(out, x, at, gain=1.0):
    """Add x into out starting at sample `at` (clipped to out's length)."""
    at = int(at)
    if at >= len(out):
        return out
    n = min(len(x), len(out) - at)
    out[at:at + n] += gain * x[:n]
    return out


def _sweep(sr, seconds, path, rng, width=0.6, bands=10, lo=150.0, hi=9000.0):
    """Noise whose centre frequency follows path(t in 0..1) -> Hz: whooshes, wind, roars."""
    n = int(sr * seconds)
    centres = np.geomspace(lo, hi, bands)
    t = np.linspace(0, 1, n)
    track = np.log(np.maximum(path(t), 1.0))
    out = np.zeros(n)
    for c in centres:
        x = _band_noise(n, sr, c / (1 + width), c * (1 + width), rng)
        x /= np.sqrt(np.mean(x**2)) + 1e-12
        out += x * np.exp(-((track - math.log(c)) ** 2) / (2 * 0.35**2))
    return out


def _thud(sr, seconds, f0, f1, decay, rng):
    """Pitch-dropping sine thump (body hits, falls, the low half of footsteps)."""
    t = _t(sr, seconds)
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / decay)


def _creak(sr, seconds, rate, rng, body=(320, 800, 1700), damp=0.012):
    """Stick-slip creak: a jittery pulse train with a gliding rate through a wooden body.
    rate(t in 0..1) -> pulses per second."""
    n = int(sr * seconds)
    t = np.linspace(0, 1, n)
    r = rate(t) * (1 + 0.15 * rng.standard_normal(n).cumsum() / math.sqrt(n) * 3)
    phase = np.cumsum(np.maximum(r, 5.0) / sr)
    pulses = np.zeros(n)
    idx = np.nonzero(np.diff(np.floor(phase)) > 0)[0]
    pulses[idx] = rng.uniform(0.5, 1.0, len(idx))
    ir = _modes(sr, 0.06, body, [damp] * len(body), [1.0, 0.6, 0.3], rng)
    out = np.convolve(pulses, ir)[:n]
    swell = np.sin(np.pi * np.clip(t * 1.1, 0, 1)) ** 0.5
    return out * swell


def _tone(sr, seconds, midi, decay, rng, partials=((1, 1.0), (2, 0.35), (3, 0.12)), attack=0.004, vib=0.0):
    t = _t(sr, seconds)
    f = midi_to_hz(midi)
    out = np.zeros_like(t)
    for k, a in partials:
        ph = 2 * np.pi * f * k * t
        if vib:
            ph += vib * k * np.sin(2 * np.pi * 5.5 * t)
        out += a * np.sin(ph + rng.uniform(0, 6.28)) * np.exp(-t / (decay / k**0.5))
    return out * np.minimum(1.0, t / attack)


def _key_notes(key, degrees, octave=5):
    root, scale = key if key else (60, MAJOR)
    base = 12 * (octave + 1) + root % 12
    return [base + 12 * (d // len(scale)) + scale[d % len(scale)] for d in degrees]


# ------------------------------------------------------------------ combat


@recipe("swing", level=-24)
def swing(sr, rng, weight=0.5):
    """Blade/club through the air. weight 0 (dagger) .. 1 (hammer): lower, longer."""
    dur = 0.22 + 0.25 * weight + rng.uniform(-0.03, 0.03)
    peak = (2600 - 1500 * weight) * rng.uniform(0.85, 1.15)
    x = _sweep(sr, dur, lambda t: 300 + peak * np.sin(np.pi * t) ** 2, rng, width=0.5)
    t = np.linspace(0, 1, len(x))
    return x * np.sin(np.pi * t) ** 2.5


@recipe("hit_flesh", level=-16)
def hit_flesh(sr, rng, power=0.5):
    body = _thud(sr, 0.25, 160 + 60 * power, 70, 0.07, rng)
    squish = _noise(sr, 0.12, 300, 1800, rng) * _env(sr, 0.12, 0.001, 0.025)
    slap = _noise(sr, 0.03, 1500, 7000, rng) * _env(sr, 0.03, 0.0005, 0.006)
    out = body * (0.8 + 0.4 * power)
    _place(out, squish, 0, 0.7)
    return _place(out, slap, 0, 0.35)


@recipe("hit_metal", level=-17)
def hit_metal(sr, rng, size=0.5):
    f = (900 - 450 * size) * rng.uniform(0.9, 1.1)
    ring = _modes(sr, 1.0, [f, f * 2.76, f * 5.4, f * 8.93], [0.35, 0.2, 0.08, 0.04], [1.0, 0.6, 0.35, 0.2], rng)
    click = _noise(sr, 0.02, 2000, 12000, rng) * _env(sr, 0.02, 0.0003, 0.004)
    return _place(ring + _thud(sr, 1.0, 180, 90, 0.05, rng) * 0.5, click, 0, 0.8)


@recipe("hit_wood", level=-18)
def hit_wood(sr, rng):
    f = 210 * rng.uniform(0.9, 1.1)
    x = _modes(sr, 0.35, [f, f * 2.8, f * 5.3], [0.06, 0.03, 0.015], [1.0, 0.5, 0.25], rng)
    return _place(x, _noise(sr, 0.03, 800, 6000, rng) * _env(sr, 0.03, 0.0003, 0.005), 0, 0.5)


def _clack(sr, rng, f=1800.0):
    return _modes(sr, 0.08, [f, f * 1.7, f * 2.9], [0.012, 0.008, 0.005], [1.0, 0.6, 0.3], rng, jitter=0.15)


@recipe("hit_bone", level=-17)
def hit_bone(sr, rng):
    out = np.zeros(int(sr * 0.3))
    at = 0
    for k in range(rng.integers(2, 4)):
        _place(out, _clack(sr, rng, rng.uniform(1100, 2600)), at, 1.0 / (1 + k))
        at += int(sr * rng.uniform(0.015, 0.05))
    crack = _noise(sr, 0.05, 1500, 9000, rng) * _env(sr, 0.05, 0.0005, 0.01)
    return _place(out, crack, 0, 0.6)


@recipe("bones_collapse", level=-19)
def bones_collapse(sr, rng):
    """A skeleton coming apart: a cascade of clacks thinning out, then a dull settle."""
    out = np.zeros(int(sr * 1.0))
    t = 0.0
    while t < 0.75:
        _place(out, _clack(sr, rng, rng.uniform(700, 2400)), t * sr, rng.uniform(0.3, 1.0) * (1 - t))
        t += rng.exponential(0.025 + 0.06 * t)
    _place(out, _thud(sr, 0.3, 120, 60, 0.06, rng), 0.12 * sr, 0.5)
    return out


@recipe("block", level=-16)
def block(sr, rng, metal=True):
    """A blow caught on a shield: a wooden thump, plus a clank if it is metal-rimmed."""
    out = _thud(sr, 0.5, 220, 110, 0.06, rng) + _modes(sr, 0.5, [180, 470, 900], [0.05, 0.03, 0.02], [0.6, 0.4, 0.2], rng)
    if metal:
        f = rng.uniform(500, 650)
        out += 0.5 * _modes(sr, 0.5, [f, f * 2.4, f * 4.1], [0.18, 0.1, 0.06], [1.0, 0.5, 0.3], rng)
    return out


@recipe("hurt", level=-17)
def hurt(sr, rng):
    """The player getting hit: a heavier, duller body blow."""
    out = _thud(sr, 0.35, 120, 55, 0.1, rng)
    return _place(out, _noise(sr, 0.15, 200, 1200, rng) * _env(sr, 0.15, 0.002, 0.04), 0, 0.6)


@recipe("body_fall", level=-18)
def body_fall(sr, rng):
    out = np.zeros(int(sr * 0.7))
    _place(out, _thud(sr, 0.4, 90, 45, 0.12, rng) + _noise(sr, 0.4, 150, 900, rng) * _env(sr, 0.4, 0.003, 0.06) * 0.6, 0)
    _place(out, _thud(sr, 0.3, 110, 60, 0.06, rng) * 0.5, rng.uniform(0.12, 0.2) * sr)  # a limb lands after
    return out


# ------------------------------------------------------------------ magic


def _crackle(sr, seconds, rng, density=60.0):
    n = int(sr * seconds)
    out = np.zeros(n)
    k = rng.poisson(density * seconds)
    for at in rng.integers(0, n, k):
        _place(out, _noise(sr, 0.006, 1500, 9000, rng) * np.hanning(int(sr * 0.006)), at, rng.uniform(0.2, 1.0))
    return out


@recipe("cast", level=-20)
def cast(sr, rng, element="fire"):
    """Release of a spell. element: fire, ice, dark."""
    t = np.linspace(0, 1, int(sr * 0.5))
    if element == "ice":
        notes = rng.uniform(2000, 4500, 5)
        out = sum(_modes(sr, 0.5, [f], [0.15], [1.0], rng) for f in notes) * 0.4
        out += _sweep(sr, 0.5, lambda u: 1500 + 6000 * u, rng) * np.sin(np.pi * t) * 0.3
        return out
    if element == "dark":
        t2 = _t(sr, 0.6)
        f = 110 * np.exp(-t2 * 1.5)
        drone = np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.5 * np.sin(4 * np.pi * np.cumsum(f * 1.01) / sr)
        air = _sweep(sr, 0.6, lambda u: 2500 - 2000 * u, rng)
        env = np.sin(np.pi * np.linspace(0, 1, len(t2))) ** 1.5
        return (drone * 0.6 + air * 0.25) * env
    roar = _sweep(sr, 0.5, lambda u: 400 + 1500 * np.sin(np.pi * u), rng, lo=80) * np.sin(np.pi * t) ** 1.5
    return roar + _crackle(sr, 0.5, rng) * 0.5


@recipe("impact", level=-15)
def impact(sr, rng, element="fire"):
    """A bolt landing."""
    if element == "ice":
        out = np.zeros(int(sr * 0.6))
        for _ in range(14):
            _place(out, _modes(sr, 0.2, [rng.uniform(2500, 7000)], [0.05], [1.0], rng), rng.uniform(0, 0.12) * sr, rng.uniform(0.3, 1))
        return _place(out, _noise(sr, 0.08, 1000, 10000, rng) * _env(sr, 0.08, 0.0005, 0.015), 0, 1.5)
    if element == "dark":
        x = _thud(sr, 0.6, 140, 40, 0.15, rng)
        return x + _sweep(sr, 0.6, lambda u: 1200 * np.exp(-4 * u) + 100, rng, lo=60) * _env(sr, 0.6, 0.01, 0.12) * 0.5
    boom = _thud(sr, 0.8, 120, 45, 0.15, rng)
    body = _noise(sr, 0.8, 100, 3000, rng) * _env(sr, 0.8, 0.002, 0.12)
    return boom + body * 0.8 + _crackle(sr, 0.8, rng, 90) * _env(sr, 0.8, 0.01, 0.25) * 0.6


@recipe("heal", level=-21)
def heal(sr, rng, key=None):
    """Rising chime arpeggio in the music's key, over a soft shimmer."""
    out = np.zeros(int(sr * 1.6))
    for k, m in enumerate(_key_notes(key, [0, 2, 4, 7], octave=5)):
        _place(out, _tone(sr, 1.2, m, 0.6, rng, ((1, 1.0), (2, 0.2), (4, 0.08))), k * 0.09 * sr, 0.8)
    sh = _sweep(sr, 1.4, lambda u: 3000 + 4000 * u, rng, width=0.3) * _env(sr, 1.4, 0.3, 0.5)
    return _place(out, sh, 0, 0.08)


@recipe("mana", level=-21)
def mana(sr, rng, key=None):
    out = np.zeros(int(sr * 1.4))
    for k, m in enumerate(_key_notes(key, [7, 4, 2, 4], octave=5)):
        _place(out, _tone(sr, 1.0, m, 0.5, rng, ((1, 1.0), (3, 0.25), (5, 0.1)), vib=0.15), k * 0.08 * sr, 0.8)
    return out


@recipe("level_up", level=-18)
def level_up(sr, rng, key=None):
    """A little fanfare in the music's key."""
    out = np.zeros(int(sr * 2.6))
    notes = _key_notes(key, [0, 2, 4, 7], octave=4) + _key_notes(key, [0, 4], octave=5)
    times = [0, 0.1, 0.2, 0.3, 0.45, 0.45]
    for m, at in zip(notes, times):
        _place(out, _tone(sr, 2.0, m, 0.9, rng, ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15))), at * sr, 0.7)
    return out


@recipe("coin", level=-22)
def coin(sr, rng, key=None):
    out = np.zeros(int(sr * 0.5))
    a, b = _key_notes(key, [4, 7], octave=7)
    for m, at in ((a, 0.0), (b, 0.07)):
        f = midi_to_hz(m)
        _place(out, _modes(sr, 0.4, [f, f * 2.4, f * 3.9], [0.12, 0.06, 0.04], [1.0, 0.3, 0.15], rng, 0.0), at * sr)
    return out


@recipe("pickup", level=-23)
def pickup(sr, rng, key=None):
    (m,) = _key_notes(key, [int(rng.integers(0, 5))], octave=5)
    return _tone(sr, 0.3, m, 0.08, rng, ((1, 1.0), (2, 0.4), (3, 0.2))) + _place(
        np.zeros(int(sr * 0.3)), _noise(sr, 0.04, 300, 3000, rng) * _env(sr, 0.04, 0.002, 0.01), 0, 0.3)


# ------------------------------------------------------------------ movement


#: Per surface: the crunch under the foot as (grains, low Hz, high Hz, decay s, gain),
#: and how much the ground gives (scales the heel thud's length). Grass is the base
#: every other step is built from.
GROUND = {
    "grass": ((40, 1500, 7000, 0.05, 0.6), 1.0),
    "dirt": ((25, 500, 4000, 0.05, 0.6), 1.0),
    "stone": ((14, 2500, 10000, 0.02, 0.35), 0.6),
    "wood": ((12, 1200, 6000, 0.03, 0.3), 0.8),
}
FOOTWEAR = ("leather", "bare", "plate")


def _grains(sr, rng, count, lo, hi, decay, seconds=0.14):
    """A crunch made of many tiny noise grains, front-loaded and dying away."""
    n = int(sr * seconds)
    out = np.zeros(n)
    for at in np.sort(rng.integers(0, int(n * 0.6), count)):
        _place(out, _noise(sr, 0.004, lo, hi, rng) * np.hanning(int(sr * 0.004)), at, rng.uniform(0.2, 1.0))
    return out * np.exp(-np.arange(n) / (decay * sr))


def _burst(sr, rng, lo, hi, attack, decay, seconds=0.08):
    """A short filtered noise hit: knocks, slaps, clicks."""
    return _noise(sr, seconds, lo, hi, rng) * _env(sr, seconds, attack, decay)


def _metal_ticks(sr, rng, count, spread, gain, seconds=0.2, pitch=1.0, ring=1.0):
    """Loose metal touching metal: a few tiny, quickly damped inharmonic pings."""
    out = np.zeros(int(sr * seconds))
    for _ in range(count):
        f = rng.uniform(2200, 6500) * pitch
        ping = _modes(sr, 0.05, [f, f * rng.uniform(1.4, 1.7)], [0.012 * ring, 0.007 * ring], [1.0, 0.5], rng)
        _place(out, ping + _burst(sr, rng, 3000, 12000, 0.0002, 0.002, 0.05) * 0.5,
               rng.uniform(0, spread) * sr, rng.uniform(0.3, 1.0) * gain)
    return out


OFF = 20000  # a tone (lowpass) knob at this value does nothing

knobs(
    "footstep",
    heel=(0.6, 0.0, 1.5, "heel thud level"),
    crunch=(1.0, 0.0, 3.0, "ground grain level (x)"),
    knock=(0.45, 0.0, 1.5, "stone: the sole's knock, level"),
    knock_lo=(250, 50, 2000, "stone: knock band low edge, Hz"),
    knock_hi=(1800, 300, 8000, "stone: knock band high edge, Hz"),
    knock_decay=(0.007, 0.001, 0.05, "stone: knock length, s"),
    room=(0.06, 0.0, 0.5, "stone: the room answering, level"),
    boards=(0.5, 0.0, 1.5, "wood: boards level"),
    slap=(0.3, 0.0, 1.0, "bare: skin slap level"),
    plate_heel=(1.3, 0.5, 3.0, "plate: heel weight (x)"),
    plate_clank=(0.35, 0.0, 1.5, "plate: ringing clank level"),
    plate_pitch=(1.0, 0.25, 2.0, "plate: clank pitch (x)"),
    plate_ring=(1.0, 0.1, 4.0, "plate: clank ring length (x)"),
    plate_click=(0.6, 0.0, 2.0, "plate: contact click level"),
    plate_tone=(OFF, 500, OFF, "plate: lowpass on clank + click, Hz"),
    plate_dull=(0.0, 0.0, 1.5, "plate: dull unpitched metal thunk, level"),
    plate_dull_lo=(300, 50, 3000, "plate: thunk band low edge, Hz"),
    plate_dull_hi=(2500, 500, 10000, "plate: thunk band high edge, Hz"),
    plate_dull_decay=(0.025, 0.003, 0.1, "plate: thunk length, s"),
    rattle=(0.12, 0.0, 0.6, "armour rattle level (scaled by armour)"),
    rattle_pitch=(1.0, 0.25, 2.0, "armour rattle pitch (x)"),
    rattle_ring=(1.0, 0.1, 4.0, "armour rattle ring length (x)"),
    rattle_spread=(0.09, 0.01, 0.3, "armour rattle spread in time, s"),
    rattle_tone=(OFF, 500, OFF, "armour rattle lowpass, Hz"),
)


@recipe("footstep", level=-27)
def footstep(sr, rng, surface="grass", weight=0.5, footwear="leather", armor=0.0):
    """One step, as layers on the grass step: the ground under it (stone, dirt, grass,
    wood, water), what is on the foot (leather, bare, plate) and armour rattling (0..1)."""
    k = tuned("footstep")
    heel = _thud(sr, 0.15, 110 + 40 * weight, 60, 0.025 + 0.015 * weight, rng) * (0.5 + 0.5 * weight)
    if surface == "water":
        splash = _noise(sr, 0.2, 400, 6000, rng) * _env(sr, 0.2, 0.01, 0.05)
        bub = sum(_modes(sr, 0.2, [rng.uniform(500, 1500)], [0.02], [1.0], rng) for _ in range(3)) * 0.2
        return _mix(heel * 0.3, splash, _place(np.zeros(int(sr * 0.2)), bub, 0.03 * sr))
    (count, lo, hi, decay, gain), give = GROUND.get(surface, GROUND["grass"])
    bare, plate = footwear == "bare", footwear == "plate"
    if give != 1.0:  # hard ground: the heel stops dead instead of sinking in
        heel = _thud(sr, 0.15, 120 + 40 * weight, 70, (0.025 + 0.015 * weight) * give, rng) * (0.5 + 0.5 * weight)
    if bare:
        heel = _thud(sr, 0.15, 90 + 30 * weight, 55, 0.02, rng) * (0.35 + 0.35 * weight)
    elif plate:
        heel = heel * k.plate_heel
    grains = _grains(sr, rng, count, lo, hi, decay)
    out = _mix(heel * k.heel, grains * (gain * (0.6 if bare else 1.0) * k.crunch))
    if surface == "stone":
        if not bare:  # the sole meeting rock: a dull, unpitched knock, not a tap
            _place(out, _burst(sr, rng, k.knock_lo, k.knock_hi, 0.0004, k.knock_decay), 0, k.knock + 0.2 * weight)
        tail = _burst(sr, rng, 300, 3500, 0.004, 0.035, 0.12)  # the room answering
        _place(out, tail, 0.012 * sr, k.room)
    elif surface == "wood":
        boards = _burst(sr, rng, 120, 700, 0.002, 0.03, 0.12)
        _place(out, _mix(boards, _modes(sr, 0.12, [rng.uniform(150, 220)], [0.03], [0.3], rng)), 0, k.boards)
    if bare:  # skin on the ground: a soft slap, heel then ball of the foot
        _place(out, _burst(sr, rng, 400, 3000, 0.0005, 0.006), 0, k.slap)
        _place(out, _burst(sr, rng, 500, 3500, 0.001, 0.008), rng.uniform(0.05, 0.08) * sr, k.slap / 2)
    elif plate:  # a steel shoe: a short clank, then the plates settling
        p, r = k.plate_pitch, k.plate_ring
        clank = _modes(sr, 0.12 * max(1.0, r),
                       [rng.uniform(1500, 2100) * p, rng.uniform(3200, 4200) * p, rng.uniform(5200, 6400) * p],
                       [0.02 * r, 0.012 * r, 0.008 * r], [0.6, 0.4, 0.25], rng)
        metal = _mix(clank * (k.plate_clank / 0.35), _burst(sr, rng, 2000, 10000, 0.0002, 0.003) * k.plate_click)
        if k.plate_tone < OFF:
            metal = _soft_filter(metal, sr, high=k.plate_tone)
        _place(out, metal, 0, 0.35)
        if k.plate_dull > 0:
            _place(out, _burst(sr, rng, k.plate_dull_lo, k.plate_dull_hi, 0.0004, k.plate_dull_decay, 0.15), 0, k.plate_dull)
        armor = max(armor, 0.6)
    if armor > 0:
        ticks = _metal_ticks(sr, rng, 2 + int(4 * armor), k.rattle_spread, k.rattle * armor,
                             pitch=k.rattle_pitch, ring=k.rattle_ring)
        if k.rattle_tone < OFF:
            ticks = _soft_filter(ticks, sr, high=k.rattle_tone)
        _place(out, ticks, 0.01 * sr)
    return out


knobs(
    "hoof",
    thump=(1.0, 0.0, 3.0, "the weight landing, level (x)"),
    thump_pitch=(1.0, 0.5, 2.0, "the weight landing, pitch (x)"),
    turf=(1.3, 0.0, 4.0, "ground crunch level"),
    click=(0.7, 0.0, 2.0, "hard ground: iron shoe click level"),
    ring=(0.12, 0.0, 1.0, "hard ground: shoe ring level"),
    click_tone=(OFF, 500, OFF, "hard ground: lowpass on click + ring, Hz"),
    room=(0.08, 0.0, 0.5, "hard ground: room answering, level"),
    gap_min=(0.015, 0.0, 0.08, "second hoof of the pair: earliest, s"),
    gap_max=(0.035, 0.0, 0.12, "second hoof of the pair: latest, s"),
    second_min=(0.55, 0.0, 1.2, "second hoof of the pair: quietest"),
    second_max=(0.8, 0.0, 1.2, "second hoof of the pair: loudest"),
)


@recipe("hoof", level=-24)
def hoof(sr, rng, surface="grass", pair=True):
    """One beat of a trot: a pair of hooves landing almost together (diagonal legs).
    The weight of a horse in the ground, plus an iron shoe's click on hard surfaces."""
    k = tuned("hoof")
    (count, lo, hi, decay, gain), give = GROUND.get(surface, GROUND["grass"])
    hard = surface in ("stone", "wood")

    def strike():
        body = _thud(sr, 0.2, 95 * k.thump_pitch, 42 * k.thump_pitch, 0.035 * give + 0.01, rng)
        turf = _grains(sr, rng, int(count * 1.5), lo * 0.7, hi * 0.8, decay * 1.3, 0.18)
        out = _mix(body * ((0.6 if hard else 1.0) * k.thump), turf * gain * k.turf)
        if hard:  # iron on rock: a click with only a breath of ring
            click = _burst(sr, rng, 1800, 11000, 0.0002, 0.003)
            ring = _modes(sr, 0.06, [rng.uniform(2600, 3600), rng.uniform(4800, 6200)], [0.01, 0.006], [1, 0.6], rng)
            shoe = _mix(click, ring * k.ring)
            if k.click_tone < OFF:
                shoe = _soft_filter(shoe, sr, high=k.click_tone)
            _place(out, shoe, 0, k.click)
            _place(out, _burst(sr, rng, 300, 3500, 0.004, 0.04, 0.12), 0.012 * sr, k.room)
        return out

    out = _place(np.zeros(int(sr * max(0.26, k.gap_max + 0.2))), strike(), 0)
    if pair:
        _place(out, strike(), rng.uniform(k.gap_min, k.gap_max) * sr, rng.uniform(k.second_min, k.second_max))
    return out


@recipe("snort", level=-21)
def snort(sr, rng):
    """Horse snort: a lip-flutter burst of breath through the nose."""
    dur = rng.uniform(0.5, 0.8)
    t = _t(sr, dur)
    breath = _noise(sr, dur, 250, 2500, rng)
    flutter = (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(22, 32) * t)) ** 2
    env = np.minimum(1.0, t / 0.03) * np.exp(-t / (dur * 0.35))
    return breath * (0.35 + 0.65 * flutter) * env


@recipe("whinny", level=-20)
def whinny(sr, rng):
    """Neigh: a high nasal voice that wobbles and falls."""
    dur = rng.uniform(1.0, 1.4)
    t = _t(sr, dur)
    u = t / dur
    f0 = (700 * (1 - u) ** 0.7 + 280) * (1 + 0.06 * np.sin(2 * np.pi * 11 * t)) * rng.uniform(0.9, 1.1)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * ph) / k ** 1.2 for k in range(1, 12))
    nasal = np.fft.irfft(np.fft.rfft(voice) * _formant_curve(len(voice), sr, ((900, 300), (2200, 500))), len(voice))
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.35)
    return nasal * env + _noise(sr, dur, 500, 4000, rng) * env * 0.1


def _formant_curve(n, sr, peaks):
    f = np.fft.rfftfreq(n, 1 / sr)
    return 0.1 + sum(np.exp(-((f - c) / w) ** 2) for c, w in peaks)


@recipe("jump", level=-27)
def jump(sr, rng):
    return _sweep(sr, 0.18, lambda u: 600 + 1500 * u, rng) * np.sin(np.pi * np.linspace(0, 1, int(sr * 0.18)))


@recipe("land", level=-22)
def land(sr, rng, surface="grass", footwear="leather", armor=0.0):
    """Both feet back on the ground: a full-weight step and the body's thump."""
    return _mix(footstep(sr, rng, surface=surface, weight=1.0, footwear=footwear, armor=armor),
                _thud(sr, 0.15, 90, 50, 0.05, rng) * 0.5)


@recipe("roll", level=-25)
def roll(sr, rng):
    """Cloth and gear tumbling."""
    dur = 0.45
    t = np.linspace(0, 1, int(sr * dur))
    bumps = 0.4 + 0.6 * np.abs(np.sin(np.pi * 3 * t))
    return _noise(sr, dur, 200, 2500, rng) * bumps * np.sin(np.pi * t)


# ------------------------------------------------------------------ the world


@recipe("door_open", level=-21)
def door_open(sr, rng):
    out = _creak(sr, 1.1, lambda t: 40 + 50 * t + 20 * np.sin(6 * t), rng, body=(260, 640, 1400))
    latch = _modes(sr, 0.1, [rng.uniform(1500, 2200), 3500], [0.01, 0.006], [1.0, 0.5], rng)
    return _place(out, latch, 0, 3.0)


@recipe("door_close", level=-18)
def door_close(sr, rng):
    out = _thud(sr, 0.5, 130, 70, 0.08, rng) + _modes(sr, 0.5, [180, 410, 890], [0.06, 0.04, 0.02], [0.8, 0.5, 0.3], rng)
    latch = _modes(sr, 0.1, [rng.uniform(1600, 2200)], [0.01], [1.0], rng)
    return _place(out, latch, 0.05 * sr, 0.6)


@recipe("chest_open", level=-21)
def chest_open(sr, rng):
    out = np.zeros(int(sr * 1.0))
    _place(out, _modes(sr, 0.1, [rng.uniform(1800, 2400), 4000], [0.008, 0.005], [1.0, 0.4], rng), 0)  # latch
    _place(out, _creak(sr, 0.6, lambda t: 70 - 30 * t, rng, body=(380, 900, 2000)), 0.08 * sr, 0.8)  # hinge
    _place(out, hit_wood(sr, rng) * 0.6, 0.62 * sr)  # lid falls back
    return out


@recipe("chest_close", level=-19)
def chest_close(sr, rng):
    out = hit_wood(sr, rng) + _thud(sr, 0.35, 140, 70, 0.05, rng) * 0.5
    return out


@recipe("ui_click", level=-28)
def ui_click(sr, rng):
    return _modes(sr, 0.05, [rng.uniform(1800, 2200)], [0.006], [1.0], rng)


@recipe("ui_open", level=-27)
def ui_open(sr, rng):
    """Bag or book opening: a quick leathery rustle."""
    t = np.linspace(0, 1, int(sr * 0.25))
    return _noise(sr, 0.25, 600, 5000, rng) * np.sin(np.pi * t) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))


# ------------------------------------------------------------------ pygame player


class SfxPlayer:
    """Plays recipes through pygame.mixer with variation pools, panning and music ducking.

        sfx = SfxPlayer(music=player)                 # music: a PygamePlayer, or None
        sfx.play("hit_bone", pan=-0.3, volume=0.8)
        sfx.play("footstep", surface="grass")
        sfx.update()                                  # once a frame: releases the music duck

    Each distinct (name, params) keeps up to `variants` renders and never plays the same
    one twice in a row. Musical recipes (heal, mana, level_up, coin, pickup) follow the
    music's current key when a music engine is attached.
    """

    MUSICAL = {"heal", "mana", "level_up", "coin", "pickup"}
    LOUD = {"hit_flesh", "hit_metal", "hit_bone", "block", "hurt", "impact", "level_up", "door_close"}

    def __init__(self, music=None, variants: int = 5, volume: float = 1.0, duck: float = 0.55,
                 max_voices: int = 3, min_gap: float = 0.03):
        import pygame

        self.pg = pygame
        self.sr, size, self.chans = pygame.mixer.get_init()[:3]
        self.music = music
        self.variants = variants
        self.volume = volume
        self.duck_depth = duck
        self.max_voices = max_voices
        self.min_gap = min_gap
        self._pool: dict[tuple, list] = {}
        self._last: dict[tuple, int] = {}
        self._voices: dict[str, list] = {}
        self._last_time: dict[str, float] = {}
        self._duck = 0.0
        self._duck_t = 0.0
        self.rng = np.random.default_rng()
        # Renders happen on a worker thread so a first play never stalls a frame;
        # only a sound that has never been heard at all renders on the spot.
        import queue
        import threading

        self._jobs: queue.Queue = queue.Queue()
        self._queued: set = set()
        self._latest: dict[str, tuple] = {}  # name -> last pool used (any key / params)
        self._lock = threading.Lock()
        threading.Thread(target=self._work, name="drift-sfx", daemon=True).start()
        if pygame.mixer.get_num_channels() < 24:
            pygame.mixer.set_num_channels(24)

    def key(self):
        eng = getattr(self.music, "engine", self.music)
        h = getattr(eng, "harmony", None)
        if h is None:
            return None
        return (60 + h.root, tuple(h.scale))

    def _make(self, name, params, rng):
        x = render(name, self.sr, rng, **params)
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        pcm = np.repeat(pcm[:, None], self.chans, axis=1) if self.chans > 1 else pcm
        return self.pg.mixer.Sound(buffer=np.ascontiguousarray(pcm).tobytes())

    def _work(self):
        rng = np.random.default_rng()
        while True:
            pid, name, params = self._jobs.get()
            try:
                snd = self._make(name, params, rng)
            except Exception:
                snd = None
            with self._lock:
                self._queued.discard(pid)
                pool = self._pool.setdefault(pid, [])
                if snd is not None and len(pool) < self.variants:
                    pool.append(snd)
                    if len(pool) < self.variants:
                        self._enqueue(pid, name, params)

    def _enqueue(self, pid, name, params):
        if pid not in self._queued:
            self._queued.add(pid)
            self._jobs.put((pid, name, params))

    def _sound(self, name, params):
        if name in self.MUSICAL:
            params = {**params, "key": self.key()}
        pid = (name, tuple(sorted(params.items())))
        with self._lock:
            pool = self._pool.setdefault(pid, [])
            if len(pool) < self.variants:
                self._enqueue(pid, name, params)
            if not pool and name in self.MUSICAL and name in self._latest:
                pid = self._latest[name]  # the key moved: play the old take while the new one renders
                pool = self._pool[pid]
            pool = list(pool)
        if not pool:
            snd = self._make(name, params, self.rng)  # never heard before: no choice but now
            with self._lock:
                self._pool[pid].append(snd)
            pool = [snd]
        self._latest[name] = pid
        i = int(self.rng.integers(len(pool)))
        if len(pool) > 1 and i == self._last.get(pid, -1):
            i = (i + 1) % len(pool)
        self._last[pid] = i
        return pool[i]

    def warm(self, *names: str, **params) -> None:
        """Queue full variation pools for these sounds in the background (non-blocking)."""
        for name in names:
            p = {**params, "key": self.key()} if name in self.MUSICAL else dict(params)
            pid = (name, tuple(sorted(p.items())))
            with self._lock:
                self._enqueue(pid, name, p)

    def play(self, name: str, *, pan: float = 0.0, volume: float = 1.0, **params):
        """Play a recipe. pan -1..1, volume 0..1 (e.g. fall off with distance)."""
        import time

        now = time.monotonic()
        if volume <= 0.01 or now - self._last_time.get(name, -1.0) < self.min_gap:
            return None
        live = [c for c in self._voices.get(name, []) if c.get_busy()]
        if len(live) >= self.max_voices:
            live[0].stop()
            live = live[1:]
        ch = self._sound(name, params).play()
        if ch is None:
            return None
        v = max(0.0, min(1.0, volume * self.volume))
        a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
        ch.set_volume(v * math.cos(a) * 1.414, v * math.sin(a) * 1.414) if self.chans > 1 else ch.set_volume(v)
        self._voices[name] = live + [ch]
        self._last_time[name] = now
        if name in self.LOUD and self.music is not None:
            held = now - self._duck_t
            current = self._duck * max(0.0, 1.0 - held / 0.6)
            self._duck = max(current, self.duck_depth * min(1.0, volume * self.volume))
            self._duck_t = now
        return ch

    def clear(self) -> None:
        """Forget every rendered take (after the tuning changed); they re-render on use."""
        with self._lock:
            self._pool.clear()
            self._latest.clear()
            self._last.clear()

    def update(self) -> None:
        """Call once a frame: eases the music back up after a duck (~0.6 s)."""
        if self.music is None or not hasattr(self.music, "channel"):
            return
        import time

        held = time.monotonic() - self._duck_t
        duck = self._duck * max(0.0, 1.0 - held / 0.6)
        if duck == 0.0:
            self._duck = 0.0
        self.music.channel.set_volume(getattr(self.music, "volume", 1.0) * (1.0 - duck))


load_tuning()
