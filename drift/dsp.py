"""Low-level DSP building blocks. numpy only, block-based, no per-sample Python loops.

Everything here processes fixed-size blocks. Recursive structures (echo, reverb)
use delay lines at least one block long so a whole block can be computed at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TAU = 2.0 * math.pi


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan. pan in [-1, 1]."""
    a = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
    return math.cos(a), math.sin(a)


# --------------------------------------------------------------------------- timbre

TABLE = 2048  # wavetable length


@dataclass
class Wave:
    """Harmonic tone played from a band-limited wavetable (cheap regardless of harmonic count).

    amps[k-1] is the amplitude of harmonic k. unison is a list of (detune cents, gain)
    oscillators. If `dark` is given the tone morphs from `amps` towards `dark` with time
    constant `morph` seconds: that's how plucks lose their brightness as they ring. (It is
    just the end spectrum: make it brighter than `amps` and the tone opens up instead.)
    """

    amps: np.ndarray
    unison: tuple = ((0.0, 1.0),)
    dark: np.ndarray | None = None
    morph: float = 0.0
    # Fixed-frequency resonances [(centre Hz, width Hz, boost)], applied per note, so every
    # pitch keeps the same body/nose: what makes bowed and reed instruments sound real.
    formants: tuple = ()

    def __post_init__(self):
        self.amps = _norm(self.amps)
        if self.dark is not None:
            self.dark = _norm(self.dark)


@dataclass
class Partials:
    """Arbitrary (inharmonic) sine partials, rendered additively. decay = extra 1/s per partial."""

    ratios: np.ndarray
    amps: np.ndarray
    decay: np.ndarray

    def __post_init__(self):
        self.ratios = np.asarray(self.ratios, dtype=np.float64)
        self.amps = _norm(self.amps)
        self.decay = np.asarray(self.decay, dtype=np.float64)


def _norm(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / max(a.sum(), 1e-9)


def harmonic_amps(n: int, brightness: float, odd_only: bool = False) -> np.ndarray:
    """Saw-like (or square-like) spectrum; brightness 0..1 sets how fast harmonics roll off."""
    k = np.arange(1, (2 * n if odd_only else n) + 1, dtype=np.float64)
    rolloff = 0.15 + 2.5 * (1.0 - brightness) ** 1.5
    amps = np.exp(-(k - 1) * rolloff) / k
    if odd_only:
        amps[1::2] = 0.0
    return amps


def wave(n: int, brightness: float, odd_only: bool = False, unison=((0.0, 1.0),)) -> Wave:
    return Wave(harmonic_amps(n, brightness, odd_only), tuple(unison))


def pluck(n: int, brightness: float, decay: float) -> Wave:
    """Bright attack that mellows towards an almost-sine as it rings."""
    return Wave(harmonic_amps(n, brightness), dark=harmonic_amps(3, 0.05), morph=max(0.05, decay * 0.3))


BELL_RATIOS = np.array([0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011, 4.166])
BELL_AMPS = np.array([0.25, 1.0, 0.35, 0.45, 0.55, 0.22, 0.25, 0.15, 0.08])


def struck(ratios, amps, brightness: float, damping: float = 1.0) -> Partials:
    """Struck/inharmonic object: upper partials quieter and shorter-lived when dark."""
    ratios, amps = np.asarray(ratios, float), np.asarray(amps, float)
    amps = amps * np.exp(-(ratios - 1).clip(0) * (1.6 - brightness))
    decay = (0.15 + ratios * (0.35 + 0.6 * (1 - brightness))) * damping
    return Partials(ratios, amps, decay)


def bell(brightness: float) -> Partials:
    return struck(BELL_RATIOS, BELL_AMPS, brightness)


def formant_amps(n: int, peaks=((3.0, 1.5), (7.0, 2.0)), tilt: float = 1.0) -> np.ndarray:
    """Harmonic spectrum with resonant bumps at the given (harmonic, width) positions."""
    k = np.arange(1, n + 1, dtype=np.float64)
    amps = 1.0 / k**tilt
    for c, w in peaks:
        amps = amps * (1.0 + 3.0 * np.exp(-(((k - c) / w) ** 2)))
    return amps


def formant_gain(freqs: np.ndarray, formants) -> np.ndarray:
    g = np.ones_like(freqs, dtype=np.float64)
    for c, w, boost in formants:
        g = g + boost * np.exp(-(((freqs - c) / w) ** 2))
    return g


def _table(amps: np.ndarray, freq: float, sr: int, rng: np.random.Generator, formants=()) -> np.ndarray:
    kmax = min(len(amps), TABLE // 2 - 1, int(sr * 0.45 / freq))
    spec = np.zeros(TABLE // 2 + 1, dtype=np.complex128)
    if kmax >= 1:
        ph = rng.uniform(0, TAU, kmax)
        a = amps[:kmax]
        if formants:
            a = a * formant_gain(freq * np.arange(1, kmax + 1), formants)
            a = a / max(a.sum(), 1e-9) * amps[:kmax].sum()
        spec[1 : kmax + 1] = a * (TABLE / 2) * np.exp(1j * ph)
    t = np.fft.irfft(spec, TABLE)
    return np.append(t, t[0])  # guard sample for interpolation


# --------------------------------------------------------------------------- envelope


@dataclass
class Envelope:
    """ADSR with a smooth (raised-cosine) attack and exponential decay/release.

    Times are seconds. With sustain=0 it is a percussive attack/decay envelope.
    """

    attack: float = 0.01
    decay: float = 0.5
    sustain: float = 1.0
    release: float = 1.0

    def _on(self, t: np.ndarray) -> np.ndarray:
        a = max(self.attack, 1e-4)
        if t[0] >= a:
            if self.sustain >= 1.0:
                return np.ones_like(t)
            return self.sustain + (1.0 - self.sustain) * np.exp(-(t - a) / max(self.decay, 1e-4))
        if t[-1] < a:
            return 0.5 - 0.5 * np.cos(np.pi * t / a)
        rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / a, 0.0, 1.0))
        fall = self.sustain + (1.0 - self.sustain) * np.exp(-np.maximum(t - a, 0.0) / max(self.decay, 1e-4))
        return np.where(t < a, rise, fall)

    def level(self, t: np.ndarray, t_off: float | None) -> np.ndarray:
        on = self._on(t)
        if t_off is None:
            return on
        start = float(self._on(np.array([t_off]))[0])
        tau = max(self.release, 1e-3) / 6.0  # ~-52 dB at `release`
        rel = start * np.exp(-np.maximum(t - t_off, 0.0) / tau)
        return np.where(t >= t_off, rel, on)

    def done(self, t: float, t_off: float | None) -> bool:
        if t_off is not None and t > t_off + self.release:
            return True
        return self.sustain <= 0.0 and t > self.attack + 6.0 * self.decay  # ~-52 dB


# --------------------------------------------------------------------------- note


class Note:
    """One sounding note, from a Wave (wavetable) or Partials (additive)."""

    __slots__ = (
        "freq", "amp", "env", "sr", "gl", "gr", "delay", "age", "t_off", "vib", "trem", "finished",
        "tabs", "morph", "osc", "ratios", "amps", "decay", "phase0", "grit", "rng", "nprev",
    )

    def __init__(
        self,
        freq: float,
        amp: float,
        env: Envelope,
        timbre: Wave | Partials,
        sr: int,
        *,
        pan: float = 0.0,
        delay: int = 0,
        duration: float | None = None,
        vibrato: tuple[float, float] = (0.0, 0.0),  # (rate Hz, depth as freq ratio)
        tremolo: tuple[float, float] = (0.0, 0.0),  # (rate Hz, depth 0..1)
        grit: float = 0.0,  # bow/breath noise riding on the waveform, 0..~0.3
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        self.grit, self.rng, self.nprev = grit, rng, 0.0
        self.freq, self.amp, self.env, self.sr = freq, amp, env, sr
        self.gl, self.gr = pan_gains(pan)
        self.delay = int(delay)
        self.age = 0
        self.t_off = duration
        self.vib, self.trem = vibrato, tremolo
        self.finished = False
        self.tabs = None
        if isinstance(timbre, Wave):
            top = freq * 2.0 ** (max(c for c, _ in timbre.unison) / 1200.0) * (1.0 + vibrato[1])
            self.tabs = [_table(timbre.amps, top, sr, rng, timbre.formants)]
            if timbre.dark is not None:
                self.tabs.append(_table(timbre.dark, top, sr, rng, timbre.formants))
            self.morph = timbre.morph
            g = sum(abs(gain) for _, gain in timbre.unison)
            # [ratio, gain, running phase in cycles]
            self.osc = [[2.0 ** (c / 1200.0), gain / g, rng.random()] for c, gain in timbre.unison]
        else:
            keep = timbre.ratios * freq * (1.0 + vibrato[1]) < sr * 0.45
            self.ratios, self.amps, self.decay = timbre.ratios[keep], timbre.amps[keep], timbre.decay[keep]
            self.phase0 = rng.uniform(0, TAU, len(self.ratios))
            self.finished = len(self.ratios) == 0

    def release(self, at_samples: int = 0) -> None:
        """Start the release stage `at_samples` from the start of the next block."""
        t = (self.age + max(0, at_samples - self.delay)) / self.sr
        if self.t_off is None or t < self.t_off:
            self.t_off = t

    def _wavetable(self, t: np.ndarray, m: int) -> np.ndarray:
        vr, vd = self.vib
        pm = (vd / vr) * np.sin(TAU * vr * t) if vd else 0.0  # vibrato as phase offset (cycles)
        ramp = np.arange(m) / self.sr
        a = self.tabs[0]
        b = self.tabs[1] if len(self.tabs) > 1 else None
        out = np.zeros(m)
        for o in self.osc:
            f = self.freq * o[0]
            ph = o[2] + f * ramp + pm * o[0]
            o[2] = (o[2] + f * m / self.sr) % 1.0
            idx = (ph - np.floor(ph)) * TABLE
            i = idx.astype(np.int64)
            fr = idx - i
            y = a[i] + fr * (a[i + 1] - a[i])
            if b is not None:
                yb = b[i] + fr * (b[i + 1] - b[i])
                w = np.exp(-t / self.morph)
                y = yb + w * (y - yb)
            out += o[1] * y
        return out

    def _additive(self, t: np.ndarray) -> np.ndarray:
        ph = self.ratios[:, None] * (TAU * self.freq * t)[None, :] + self.phase0[:, None]
        return (self.amps[:, None] * np.exp(-self.decay[:, None] * t[None, :]) * np.sin(ph)).sum(axis=0)

    def render(self, n: int, out: np.ndarray) -> None:
        """Add this note's next n samples into out (n, 2)."""
        if self.finished:
            return
        if self.delay >= n:
            self.delay -= n
            return
        start, self.delay = self.delay, 0
        m = n - start
        t = (self.age + np.arange(m)) / self.sr
        self.age += m

        sig = self._wavetable(t, m) if self.tabs is not None else self._additive(t)
        if self.grit:
            # Multiplying by smoothed noise gives a rasp that follows the tone's own spectrum.
            nz = self.rng.standard_normal(m)
            nz = 0.5 * (nz + np.concatenate([[self.nprev], nz[:-1]]))
            self.nprev = nz[-1]
            sig = sig * (1.0 + self.grit * 2.0 * nz)
        env = self.env.level(t, self.t_off) * self.amp
        tr, td = self.trem
        if td:
            env = env * (1.0 - td * 0.5 * (1.0 + np.sin(TAU * tr * t)))
        sig *= env
        out[start:, 0] += sig * self.gl
        out[start:, 1] += sig * self.gr

        if self.env.done(t[-1], self.t_off):
            self.finished = True


# --------------------------------------------------------------------------- one-shots
# Drums and sweeps are synthesised once into a buffer, then played back like samples.


class Sample:
    """A precomputed mono sound."""

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray):
        peak = np.abs(data).max()
        self.data = data / peak if peak > 0 else data


class SampleNote:
    """Plays a Sample. Same interface as Note so layers can mix both."""

    __slots__ = ("data", "amp", "gl", "gr", "delay", "pos", "finished")

    def __init__(self, sample: Sample, amp: float, *, pan: float = 0.0, delay: int = 0):
        self.data = sample.data
        self.amp = amp
        self.gl, self.gr = pan_gains(pan)
        self.delay = int(delay)
        self.pos = 0
        self.finished = False

    def release(self, at_samples: int = 0) -> None:
        pass  # one-shots always play out

    def render(self, n: int, out: np.ndarray) -> None:
        if self.finished:
            return
        if self.delay >= n:
            self.delay -= n
            return
        start, self.delay = self.delay, 0
        chunk = self.data[self.pos : self.pos + n - start] * self.amp
        self.pos += len(chunk)
        end = start + len(chunk)
        out[start:end, 0] += chunk * self.gl
        out[start:end, 1] += chunk * self.gr
        if self.pos >= len(self.data):
            self.finished = True


def _band_noise(n: int, sr: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """White noise shaped by a soft band-pass in the frequency domain."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr) + 1e-9
    hp = (f / np.sqrt(f**2 + lo**2)) ** 2
    lp = 1.0 / (1.0 + (f / hi) ** 4)
    return np.fft.irfft(spec * hp * lp, n)


def kick(sr: int, freq: float, rng: np.random.Generator, punch: float = 1.0, decay: float = 0.35,
         drive: float = 1.6) -> Sample:
    n = int(sr * decay * 2.2)
    t = np.arange(n) / sr
    f = freq * (1.0 + 3.5 * punch * np.exp(-t / 0.035))  # fast pitch drop = the "thump"
    body = np.sin(TAU * np.cumsum(f) / sr) * np.exp(-t / decay)
    click = _band_noise(n, sr, 1500, 6000, rng) * np.exp(-t / 0.004) * 0.25 * punch
    x = np.tanh(drive * (body + click))
    x[: int(sr * 0.001)] *= np.linspace(0, 1, int(sr * 0.001))
    return Sample(x)


def snare(sr: int, rng: np.random.Generator, tone: float = 190.0, decay: float = 0.16, bright: float = 0.5) -> Sample:
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    noise = _band_noise(n, sr, 900, 5000 + 6000 * bright, rng) * np.exp(-t / decay)
    body = np.sin(TAU * tone * t * (1 + 0.5 * np.exp(-t / 0.01))) * np.exp(-t / 0.045)
    return Sample(noise * 0.8 + body * 0.6)


def clap(sr: int, rng: np.random.Generator, decay: float = 0.14) -> Sample:
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    env = np.exp(-t / decay)
    for k in (0.0, 0.011, 0.022):  # a few hands slightly out of sync
        env = env + 1.5 * np.exp(-np.maximum(t - k, 0) / 0.004) * (t >= k)
    return Sample(_band_noise(n, sr, 800, 3500, rng) * env)


def hat(sr: int, rng: np.random.Generator, decay: float = 0.045, bright: float = 0.5) -> Sample:
    n = int(sr * max(decay * 6, 0.05))
    t = np.arange(n) / sr
    metal = sum(np.sign(np.sin(TAU * f * t)) for f in (317, 421, 541, 593, 727, 863))  # 808-ish square cluster
    x = _band_noise(n, sr, 6500, 14000 + 4000 * bright, rng) + 0.04 * metal
    x = _band_noise_filter(x, sr, 6000)
    return Sample(x * np.exp(-t / decay))


def _band_noise_filter(x: np.ndarray, sr: int, lo: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr) + 1e-9
    return np.fft.irfft(spec * (f / np.sqrt(f**2 + lo**2)) ** 4, len(x))


def rim(sr: int, rng: np.random.Generator, decay: float = 0.06) -> Sample:
    """Dry rimshot: a short wooden crack from a snare rim hit. Useful as a tight offbeat accent."""
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    # Two slightly detuned tones (the two nodes of the rim) plus a burst of noise.
    body = np.sin(TAU * 1050.0 * t) * 0.6 + np.sin(TAU * 1480.0 * t) * 0.4
    noise = _band_noise(n, sr, 1500, 9000, rng) * 0.5
    env = np.exp(-t / decay)
    click = _band_noise(n, sr, 3000, 12000, rng) * np.exp(-t / 0.003)
    return Sample((body + noise + click) * env)


def cowbell(sr: int, rng: np.random.Generator, decay: float = 0.22) -> Sample:
    """808-style cowbell: two square-wave oscillators (561 Hz + 845 Hz) into a band-pass,
    with a short noise transient at the strike. The beat ratio 1:1.506 gives the
    characteristic metallic clang. Used as the offbeat accent in Euro/italo grooves."""
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    sq = lambda f: np.sign(np.sin(TAU * f * t))  # square wave
    body = sq(561.0) + 0.7 * sq(845.0)
    body = _band_noise_filter(body, sr, 400.0)
    body = body / (np.abs(body).max() + 1e-9)
    env = np.exp(-t / decay)
    click = _band_noise(n, sr, 2000, 8000, rng) * np.exp(-t / 0.003) * 0.3
    return Sample(body * env + click)


def crash(sr: int, rng: np.random.Generator, decay: float = 1.3) -> Sample:
    n = int(sr * decay * 4)
    t = np.arange(n) / sr
    x = _band_noise(n, sr, 3000, 15000, rng) * np.exp(-t / decay)
    x += _band_noise(n, sr, 400, 3000, rng) * np.exp(-t / 0.15) * 0.4
    return Sample(x)


def reverse_cymbal(sr: int, seconds: float, rng: np.random.Generator) -> Sample:
    x = crash(sr, rng, decay=seconds / 3.0).data[: int(sr * seconds)][::-1].copy()
    fade = int(sr * 0.02)
    x[-fade:] *= np.linspace(1, 0, fade)
    return Sample(x)


def riser(sr: int, seconds: float, base_freq: float, rng: np.random.Generator,
          noise: float = 0.7, tone: float = 0.25, octaves: float = 1.5) -> Sample:
    """Noise sweeping from low to high bands plus a saw-ish tone gliding up `octaves`."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    pos = t / seconds  # 0..1
    edges = np.geomspace(300, 12000, 7)
    out = np.zeros(n)
    for i in range(len(edges) - 1 if noise else 0):
        center = i / (len(edges) - 2)
        w = np.exp(-((pos - center) ** 2) / 0.04)  # each band peaks as the sweep passes it
        out += _band_noise(n, sr, edges[i], edges[i + 1], rng) * w
    f = base_freq * 2.0 ** (octaves * pos**2)
    ph = TAU * np.cumsum(f) / sr
    saw = sum(np.sin(k * ph) / k for k in range(1, 7))
    x = (out * noise + saw * tone) * pos**2.2
    fade = int(sr * 0.02)
    x[-fade:] *= np.linspace(1, 0, fade)
    return Sample(x)


# --------------------------------------------------------------------------- acoustic one-shots


def _shape(x: np.ndarray, sr: int, formants=(), lowpass: float | None = None, highpass: float | None = None) -> np.ndarray:
    """Static EQ of a whole buffer in the frequency domain."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr) + 1e-9
    g = formant_gain(f, formants) if formants else np.ones_like(f)
    if lowpass:
        g = g / (1.0 + (f / lowpass) ** 2)
    if highpass:
        g = g * (f / np.sqrt(f**2 + highpass**2)) ** 2
    return np.fft.irfft(spec * g, len(x))


LUTE_BODY = ((110.0, 60.0, 1.2), (240.0, 90.0, 0.8), (2600.0, 900.0, 0.5))


def ks_pluck(sr: int, freq: float, rng: np.random.Generator, decay: float = 1.5, bright: float = 0.5,
             body=LUTE_BODY, courses: int = 1, detune: float = 0.0) -> Sample:
    """Karplus-Strong plucked string, computed one period at a time (vectorised), then
    resampled to the exact pitch. `courses` > 1 adds slightly detuned doubled strings
    (lute/dulcimer courses), which shimmer against each other."""
    decay = decay * min(1.0, (220.0 / freq) ** 0.3)  # high strings ring shorter, like real ones
    total = int(sr * decay * 2.2)
    out = np.zeros(total)
    for c in range(courses):
        f = freq * 2.0 ** ((c - (courses - 1) / 2) * detune / 1200.0)
        period = sr / f
        n = int(np.ceil(period))
        reps = total // n + 2
        g = 10.0 ** (-3.0 * n / (sr * decay))  # loss per period -> ~T60 = decay
        a = 0.5 + 0.45 * bright  # >0.5 keeps more highs
        x = rng.uniform(-1, 1, n)
        for _ in range(int(3 * (1 - bright))):  # softer pluck = smoother excitation
            x = 0.5 * (x + np.roll(x, 1))
        x -= x.mean()
        y = np.empty(reps * n)
        y[:n] = x
        before = 0.0  # sample preceding the current period
        for k in range(1, reps):
            prev = y[(k - 1) * n : k * n]
            shifted = np.concatenate([[before], prev[:-1]])
            y[k * n : (k + 1) * n] = g * (a * prev + (1 - a) * shifted)
            before = prev[-1]
        # The averaging filter adds (1 - a) samples of delay per period; resample so the
        # pitch comes out exact.
        pos = np.arange(total) * ((n + 1 - a) / period)
        out += np.interp(pos, np.arange(len(y)), y)
    if body:
        out = _shape(out, sr, body, highpass=60.0)
    fade = int(sr * 0.05)
    out[-fade:] *= np.linspace(1, 0, fade)
    return Sample(out)


def buzz(sr: int, freq: float, rng: np.random.Generator, dur: float = 0.14) -> Sample:
    """Hurdy-gurdy trompette 'dog': the bridge rattles against the soundboard in pulses."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    ph = (freq * t) % 1.0
    rattle = (ph < 0.12).astype(np.float64) * rng.uniform(0.6, 1.0, n)  # one hit per cycle
    rattle += 0.3 * _band_noise(n, sr, 1500, 5000, rng)
    env = np.minimum(1.0, t / 0.008) * np.exp(-t / (dur * 0.35))
    return Sample(_shape(rattle * env, sr, ((1200.0, 500.0, 2.0), (2800.0, 900.0, 1.5)), highpass=300.0))


def frame_drum(sr: int, rng: np.random.Generator, freq: float = 85.0, decay: float = 0.22, slap: float = 0.25) -> Sample:
    """Bodhran/frame drum: a few membrane modes plus the slap of the hand or tipper."""
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    modes = ((1.0, 1.0), (1.59, 0.5), (2.14, 0.35), (2.65, 0.2))
    x = sum(a * np.sin(TAU * freq * r * t * (1 + 0.08 * np.exp(-t / 0.02))) * np.exp(-t * r / decay)
            for r, a in modes)
    x = x + slap * _band_noise(n, sr, 300, 3000, rng) * np.exp(-t / 0.012)
    return Sample(x)


def tambourine(sr: int, rng: np.random.Generator, decay: float = 0.12, shake: bool = False) -> Sample:
    """Jingles: clusters of high inharmonic partials over noise. shake=True softens the attack."""
    n = int(sr * decay * 5)
    t = np.arange(n) / sr
    x = 0.6 * _band_noise(n, sr, 5000, 14000, rng)
    for f in rng.uniform(4000, 11000, 14):
        x += 0.15 * np.sin(TAU * f * t + rng.uniform(0, TAU)) * np.exp(-t / rng.uniform(0.04, 0.12))
    env = np.exp(-t / decay)
    if shake:
        env *= np.minimum(1.0, t / 0.03)
    return Sample(x * env)


def drum_roll(sr: int, seconds: float, rng: np.random.Generator, freq: float = 85.0) -> Sample:
    """Frame drum roll that speeds up and swells into the downbeat, over a tambourine shake."""
    n = int(sr * seconds)
    out = np.zeros(n + sr)
    hit = frame_drum(sr, rng, freq, 0.12, 0.35).data
    t = 0.0
    while t < seconds:
        pos = t / seconds
        i = int(t * sr)
        out[i : i + len(hit)] += hit * (0.15 + 0.85 * pos**1.5) * rng.uniform(0.8, 1.0)
        t += 0.22 * (1 - pos) ** 1.3 + 0.035
    shake = tambourine(sr, rng, decay=seconds / 2).data
    m = min(n, len(shake))
    out[:m] += 0.35 * shake[:m][::-1] * np.linspace(0, 1, m) ** 2
    x = out[:n]
    fade = int(sr * 0.02)
    x[-fade:] *= np.linspace(1, 0, fade)
    return Sample(x)


# --------------------------------------------------------------------------- effects


class _Lowpass1:
    """Two-tap smoothing filter applied blockwise along the last axis, with carry state."""

    def __init__(self, shape: tuple[int, ...]):
        self.prev = np.zeros(shape)

    def __call__(self, x: np.ndarray, amount: float) -> np.ndarray:
        shifted = np.concatenate([self.prev[..., None], x[..., :-1]], axis=-1)
        self.prev = x[..., -1].copy()
        return (1.0 - amount) * x + amount * shifted


class Echo:
    """Stereo ping-pong delay with darkening repeats. Delay time glides smoothly when changed."""

    def __init__(self, sr: int, block: int, max_seconds: float = 4.0):
        self.sr = sr
        self.block = block
        self.size = int(max_seconds * sr) + 2 * block
        self.buf = np.zeros((2, self.size))
        self.w = 0
        self.d = None  # current delay in samples (float)
        self.lp = _Lowpass1((2,))

    def process(self, x: np.ndarray, time_s: float, feedback: float, damp: float) -> np.ndarray:
        n = len(x)
        target = min(max(time_s * self.sr, self.block + 2), self.size - self.block - 2)
        d0 = target if self.d is None else self.d
        d1 = d0 + (target - d0) * 0.05
        self.d = d1
        pos = self.w - (d0 + (d1 - d0) * np.arange(n) / n) + np.arange(n)
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        i0 %= self.size
        i1 = (i0 + 1) % self.size
        y = self.buf[:, i0] * (1 - frac) + self.buf[:, i1] * frac  # (2, n)

        yd = self.lp(y, 0.5 * damp)
        mono = 0.5 * (x[:, 0] + x[:, 1])
        wi = (self.w + np.arange(n)) % self.size
        self.buf[0, wi] = mono + feedback * yd[1]  # ping
        self.buf[1, wi] = feedback * yd[0]  # pong
        self.w = (self.w + n) % self.size
        return y.T


class Reverb:
    """8-line feedback delay network with Hadamard mixing and damping, preceded by a short
    noise-burst diffuser (FFT overlap-add) so plucks don't turn into flutter echoes."""

    BASE_MS = np.array([43.7, 51.1, 59.3, 67.9, 73.1, 83.3, 91.7, 103.1])

    def __init__(self, sr: int, block: int, size: float = 1.0, seed: int = 0, low_cut: float = 220.0):
        self.sr = sr
        self.block = block
        d = np.round(self.BASE_MS * size * sr / 1000.0).astype(np.int64)
        self.d = np.maximum(d, block + 1)
        self.size = int(self.d.max()) + block + 1
        self.buf = np.zeros((8, self.size))
        self.w = 0
        h = np.array([[1.0]])
        for _ in range(3):
            h = np.block([[h, h], [h, -h]])
        self.H = h / math.sqrt(8.0)
        self.lp = _Lowpass1((8,))
        rng = np.random.default_rng(seed)
        self.in_sign = rng.choice([-1.0, 1.0], size=(8, 2))
        self.out_sign = rng.choice([-1.0, 1.0], size=8)

        # Diffuser: ~40 ms of decaying stereo noise.
        m = int(0.04 * sr)
        tail = np.exp(-np.arange(m) / (m / 4.0))
        ir = rng.standard_normal((2, m)) * tail
        ir /= np.sqrt((ir**2).sum(axis=1, keepdims=True))
        self.nfft = 1 << int(math.ceil(math.log2(block + m)))
        self.ir_f = np.fft.rfft(ir, self.nfft)
        # Second-order high-pass folded into the diffuser: sustained bass notes otherwise
        # excite the network's low resonances and the level swings with pitch.
        f = np.fft.rfftfreq(self.nfft, 1.0 / sr)
        self.ir_f *= (f / np.sqrt(f**2 + low_cut**2)) ** 2
        self.ola = np.zeros((2, self.nfft))

    def _diffuse(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        y = np.fft.irfft(np.fft.rfft(x.T, self.nfft) * self.ir_f, self.nfft)
        self.ola += y
        out = self.ola[:, :n].copy()
        self.ola = np.concatenate([self.ola[:, n:], np.zeros((2, n))], axis=1)
        return out  # (2, n)

    def process(self, x: np.ndarray, t60: float, damp: float) -> np.ndarray:
        n = len(x)
        xin = self._diffuse(x)
        idx = np.arange(n)
        rd = (self.w - self.d[:, None] + idx[None, :]) % self.size
        y = np.take_along_axis(self.buf, rd, axis=1)  # (8, n)
        g = 10.0 ** (-3.0 * self.d / (self.sr * max(t60, 0.1)))
        fb = self.H @ (self.lp(y, damp) * g[:, None])
        wi = (self.w + idx) % self.size
        self.buf[:, wi] = fb + self.in_sign @ xin * 0.35
        self.w = (self.w + n) % self.size
        ys = y * self.out_sign[:, None]
        return np.stack([ys[0::2].sum(axis=0), ys[1::2].sum(axis=0)], axis=1) * 0.5
