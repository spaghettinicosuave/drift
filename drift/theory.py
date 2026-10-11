"""Scales, chords and the slowly wandering harmony that everything else follows."""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Ordered bright -> dark. `tension` picks a position along this list.
MODES: dict[str, list[int]] = {
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
}
MODE_ORDER = list(MODES)

# Root movements in scale steps and their weights: 4ths/5ths and 3rds feel smooth.
_MOVES = [(3, 4.0), (4, 3.0), (5, 3.0), (2, 2.0), (-2, 1.5), (1, 1.0), (-1, 1.0)]


def note_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


# ------------------------------------------------------------------ meters

# Steps (16th notes) per note value of the time signature's bottom number.
_UNIT = {2: 8, 4: 4, 8: 2, 16: 1}


@dataclass(frozen=True)
class Meter:
    """A time signature as beat groups. `groups` are beat lengths in 16th-note steps:
    4/4 = (4, 4, 4, 4), 6/8 = (6, 6), 7/8 = (4, 4, 6). Tempo counts `beat_steps`-long beats:
    the beat when all beats are equal (dotted quarters in 6/8), else quarter notes."""

    name: str
    groups: tuple[int, ...]

    @property
    def bar_steps(self) -> int:
        return sum(self.groups)

    @property
    def beat_steps(self) -> int:
        return self.groups[0] if len(set(self.groups)) == 1 else 4

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(sum(self.groups[:i]) for i in range(len(self.groups)))

    @property
    def even(self) -> bool:
        return len(set(self.groups)) == 1

    def beat_map(self) -> list[tuple[int, int, int]]:
        """For each step in the bar: (beat index, step within the beat, beat length)."""
        return [(i, k, n) for i, n in enumerate(self.groups) for k in range(n)]


@functools.lru_cache(maxsize=64)
def parse_meter(text: str) -> Meter:
    """'4/4', '3/4', '6/8', '5/4', '7/8', '12/8', '2/2'... or explicit groups like '3+2+2/8'.
    /8 meters group in 3s when they can (6/8, 9/8, 12/8), else in 2s with a 3 at the end
    (7/8 = 2+2+3); 8/8 is 3+3+2."""
    s = text.replace(" ", "")
    try:
        top, bottom = s.split("/")
        unit = _UNIT[int(bottom)]
        if "+" in top:
            counts = [int(x) for x in top.split("+")]
        else:
            n = int(top)
            if unit > 2:
                counts = [1] * n
            elif n % 3 == 0:
                counts = [3] * (n // 3)
            elif n == 8:
                counts = [3, 3, 2]
            else:
                counts = [2] * ((n - 3) // 2) + [3] if n % 2 else [2] * (n // 2)
    except (ValueError, KeyError):
        raise ValueError(f"can't read time signature {text!r}; try e.g. '4/4', '7/8' or '3+3+2/8'") from None
    groups = tuple(c * unit for c in counts)
    if not groups or min(groups) < 2 or not 4 <= sum(groups) <= 64:
        raise ValueError(f"time signature {text!r} is out of range (bars of 4-64 sixteenths, beats of 2+)")
    return Meter(s, groups)


class Harmony:
    """Key + mode + current chord, expressed in scale degrees (0 = tonic)."""

    def __init__(self, rng: random.Random, root: int | None = None, mode: str = "ionian"):
        self.rng = rng
        self.root = rng.randrange(12) if root is None else root % 12
        self.mode = mode
        self.degree = 0
        self.chord: list[int] = [0, 2, 4]
        self.bars_since_home = 0

    # ---- pitch helpers

    @property
    def scale(self) -> list[int]:
        return MODES[self.mode]

    def midi(self, degree: int, octave: int = 4) -> int:
        """Scale degree (any integer, wraps across octaves) -> MIDI note. octave 4 = middle C area."""
        o, d = divmod(degree, 7)
        return 12 * (octave + 1 + o) + self.root + self.scale[d]

    def near(self, degree: int, target_midi: float) -> int:
        """MIDI note for `degree` in whichever octave lands closest to target_midi."""
        base = self.midi(degree % 7, 0)
        k = round((target_midi - base) / 12.0)
        return base + 12 * k

    def chord_midis(self, low: int, high: int) -> list[int]:
        """All chord tones between low and high (inclusive)."""
        pcs = {(self.root + self.scale[d % 7]) % 12 for d in self.chord}
        return [m for m in range(low, high + 1) if m % 12 in pcs]

    def scale_midis(self, low: int, high: int) -> list[int]:
        pcs = {(self.root + s) % 12 for s in self.scale}
        return [m for m in range(low, high + 1) if m % 12 in pcs]

    def _interval(self, degree: int, steps: int) -> int:
        return (self.midi(degree + steps, 4) - self.midi(degree, 4)) % 12

    def _fifth_ok(self, degree: int) -> bool:
        return self._interval(degree, 4) == 7

    # ---- movement

    def next_chord(self, tension: float) -> None:
        """Pick the next chord. More tension -> fewer returns home, more colour tones."""
        self.bars_since_home += 1
        home_pull = min(0.6, 0.08 * self.bars_since_home) * (1.0 - 0.6 * tension)
        if self.degree != 0 and self.rng.random() < home_pull:
            deg = 0
        else:
            for _ in range(20):
                step = self.rng.choices([m for m, _ in _MOVES], [w for _, w in _MOVES])[0]
                deg = (self.degree + step) % 7
                if deg != self.degree and self._fifth_ok(deg):
                    break
            else:
                deg = 0
        if deg == 0:
            self.bars_since_home = 0
        self.degree = deg

        chord = [deg, deg + 2, deg + 4]
        r = self.rng.random()
        if r < 0.25 + 0.4 * tension:
            chord.append(deg + 6)  # 7th
        if self.rng.random() < 0.3:
            chord.append(deg + 8)  # add9
        if self.rng.random() < 0.12 + 0.2 * tension:
            chord[1] = deg + 3  # sus4
        self.chord = chord

    def drift(self, tension: float) -> str | None:
        """Occasionally modulate. Returns a description of the change, or None."""
        target = MODE_ORDER[min(len(MODE_ORDER) - 1, int(tension * len(MODE_ORDER)))]
        if target != self.mode and self.rng.random() < 0.5:
            # Step one mode towards the target, keeping the same tonic (a colour change).
            i, j = MODE_ORDER.index(self.mode), MODE_ORDER.index(target)
            self.mode = MODE_ORDER[i + (1 if j > i else -1)]
            return f"mode -> {self.mode}"
        if self.rng.random() < 0.08:
            # Move the key by a fifth: shares all but one note, so it feels seamless.
            self.root = (self.root + self.rng.choice([7, 5])) % 12
            self.degree = 0
            return f"key -> {note_name(self.root)}"
        return None

    # ---- naming

    def chord_name(self) -> str:
        d = self.degree
        root = note_name(self.root + self.scale[d % 7])
        tones = {(self.midi(x, 4) - self.midi(d, 4)) % 12 for x in self.chord}
        sus = 5 in tones and 3 not in tones and 4 not in tones
        q = "m" if 3 in tones else ""
        if 11 in tones:
            q += "maj7"
        elif 10 in tones:
            q += "7"
        if sus:
            q += "sus4"
        if 2 in tones:
            q += "add9"
        return root + q

    def key_name(self) -> str:
        return f"{note_name(self.root)} {self.mode}"
