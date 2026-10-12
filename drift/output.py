"""Ways to get audio out of an Engine: pygame mixer streaming, WAV files, raw PCM."""

from __future__ import annotations

import collections
import json
import sys
import threading
import time
import wave

import numpy as np

from .engine import Engine


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


class PygamePlayer:
    """Streams an Engine through pygame.mixer on a dedicated channel, from a background thread.

    pygame.mixer must already be initialised (16-bit). The engine's sample rate must match
    the mixer's; use PygamePlayer.create() to get a matching engine automatically.
    By default it plays on channel 0 and reserves it, leaving the others to Sound.play().
    For crossfades between two players, reserve 2 channels and pass channel=0 / channel=1.
    Parameter changes are heard after roughly `chunk * (ahead + 1)` seconds.

    Rendering and feeding the mixer happen on two threads: a renderer keeps `ahead`
    chunks ready, and a feeder only hands them to pygame (whose channel queue holds a
    single chunk). So a slow render, or another thread hogging Python for a moment,
    eats into that reserve instead of being heard as a gap.
    """

    def __init__(self, engine: Engine, chunk: float = 0.5, channel: int | None = None,
                 ahead: int = 1):
        import pygame

        init = pygame.mixer.get_init()
        if not init:
            raise RuntimeError("call pygame.mixer.init() first")
        freq, size, chans = init[:3]
        if freq != engine.sr:
            raise ValueError(f"engine sr {engine.sr} != mixer frequency {freq}")
        if abs(size) != 16:
            raise ValueError("mixer must be 16-bit")
        self.pg = pygame
        self.engine = engine
        self.chans = chans
        self.frames = int(chunk * engine.sr)
        if channel is None:
            channel = 0
            pygame.mixer.set_reserved(1)  # keep Sound.play() from stealing it
        self.channel = pygame.mixer.Channel(channel)
        self.volume = 1.0
        self.ahead = max(1, ahead)
        self.underruns = 0  # times the queue ran dry and playback had to restart
        self.render_ms: list[float] = []  # recent chunk render times
        self._stop = threading.Event()
        self._ready: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._threads: list[threading.Thread] = []

    @classmethod
    def create(cls, preset="meadow", seed=None, **kw) -> PygamePlayer:
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init(44100, -16, 2, 1024)
        return cls(Engine(preset, sr=pygame.mixer.get_init()[0], seed=seed), **kw)

    def _sound(self):
        t = time.perf_counter()
        x = self.engine.render(self.frames)
        self.render_ms = self.render_ms[-199:] + [(time.perf_counter() - t) * 1000]
        if self.chans == 1:
            x = x.mean(axis=1, keepdims=True)
        elif self.chans > 2:
            x = np.concatenate([x, np.zeros((len(x), self.chans - 2), np.float32)], axis=1)
        return self.pg.mixer.Sound(buffer=to_int16(x).tobytes())

    def _render(self):
        while not self._stop.is_set():
            with self._cv:
                while len(self._ready) >= self.ahead and not self._stop.is_set():
                    self._cv.wait(0.1)
            if not self._stop.is_set():
                sound = self._sound()
                with self._cv:
                    self._ready.append(sound)
                    self._cv.notify_all()

    def _next(self, wait: bool = False):
        with self._cv:
            if wait:
                while not self._ready and not self._stop.is_set():
                    self._cv.wait(0.1)
            sound = self._ready.popleft() if self._ready else None
            self._cv.notify_all()
            return sound

    def _feed(self):
        ch = self.channel
        ch.play(self._next(wait=True))
        while not self._stop.is_set():
            if ch.get_queue() is None and (sound := self._next()) is not None:
                if not ch.get_busy():  # ran dry before the next chunk was ready
                    self.underruns += 1
                ch.queue(sound)  # starts at once on an idle channel
            elif not ch.get_busy() and (sound := self._next()) is not None:
                self.underruns += 1
                ch.play(sound)
            time.sleep(0.01)

    def start(self) -> PygamePlayer:
        if not self._threads:
            self._stop.clear()
            self._ready.clear()
            self._threads = [threading.Thread(target=f, name=n, daemon=True)
                             for f, n in ((self._render, "drift-render"), (self._feed, "drift-feed"))]
            for t in self._threads:
                t.start()
        return self

    def stop(self, fade_ms: int = 2000) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join()
        self._threads = []
        if fade_ms:
            self.channel.fadeout(fade_ms)
        else:
            self.channel.stop()

    def set_volume(self, v: float) -> None:
        self.volume = v
        self.channel.set_volume(v)


def write_wav(engine: Engine, path: str, seconds: float, progress: bool = False) -> None:
    frames = int(seconds * engine.sr)
    chunk = engine.sr * 5
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(engine.sr)
        done = 0
        while done < frames:
            n = min(chunk, frames - done)
            w.writeframes(to_int16(engine.render(n)).tobytes())
            done += n
            if progress:
                print(f"\r{done / engine.sr:6.0f}s / {seconds:.0f}s  {engine.state()['chord']:10}",
                      end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)


def stream_raw(engine: Engine, out=None, chunk: float = 0.25) -> None:
    """Write s16le stereo PCM forever (pipe into pw-play / aplay / ffplay)."""
    out = out or sys.stdout.buffer
    n = int(chunk * engine.sr)
    try:
        while True:
            out.write(to_int16(engine.render(n)).tobytes())
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass


def save_config(engine: Engine, path: str) -> str:
    with open(path, "w") as f:
        json.dump(engine.config(), f, indent=2)
        f.write("\n")
    return path


def load_config(path: str, **kw) -> Engine:
    with open(path) as f:
        return Engine.from_config(json.load(f), **kw)
