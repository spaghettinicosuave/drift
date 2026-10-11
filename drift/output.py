"""Ways to get audio out of an Engine: pygame mixer streaming, WAV files, raw PCM."""

from __future__ import annotations

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
    Parameter changes are heard after roughly `chunk` seconds.
    """

    def __init__(self, engine: Engine, chunk: float = 0.5, channel: int | None = None):
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
            channel = pygame.mixer.get_num_channels() - 1
            pygame.mixer.set_reserved(channel + 1)  # keep Sound.play() from stealing it
        self.channel = pygame.mixer.Channel(channel)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def create(cls, preset="meadow", seed=None, **kw) -> PygamePlayer:
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init(44100, -16, 2, 1024)
        return cls(Engine(preset, sr=pygame.mixer.get_init()[0], seed=seed), **kw)

    def _sound(self):
        x = self.engine.render(self.frames)
        if self.chans == 1:
            x = x.mean(axis=1, keepdims=True)
        elif self.chans > 2:
            x = np.concatenate([x, np.zeros((len(x), self.chans - 2), np.float32)], axis=1)
        return self.pg.mixer.Sound(buffer=to_int16(x).tobytes())

    def _run(self):
        ch = self.channel
        ch.play(self._sound())
        while not self._stop.is_set():
            if ch.get_queue() is None:
                ch.queue(self._sound())
            elif not ch.get_busy():  # underrun (e.g. the game stalled): restart
                ch.play(self._sound())
            time.sleep(0.02)

    def start(self) -> PygamePlayer:
        if not self._thread:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="drift", daemon=True)
            self._thread.start()
        return self

    def stop(self, fade_ms: int = 2000) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        if fade_ms:
            self.channel.fadeout(fade_ms)
        else:
            self.channel.stop()

    def set_volume(self, v: float) -> None:
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
