"""Run all audio (music + sound effects) in a separate process.

In a game the audio has to keep flowing while the main loop is busy: loading a room,
building a level, a heavy frame. Inside the game's process it can't, because Python
runs one thread at a time, so a busy game starves the music thread and the queue runs
dry (an audible hitch). In its own process it has its own interpreter and its own
core, and the game only ever sends small messages.

    from drift.server import AudioServer
    audio = AudioServer("cave", music_volume=0.6)
    audio.play("hit_bone", pan=0.3)           # returns at once
    audio.set(drive=0.7, tension=0.6)          # music parameters, glided as usual
    audio.load("meadow")                       # presets, variants, meters all work
    audio.stop()

The child is `python -m drift.server`; it opens its own audio device through pygame's
mixer (the game does not need pygame.mixer at all) and exits when the game does.
Set DRIFT_DISKAUDIOFILE=path to record exactly what it plays (SDL's disk driver),
which is how hitches are measured: see stats().
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading


class AudioServer:
    """Parent-side handle. Every method just writes one JSON line to the child."""

    def __init__(self, preset: str | dict = "meadow", *, seed: int | None = None, sr: int = 44100,
                 music_volume: float = 1.0, sfx_volume: float = 1.0, glide: float = 6.0,
                 python: str | None = None):
        cfg = dict(preset=preset, seed=seed, sr=sr, music_volume=music_volume,
                   sfx_volume=sfx_volume, glide=glide)
        env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1")
        self.proc = subprocess.Popen(
            [python or sys.executable, "-m", "drift.server", json.dumps(cfg)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1,
        )
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, *msg) -> None:
        if not self.alive:
            return
        line = json.dumps(msg) + "\n"
        with self._lock:
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    # music
    def set(self, **params) -> None:
        self.send("set", params)

    def load(self, preset: str | dict, fade: float | None = None) -> None:
        self.send("load", preset, fade)

    def set_variant(self, layer: str, variant: str, fade: float | None = None) -> None:
        self.send("variant", layer, variant, fade)

    def set_meter(self, meter: str) -> None:
        self.send("meter", meter)

    # effects
    def play(self, name: str, pan: float = 0.0, volume: float = 1.0, **params) -> None:
        self.send("play", name, pan, volume, params)

    def warm(self, *names: str, **params) -> None:
        self.send("warm", names, params)

    def volume(self, music: float | None = None, sfx: float | None = None) -> None:
        self.send("volume", music, sfx)

    def stats(self, timeout: float = 2.0) -> dict | None:
        """Ask the child how it is doing: underruns, render times, state."""
        import select

        self.send("stats")
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        return json.loads(self.proc.stdout.readline()) if ready else None

    def stop(self, timeout: float = 2.0) -> None:
        if self.alive:
            self.send("quit")
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---------------------------------------------------------------------- child


def _serve(cfg: dict) -> None:
    import queue
    import time

    if os.environ.get("DRIFT_DISKAUDIOFILE"):
        os.environ["SDL_AUDIODRIVER"] = "disk"
        os.environ["SDL_DISKAUDIOFILE"] = os.environ["DRIFT_DISKAUDIOFILE"]
    import pygame

    from . import sfx as sfxmod
    from .engine import Engine
    from .output import PygamePlayer

    pygame.mixer.init(cfg["sr"], -16, 2, 512)
    sr = pygame.mixer.get_init()[0]
    engine = Engine(cfg["preset"], sr=sr, seed=cfg["seed"], glide=cfg["glide"])
    music = PygamePlayer(engine, chunk=0.25, ahead=4).start()
    music.set_volume(cfg["music_volume"])
    fx = sfxmod.SfxPlayer(music=music, volume=cfg["sfx_volume"])

    inbox: queue.Queue = queue.Queue()

    def read():
        for line in sys.stdin:
            inbox.put(line)
        inbox.put(None)  # the game went away

    def mtime():
        try:
            return os.stat(sfxmod.TUNING_PATH).st_mtime
        except OSError:
            return None

    threading.Thread(target=read, daemon=True).start()
    tuning, checked = mtime(), time.monotonic()
    while True:
        if time.monotonic() - checked > 1.0:  # sfx_lab saved: hear it without a restart
            checked = time.monotonic()
            if (m := mtime()) != tuning:
                tuning = m
                sfxmod.load_tuning()
                fx.clear()
        try:
            line = inbox.get(timeout=0.01)
        except queue.Empty:
            fx.update()
            continue
        if line is None:
            break
        try:
            cmd, *args = json.loads(line)
            if cmd == "quit":
                break
            if cmd == "play":
                name, pan, volume, params = args
                fx.play(name, pan=pan, volume=volume, **params)
            elif cmd == "set":
                engine.set(**args[0])
            elif cmd == "load":
                engine.load(args[0], fade=args[1])
            elif cmd == "variant":
                engine.set_variant(*args)
            elif cmd == "meter":
                engine.set_meter(args[0])
            elif cmd == "warm":
                fx.warm(*args[0], **args[1])
            elif cmd == "volume":
                if args[0] is not None:
                    music.set_volume(args[0])
                if args[1] is not None:
                    fx.volume = args[1]
            elif cmd == "stats":
                ms = music.render_ms or [0.0]
                print(json.dumps({
                    "underruns": music.underruns, "render_ms_mean": sum(ms) / len(ms),
                    "render_ms_max": max(ms), "chunk_ms": music.frames / sr * 1000,
                    "slow_renders": sum(m > music.frames / sr * 1000 for m in ms),
                    "state": {k: engine.state()[k] for k in ("key", "chord", "meter", "variants")},
                }), flush=True)
        except Exception as e:  # one bad message must not kill the music
            print(f"drift.server: {line.strip()[:80]}: {e}", file=sys.stderr)
        fx.update()
    music.stop(fade_ms=300)
    time.sleep(0.3)
    pygame.mixer.quit()


if __name__ == "__main__":
    _serve(json.loads(sys.argv[1]))
