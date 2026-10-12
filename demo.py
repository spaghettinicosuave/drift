"""Interactive demo: python demo.py [preset | saved-config.json]

  Up/Down      select parameter        Left/Right   change it
  1-9, 0       glide to a preset        D            fade back to the default instruments
  Q..P         toggle a layer          Ctrl+Q..P    solo a layer (again to unsolo)
  Shift+Q..P   crossfade a layer to its next variant
  V            re-roll all variants (crossfades)
  M            next time signature     Shift+M      type one (7/8, 5/4, 3+3+2/8...), Enter to go
  N            new random seed         G            type a seed (Ctrl+V pastes), Enter to go
  C            copy seed               S            save config to configs/ (and copy its JSON)
  Esc          quit

Variants, layer toggles and parameters carry over when the seed changes.
"""

import collections
import json
import os
import shutil
import subprocess
import sys
import time

import pygame

from drift import PARAMS, PRESETS, Engine, PygamePlayer, load_config, save_config
from drift.engine import METERS

STEP = {"tempo": 4.0, "energy": 0.05, "drive": 0.05, "tension": 0.05, "brightness": 0.05, "space": 0.05, "volume": 0.05}
LIMITS = {"tempo": (40.0, 140.0)}
LAYER_KEYS = [pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y,
              pygame.K_u, pygame.K_i, pygame.K_o, pygame.K_p]
PRESET_NAMES = list(PRESETS)
PRESET_KEYS = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
               pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

BG, FG, DIM, ACC, BAR, WARN = (16, 18, 26), (220, 222, 230), (110, 115, 135), (130, 190, 255), (60, 66, 88), (255, 200, 120)


def copy_text(text: str) -> bool:
    # External tools first: on X11 they keep serving the clipboard after we exit.
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=text.encode(), check=False)
            return True
    try:
        pygame.scrap.put_text(text)
        return True
    except Exception:
        return False


def paste_text() -> str:
    for cmd in (["wl-paste", "-n"], ["xclip", "-o", "-selection", "clipboard"], ["xsel", "-ob"]):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, capture_output=True, check=False)
            if r.returncode == 0:
                return r.stdout.decode(errors="ignore")
    try:
        return pygame.scrap.get_text() or ""
    except Exception:
        return ""


def main() -> None:
    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    pygame.mixer.set_reserved(2)
    screen = pygame.display.set_mode((880, 600))
    pygame.display.set_caption("drift")
    font = pygame.font.SysFont("monospace", 16)
    big = pygame.font.SysFont("monospace", 28, bold=True)
    clock = pygame.time.Clock()
    sr = pygame.mixer.get_init()[0]

    log: collections.deque[str] = collections.deque(maxlen=8)
    arg = sys.argv[1] if len(sys.argv) > 1 else "meadow"
    chan = 0

    def hook(eng: Engine) -> Engine:
        eng.on_event(lambda kind, info: log.append(
            f"{kind:5} {info.get('chord') or info.get('change')}") if kind != "bar" else None)
        return eng

    def start(eng: Engine) -> PygamePlayer:
        log.append(f"seed {eng.seed}")
        return PygamePlayer(hook(eng), chunk=0.25, ahead=2, channel=chan).start()  # ~0.75 s to hear a change

    def reseed(old: Engine, seed=None) -> Engine:
        """Same params, variants and toggles; new music."""
        cfg = old.config()
        cfg["seed"] = seed
        eng = Engine.from_config(cfg, sr=sr)
        eng.set(immediate=True, **{k: old.p[k] for k in PARAMS})  # continue from where the glide was
        eng.set(**old.target)
        return eng

    if arg.endswith(".json"):
        player = start(load_config(arg, sr=sr))
        log.append(f"loaded {os.path.basename(arg)}")
    else:
        player = start(Engine(arg, sr=sr))

    sel = 1
    seed_input: str | None = None  # not None while typing a seed
    meter_input: str | None = None  # not None while typing a time signature
    solo: tuple[str, dict[str, bool]] | None = None  # (soloed layer, enabled states before)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type != pygame.KEYDOWN:
                continue
            eng = player.engine
            ctrl, shift = ev.mod & pygame.KMOD_CTRL, ev.mod & pygame.KMOD_SHIFT

            if meter_input is not None:  # ---- typing a time signature
                if ev.key == pygame.K_ESCAPE:
                    meter_input = None
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    try:
                        eng.set_meter(meter_input)
                        log.append(f"meter -> {meter_input} (next bar)")
                    except ValueError as e:
                        log.append(str(e)[:80])
                    meter_input = None
                elif ev.key == pygame.K_BACKSPACE:
                    meter_input = meter_input[:-1]
                elif ev.unicode and ev.unicode in "0123456789/+":
                    meter_input = (meter_input + ev.unicode)[:12]
                continue

            if seed_input is not None:  # ---- typing a seed
                if ev.key == pygame.K_ESCAPE:
                    seed_input = None
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if seed_input:
                        player.stop(fade_ms=4000)
                        chan = 1 - chan
                        player = start(reseed(eng, int(seed_input)))
                    seed_input = None
                elif ev.key == pygame.K_BACKSPACE:
                    seed_input = seed_input[:-1]
                elif ctrl and ev.key == pygame.K_v:
                    seed_input = (seed_input + "".join(c for c in paste_text() if c.isdigit()))[:12]
                elif ev.unicode.isdigit():
                    seed_input = (seed_input + ev.unicode)[:12]
                continue

            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_UP:
                sel = (sel - 1) % len(PARAMS)
            elif ev.key == pygame.K_DOWN:
                sel = (sel + 1) % len(PARAMS)
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                k = PARAMS[sel]
                lo, hi = LIMITS.get(k, (0.0, 1.0))
                d = STEP[k] * (1 if ev.key == pygame.K_RIGHT else -1)
                eng.set(**{k: min(hi, max(lo, eng.target[k] + d))})
            elif ev.key in PRESET_KEYS[: len(PRESET_NAMES)]:
                name = PRESET_NAMES[PRESET_KEYS.index(ev.key)]
                eng.load(name)
                log.append(f"preset {name}")
            elif ev.key == pygame.K_d:
                for layer, name in eng.default_variants().items():
                    eng.set_variant(layer, name)
                log.append("default instruments")
            elif ev.key in LAYER_KEYS[: len(eng.layers)]:
                layer = eng.layers[LAYER_KEYS.index(ev.key)]
                if shift:
                    names = list(layer.VARIANTS)
                    nxt = names[(names.index(layer.variant) + 1) % len(names)]
                    eng.set_variant(layer.name, nxt)
                    log.append(f"{layer.name} -> {nxt}")
                elif ctrl:
                    if solo and solo[0] == layer.name:  # unsolo: restore previous toggles
                        for x in eng.layers:
                            x.enabled = solo[1].get(x.name, True)
                        solo = None
                        log.append("solo off")
                    else:
                        before = solo[1] if solo else {x.name: x.enabled for x in eng.layers}
                        for x in eng.layers:
                            x.enabled = x is layer
                            if x is not layer:
                                x.release_all()
                        solo = (layer.name, before)
                        log.append(f"solo {layer.name}")
                        if eng.target[layer.param] < layer.threshold + 0.15:
                            log.append(f"  (raise {layer.param} to hear it)")
                else:
                    layer.enabled = not layer.enabled
                    if not layer.enabled:
                        layer.release_all()
            elif ev.key == pygame.K_m and shift:
                meter_input = ""
            elif ev.key == pygame.K_m:
                cur = eng.state()["next_meter"] or eng.meter
                nxt = METERS[(METERS.index(cur) + 1) % len(METERS)] if cur in METERS else METERS[0]
                eng.set_meter(nxt)
                log.append(f"meter -> {nxt} (next bar)")
            elif ev.key == pygame.K_v:
                eng.reroll_variants()
                log.append("variants re-rolled")
            elif ev.key == pygame.K_n:
                player.stop(fade_ms=4000)
                chan = 1 - chan
                player = start(reseed(eng))
                solo = None
            elif ev.key == pygame.K_g:
                seed_input = ""
            elif ev.key == pygame.K_c:
                ok = copy_text(str(eng.seed))
                log.append(f"copied seed {eng.seed}" if ok else "no clipboard tool found")
            elif ev.key == pygame.K_s:
                os.makedirs(CONFIG_DIR, exist_ok=True)
                path = os.path.join(CONFIG_DIR, f"drift-{eng.seed}-{time.strftime('%Y%m%d-%H%M%S')}.json")
                save_config(eng, path)
                copy_text(json.dumps(eng.config(), indent=2))
                log.append(f"saved configs/{os.path.basename(path)}")
                log.append("  (JSON copied to clipboard)")

        # ------------------------------------------------------------------ draw
        eng = player.engine
        st = eng.state()
        screen.fill(BG)
        screen.blit(big.render(st["chord"], True, FG), (24, 18))
        if meter_input is not None:
            seed_line, col = f"meter: {meter_input}_   (e.g. 5/4, 3+3+2/8; Enter, Esc)", WARN
        elif seed_input is not None:
            seed_line, col = f"seed: {seed_input}_   (Enter go, Ctrl+V paste, Esc cancel)", WARN
        else:
            seed_line, col = f"seed {eng.seed}", ACC
        screen.blit(font.render(seed_line, True, col), (300, 28))
        meter = st["meter"] + (f" -> {st['next_meter']}" if st["next_meter"] else "")
        screen.blit(font.render(f"{st['key']}   {meter}   bar {st['bar']}   {st['notes']} notes   "
                                f"{int(st['seconds']) // 60}:{int(st['seconds']) % 60:02}", True, DIM), (24, 56))

        y = 96
        for i, k in enumerate(PARAMS):
            cur, tgt = eng.p[k], eng.target[k]
            lo, hi = LIMITS.get(k, (0.0, 1.0))
            col = ACC if i == sel else FG
            screen.blit(font.render(f"{'>' if i == sel else ' '} {k:<10} {tgt:6.2f}", True, col), (24, y))
            pygame.draw.rect(screen, BAR, (230, y + 4, 160, 10))
            pygame.draw.rect(screen, col, (230, y + 4, int(160 * (cur - lo) / (hi - lo)), 10))
            y += 26

        y = 96
        for i, layer in enumerate(eng.layers[: len(LAYER_KEYS)]):
            key = pygame.key.name(LAYER_KEYS[i]).upper()
            col = FG if layer.enabled else DIM
            if solo and solo[0] == layer.name:
                col = WARN
            sounding = any(not n.finished for n in layer.notes)
            pygame.draw.circle(screen, ACC if sounding else BAR, (432, y + 9), 5)
            screen.blit(font.render(f"{key} {layer.name:<7}", True, col), (444, y))
            screen.blit(font.render(layer.variant, True, DIM), (564, y))
            pygame.draw.rect(screen, BAR, (690, y + 4, 150, 10))
            if layer.enabled:
                pygame.draw.rect(screen, ACC, (690, y + 4, int(150 * layer.level), 10))
            y += 26

        y = 370
        for line in list(log)[-5:]:
            screen.blit(font.render(line, True, DIM), (24, y))
            y += 20
        presets = [f"{pygame.key.name(k)}:{n}" for k, n in zip(PRESET_KEYS, PRESET_NAMES)]
        for i, line in enumerate([
            "arrows: params   Q-P: toggle layer   Ctrl+Q-P: solo   Shift+Q-P: next variant",
            "V: re-roll variants   D: default variants   N: new seed   G: go to seed",
            "M: time signature (Shift+M: type one)   C: copy seed   S: save config",
            "  ".join(presets[:5]),
            "  ".join(presets[5:]),
        ]):
            screen.blit(font.render(line, True, DIM if i < 3 else FG), (24, 482 + 22 * i))
        pygame.display.flip()
        clock.tick(30)

    player.stop(fade_ms=0)
    pygame.quit()


if __name__ == "__main__":
    main()
