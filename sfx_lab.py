"""Sound-effects lab: tune drift.sfx recipes by ear.  python sfx_lab.py [recipe]

  Left list        click a recipe (or PgUp / PgDn)
  Up / Down        select a row          Left / Right   change it (Shift: finely)
  Mouse            drag a slider, wheel nudges it, right-click resets it
  Space            play a new take       R              repeat at the rhythm (steps at walking pace)
  B                A/B: tuned vs default  M              music under it (cave, meadow, off)
  Backspace        reset the row         D              reset the whole recipe
  S                save to drift/sfx_tuning.json (a running game picks it up within a second)
  Esc              quit

The top rows are what the game asks for (surface, footwear...): they pick which
sound you hear, they are not saved. Below them are the recipe's own knobs, then the
four every sound has. Knobs greyed out don't touch the sound playing (a stone knob on
grass). Saved values are only the ones that differ from the defaults.
"""

import inspect
import sys
import time

import numpy as np
import pygame

from drift import Engine, PygamePlayer, sfx

SR = 44100
W, H = 1180, 760
LIST_W = 190
ROW = 20
TOP = 120
BG, FG, DIM, ACC, BAR, WARN, OK = (16, 18, 26), (220, 222, 230), (110, 115, 135), (130, 190, 255), (60, 66, 88), (255, 200, 120), (140, 220, 150)

#: Choices for the recipes' string parameters.
CHOICES = {"surface": sfx.SURFACES, "footwear": sfx.FOOTWEAR, "element": ("fire", "ice", "dark")}
#: Seconds between takes when repeating: how the game paces them.
RHYTHM = {"footstep": 0.433, "hoof": 0.29, "swing": 0.6}
#: How loud the game plays them (DUNGEON_SFX_VOLUME 0.9, steps at 0.9 of that).
GAME_VOLUME = 0.81
MUSIC = (None, "cave", "meadow")


class Row:
    """One line of the panel: a recipe parameter, a knob, or the repeat rate."""

    def __init__(self, kind, name, default, lo=0.0, hi=1.0, doc="", choices=None):
        self.kind, self.name, self.default, self.lo, self.hi, self.doc = kind, name, default, lo, hi, doc
        self.choices = choices

    @property
    def digits(self):
        span = self.hi - self.lo
        return 0 if span >= 100 else 2 if span >= 10 else 3 if span >= 0.2 else 4

    def fmt(self, v):
        if self.choices is not None or isinstance(v, bool):
            return str(v)
        if self.name.endswith("tone") or self.name == "high_cut":
            if v >= sfx.OFF:
                return "off"
        if self.name == "low_cut" and v <= 0:
            return "off"
        return f"{v:.{self.digits}f}"


def params_of(name):
    fn = sfx.RECIPES[name][0]
    rows = []
    for p in list(inspect.signature(fn).parameters.values())[2:]:
        if p.name == "key":
            continue
        d = p.default
        if isinstance(d, bool):
            rows.append(Row("param", p.name, d, choices=(True, False)))
        elif isinstance(d, str):
            rows.append(Row("param", p.name, d, choices=CHOICES.get(p.name, (d,))))
        elif isinstance(d, (int, float)):
            rows.append(Row("param", p.name, float(d), 0.0, 1.0))
    return rows


def rows_for(name):
    rows = params_of(name)
    rows.append(Row("rhythm", "repeat every", RHYTHM.get(name, 1.0), 0.1, 2.0, "seconds between takes with R"))
    for k, (d, lo, hi, doc) in sfx.KNOBS.get(name, {}).items():
        rows.append(Row("knob", k, d, lo, hi, doc))
    for k, (d, lo, hi, doc) in sfx.POST.items():
        rows.append(Row("knob", k, d, lo, hi, doc))
    return rows


def main():
    pygame.mixer.pre_init(SR, -16, 2, 512)
    pygame.init()
    pygame.mixer.set_num_channels(16)
    pygame.mixer.set_reserved(1)  # channel 0: the music bed
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("drift sfx lab")
    font = pygame.font.SysFont("monospace", 15)
    big = pygame.font.SysFont("monospace", 24, bold=True)
    clock = pygame.time.Clock()

    names = sorted(sfx.RECIPES)
    current = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in sfx.RECIPES else "footstep"
    state = dict(rows=rows_for(current), values={}, sel=0, scroll=0, repeat=False, next_at=0.0,
                 ab_default=False, music=0, player=None, wave=None, dirty=False, msg="", msg_t=0.0,
                 drag=None, quit_armed=False)
    rng = np.random.default_rng()

    def select(name):
        nonlocal current
        current = name
        state.update(rows=rows_for(name), values={}, sel=0, scroll=0, repeat=False, wave=None)

    def value(row):
        if row.kind == "knob":
            return sfx.TUNING.get(current, {}).get(row.name, row.default)
        return state["values"].get(row.name, row.default)

    def set_value(row, v):
        if row.choices is None:
            v = round(float(min(row.hi, max(row.lo, v))), row.digits)
            if row.digits == 0:
                v = float(int(v))
        if row.kind == "knob":
            t = sfx.TUNING.setdefault(current, {})
            if v == row.default:
                t.pop(row.name, None)
            else:
                t[row.name] = v
            state["dirty"] = True
        else:
            state["values"][row.name] = v

    def applies(row):
        """Does this knob change what is playing? Knobs say so in their description,
        "stone: ..." or "plate: ...", matched against the parameters chosen above."""
        if row.kind != "knob" or ":" not in row.doc:
            return True
        what = row.doc.split(":")[0]
        chosen = {str(value(r)) for r in state["rows"] if r.kind == "param"}
        if what == "hard ground":
            return bool(chosen & {"stone", "wood"})
        return what in chosen

    def say(text):
        state["msg"], state["msg_t"] = text, time.monotonic()

    def play():
        kw = {r.name: value(r) for r in state["rows"] if r.kind == "param"}
        saved = sfx.TUNING.get(current)
        if state["ab_default"]:
            sfx.TUNING.pop(current, None)
        try:
            x = sfx.render(current, SR, np.random.default_rng(rng.integers(1 << 31)), **kw)
        except Exception as e:  # a knob pushed somewhere the recipe can't go
            say(f"render failed: {e}")
            return
        finally:
            if saved is not None:
                sfx.TUNING[current] = saved
        state["wave"] = x
        pcm = (np.clip(x * GAME_VOLUME, -1, 1) * 32767).astype(np.int16)
        pygame.mixer.Sound(buffer=np.ascontiguousarray(np.repeat(pcm[:, None], 2, axis=1)).tobytes()).play()

    def nudge(row, direction, fine=False):
        if row.choices is not None:
            i = row.choices.index(value(row)) if value(row) in row.choices else 0
            set_value(row, row.choices[(i + direction) % len(row.choices)])
        else:
            set_value(row, value(row) + direction * (row.hi - row.lo) * (0.005 if fine else 0.02))

    def music(i):
        if state["player"] is not None:
            state["player"].stop(fade_ms=300)
            state["player"] = None
        state["music"] = i
        if MUSIC[i]:
            p = PygamePlayer(Engine(MUSIC[i], sr=SR), chunk=0.25, ahead=2, channel=0).start()
            p.set_volume(0.6)  # the game's DUNGEON_MUSIC_VOLUME
            state["player"] = p

    def row_at(pos):
        x, y = pos
        if x < LIST_W:
            i = (y - TOP) // ROW
            return ("list", i) if 0 <= i < len(names) else None
        i = (y - TOP) // ROW + state["scroll"]
        return ("row", i) if y >= TOP and 0 <= i < len(state["rows"]) else None

    SLIDER_X, SLIDER_W = LIST_W + 210, 260

    def slider_set(row, x):
        if row.choices is None:
            u = (x - SLIDER_X) / SLIDER_W
            set_value(row, row.lo + max(0.0, min(1.0, u)) * (row.hi - row.lo))

    play()
    running = True
    while running:
        now = time.monotonic()
        rows = state["rows"]
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                shift = ev.mod & pygame.KMOD_SHIFT
                row = rows[state["sel"]]
                if ev.key == pygame.K_ESCAPE:
                    if state["dirty"] and not state["quit_armed"]:
                        state["quit_armed"] = True
                        say("unsaved changes: S to save, Esc again to quit without")
                    else:
                        running = False
                    continue
                state["quit_armed"] = False
                if ev.key == pygame.K_UP:
                    state["sel"] = (state["sel"] - 1) % len(rows)
                elif ev.key == pygame.K_DOWN:
                    state["sel"] = (state["sel"] + 1) % len(rows)
                elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    nudge(row, 1 if ev.key == pygame.K_RIGHT else -1, bool(shift))
                    if not state["repeat"]:
                        play()
                elif ev.key == pygame.K_SPACE:
                    play()
                elif ev.key == pygame.K_r:
                    state["repeat"] = not state["repeat"]
                    state["next_at"] = now
                elif ev.key == pygame.K_b:
                    state["ab_default"] = not state["ab_default"]
                    if not state["repeat"]:
                        play()
                elif ev.key == pygame.K_m:
                    music((state["music"] + 1) % len(MUSIC))
                elif ev.key == pygame.K_BACKSPACE:
                    set_value(row, row.default)
                    if not state["repeat"]:
                        play()
                elif ev.key == pygame.K_d:
                    if sfx.TUNING.pop(current, None):
                        state["dirty"] = True
                    say(f"{current}: back to defaults")
                elif ev.key == pygame.K_s:
                    path = sfx.save_tuning()
                    state["dirty"] = False
                    say(f"saved {path}")
                elif ev.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    i = names.index(current) + (1 if ev.key == pygame.K_PAGEDOWN else -1)
                    select(names[i % len(names)])
                    play()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
                hit = row_at(ev.pos)
                if hit and hit[0] == "list":
                    select(names[hit[1]])
                    play()
                elif hit:
                    state["sel"] = hit[1]
                    row = rows[hit[1]]
                    if ev.button == 3:
                        set_value(row, row.default)
                        if not state["repeat"]:
                            play()
                    elif row.choices is not None:
                        nudge(row, 1)
                        if not state["repeat"]:
                            play()
                    elif SLIDER_X - 6 <= ev.pos[0] <= SLIDER_X + SLIDER_W + 6:
                        state["drag"] = row
                        slider_set(row, ev.pos[0])
            elif ev.type == pygame.MOUSEMOTION and state["drag"] is not None:
                slider_set(state["drag"], ev.pos[0])
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and state["drag"] is not None:
                state["drag"] = None
                if not state["repeat"]:
                    play()
            elif ev.type == pygame.MOUSEWHEEL:
                hit = row_at(pygame.mouse.get_pos())
                if hit and hit[0] == "row":
                    state["sel"] = hit[1]
                    nudge(rows[hit[1]], 1 if ev.y > 0 else -1, fine=True)
                    if not state["repeat"]:
                        play()

        if state["repeat"] and now >= state["next_at"]:
            period = next(value(r) for r in rows if r.kind == "rhythm")
            state["next_at"] = max(now, state["next_at"] + period)
            play()

        # keep the selection on screen
        visible = (H - TOP - 40) // ROW
        if state["sel"] < state["scroll"]:
            state["scroll"] = state["sel"]
        elif state["sel"] >= state["scroll"] + visible:
            state["scroll"] = state["sel"] - visible + 1

        # ---------------------------------------------------------------- draw
        screen.fill(BG)
        screen.blit(big.render(current, True, ACC), (24, 18))
        flags = [("REPEAT", state["repeat"]), ("DEFAULTS (B)" if state["ab_default"] else "TUNED", True),
                 (f"music: {MUSIC[state['music']] or 'off'}", bool(state["music"])),
                 ("unsaved" if state["dirty"] else "saved", True)]
        x = 24
        for text, on in flags:
            col = WARN if text in ("DEFAULTS (B)", "unsaved") else (OK if on else DIM)
            img = font.render(text, True, col)
            screen.blit(img, (x, 56))
            x += img.get_width() + 24
        doc = " ".join((inspect.getdoc(sfx.RECIPES[current][0]) or "").split())
        room = (W - 420 - 24) // font.size("x")[0]
        screen.blit(font.render(doc[:room - 1] + "…" if len(doc) > room else doc, True, DIM), (24, 80))

        # waveform of the last take
        wave = state["wave"]
        box = pygame.Rect(W - 400, 14, 380, 90)
        pygame.draw.rect(screen, (24, 27, 38), box)
        if wave is not None and len(wave):
            n = box.w
            seg = np.array_split(np.abs(wave), n)
            peak = max(1e-9, np.abs(wave).max())
            for i, s in enumerate(seg):
                h = int((s.max() if len(s) else 0) / peak * (box.h // 2 - 2))
                pygame.draw.line(screen, ACC, (box.x + i, box.centery - h), (box.x + i, box.centery + h))
            screen.blit(font.render(f"{len(wave) / SR * 1000:.0f} ms", True, DIM), (box.right - 70, box.bottom + 2))

        # recipe list
        for i, n in enumerate(names):
            y = TOP + i * ROW
            if y > H - 20:
                break
            tuned = bool(sfx.TUNING.get(n))
            col = ACC if n == current else (WARN if tuned else FG)
            screen.blit(font.render(("* " if tuned else "  ") + n, True, col), (8, y))

        # rows
        last_kind = None
        for i in range(state["scroll"], min(len(rows), state["scroll"] + visible)):
            row = rows[i]
            y = TOP + (i - state["scroll"]) * ROW
            if row.kind != last_kind and last_kind is not None:
                pygame.draw.line(screen, BAR, (LIST_W + 10, y - 2), (W - 20, y - 2))
            last_kind = row.kind
            v = value(row)
            changed = v != row.default
            sel = i == state["sel"]
            live = applies(row)
            name_col = ACC if sel else (WARN if changed and row.kind == "knob" else FG if live else DIM)
            screen.blit(font.render(("> " if sel else "  ") + row.name, True, name_col), (LIST_W + 10, y))
            if row.choices is not None:
                screen.blit(font.render(f"< {row.fmt(v)} >", True, FG), (SLIDER_X, y))
            else:
                r = pygame.Rect(SLIDER_X, y + 6, SLIDER_W, 6)
                pygame.draw.rect(screen, BAR, r)
                u = (v - row.lo) / (row.hi - row.lo) if row.hi > row.lo else 0
                d = (row.default - row.lo) / (row.hi - row.lo) if row.hi > row.lo else 0
                pygame.draw.line(screen, DIM, (SLIDER_X + d * SLIDER_W, y + 2), (SLIDER_X + d * SLIDER_W, y + 15))
                pygame.draw.rect(screen, WARN if changed else ACC, (SLIDER_X + u * SLIDER_W - 4, y + 3, 8, 12))
                screen.blit(font.render(row.fmt(v), True, FG), (SLIDER_X + SLIDER_W + 14, y))
            fit = (W - SLIDER_X - SLIDER_W - 100) // font.size("x")[0]
            screen.blit(font.render(row.doc[:fit], True, DIM if live else BAR), (SLIDER_X + SLIDER_W + 90, y))

        if state["msg"] and now - state["msg_t"] < 4:
            screen.blit(font.render(state["msg"], True, WARN), (LIST_W + 10, H - 26))
        else:
            screen.blit(font.render("Space play  R repeat  B A/B  M music  S save  Backspace/right-click reset  D reset all",
                                    True, DIM), (LIST_W + 10, H - 26))
        pygame.display.flip()
        clock.tick(60)

    if state["player"] is not None:
        state["player"].stop(fade_ms=0)
    pygame.quit()


if __name__ == "__main__":
    main()
