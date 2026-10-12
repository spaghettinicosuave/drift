# drift

Endless generative ambient music: no beginning, no end, steered by a handful of
parameters. Pure numpy; pygame is optional. In the spirit of Spore's procedural score.

## Try it

```sh
python demo.py            # needs pygame-ce with mixer
python -m drift wav out.wav --preset night --seconds 180 --variant drums=808 --variant pad=bloom
python -m drift wav out.wav --config configs/drift-4242-20261010-172855.json
python -m drift raw --preset city | pw-play --rate 44100 --channels 2 --format s16 -
```

Demo keys:

| key | does |
|---|---|
| arrows | select / change a parameter |
| `1`-`9`, `0` | glide to a preset |
| `D` | fade back to the default variants (e.g. after `tavern`) |
| `Q`-`P` | toggle a layer; `Ctrl` solos it, `Shift` crossfades to its next variant |
| `V` | re-roll all variants |
| `N` | new random seed (crossfade). Variants, toggles and parameters carry over |
| `M` | next time signature (4/4, 3/4, 6/8, 2/4, 5/4, 7/8, 5/8, 9/8, 12/8, 7/4) |
| `Shift+M` | type any time signature, e.g. `11/8` or `3+3+2/8`, `Enter` to go |
| `G` | type a seed (`Ctrl+V` pastes), `Enter` to go |
| `C` | copy the current seed |
| `S` | save everything to `configs/drift-<seed>-<time>.json` and copy the JSON |

The dot next to each layer lights while it is sounding. `python demo.py configs/x.json`
opens a saved config.

Presets: `meadow city night cave still` are pure ambient; `stroll chase battle` add
drums and synths through `drive`. Switching presets glides, so `meadow` -> `battle`
builds up over a few seconds instead of cutting.

`tavern` (a slow 6/8 jig) and `hearth` (a 3/4 waltz) are folk: hurdy-gurdy, lute,
hammered dulcimer, fiddle or recorder, frame drum and tambourine. Raise `drive` and
strummed chords, an oom-pah bass and drum rolls join in. These presets crossfade the
instruments as well as the parameters. Other presets keep whatever instruments are
playing, so press `D` to fade back to the synths.

## Use it

```python
from drift import Engine

eng = Engine("meadow", seed=42)   # same seed -> same music
audio = eng.render(44100)         # (n, 2) float32, any n, any time
eng.set(energy=0.7, tension=0.4)  # glides over ~6 s, never jumps
eng.load("night")                 # glide to another preset
eng.set_meter("7/8")              # any time signature; takes effect at the next bar line
eng.state()                       # key, chord, bar, layer levels...
eng.on_event(lambda kind, info: ...)  # 'chord', 'bar', 'drift' (audio thread!)
```

In pygame:

```python
from drift import PygamePlayer
player = PygamePlayer.create("meadow").start()  # background thread, reserved channel
player.engine.set(tension=0.8)                   # e.g. when a fight starts
player.stop(fade_ms=3000)
```

`Engine.render` is all a backend needs, so the same engine works with sounddevice,
a game engine's audio callback, or offline rendering.

## Sound effects

`drift.sfx` synthesizes game sound effects the same way: no samples, and every call is
a slightly different take, so repeated sounds never machine-gun.

```python
from drift import sfx
x = sfx.render("hit_bone")                    # mono float32 at the recipe's calibrated level
x = sfx.render("footstep", surface="stone", footwear="plate", armor=0.5)
x = sfx.render("heal", key=(62, sfx.MAJOR))   # musical ones can follow a key
```

Recipes: `swing` (weight), `hit_flesh`, `hit_metal`, `hit_wood`, `hit_bone`,
`bones_collapse`, `block`, `hurt`, `body_fall`, `cast` / `impact` (fire, ice, dark),
`heal`, `mana`, `level_up`, `coin`, `pickup`, `footstep` (surface: grass, dirt, stone, wood,
water; footwear: leather, bare, plate; armor 0..1), `hoof` (one trot beat), `snort`, `whinny`,
`jump`, `land`, `roll`, `door_open`, `door_close`, `chest_open`, `chest_close`,
`ui_click`, `ui_open`. Add your own with `@sfx.recipe("name", level=-20)`.

Each is scaled so its loudest 50 ms sits at a set level, balanced against the others
and against the music (about -18 dB RMS). In pygame, `SfxPlayer` adds the rest:

```python
player = PygamePlayer.create("cave").start()        # music on channel 0
fx = sfx.SfxPlayer(music=player)
fx.warm("swing", "hit_flesh")                       # render variations in the background
fx.play("hit_flesh", pan=0.4, volume=0.8)           # pan -1..1
fx.update()                                         # once a frame
```

It keeps a pool of variations per sound and never plays the same take twice in a
row. Renders happen on a worker thread. Heals, level-ups and coins play in the
music's current key. Loud hits dip the music for half a second. `PygamePlayer`
plays on channel 0 and reserves only that one, leaving the rest to effects.

### Tuning by ear

`python sfx_lab.py [recipe]` opens a lab: pick a recipe and its parameters (surface,
footwear...), drag knobs, and hear a fresh take each time, or repeat at walking pace
(R), over the music (M), against the defaults (B). Every recipe has gain, pitch, low
cut and high cut, and some have knobs for their layers (`sfx.knobs()`). S saves the
changes to `drift/sfx_tuning.json`; `drift.server` reloads it within a second, so a
running game picks it up.

### In a game: run the audio in its own process

Inside a game's process the music thread shares Python with the game loop, so a
level load or a long frame starves it and the music hitches. `AudioServer` runs the
music and effects in a child process (`python -m drift.server`) that keeps a second
of music rendered ahead; the game only sends small messages and never blocks:

```python
from drift.server import AudioServer
audio = AudioServer("cave", music_volume=0.6)
audio.play("hit_bone", pan=0.3)
audio.set(drive=0.7, tension=0.6)      # glided as usual
audio.load("meadow")                   # presets, variants, meters
audio.stats()                          # underruns, render times, key/chord
```

The child exits when the game does. `DRIFT_DISKAUDIOFILE=out.raw` records exactly
what it plays, for checking a session for dropouts.

## Time signatures

Any preset can play in any meter: `eng.set_meter("7/8")`, `M` in the demo, or
`--meter 7/8` on the command line. Write `N/4`, `N/8` or `N/2`. Eighth-note meters group
into dotted beats when they can (6/8, 9/8, 12/8), otherwise into 2s with a 3 at the end
(7/8 = 2+2+3). Spell the grouping out to change it: `3+3+2/8`, `3+2+2/8`. Drums, bass,
strums, the hurdy-gurdy buzz and the accents follow the beat groups, so 7/8 battle music
kicks on 2+2+3. The change waits for the next bar line.

`tempo` counts beats: quarter notes in 4/4, dotted quarters in 6/8 and 12/8, and quarter
notes in uneven meters like 7/8. Presets that name a `meter` (`tavern`, `hearth`) switch
to it; the others keep whatever meter is playing.

## Variants

Each layer has four or more variants that change both its sound and how it plays. Variants
are independent of the seed: an Engine starts with each layer's first variant, and
you pick others or roll dice with `reroll_variants()`. That way you can keep a sound
and try it with different seeds.

| layer | variants |
|---|---|
| drone | `warm`, `organ` (root + fifth), `choir` (formants), `sub` (deep, pulsing), `hurdy` (hurdy-gurdy drone strings with buzzing trompette) |
| pad | `strings`, `glass` (+ octave shimmer), `breath` (slow, airy), `viols` (bowed consort), `bloom` (filter opens as it swells) |
| swell | `flute`, `strings`, `shimmer` (octaves, flutter), `rise` (brassy, opens up) |
| arp | `pluck`, `marimba`, `glass`, `lute` (plucked string), `sequence` (gated saw, up/down in order) |
| bells | `chime`, `glock` (little runs), `gong` (rare, low), `dulcimer` (hammered, double-strung), `kalimba` (runs) |
| melody | `flute`, `whistle`, `reed`, `hurdy` (hurdy-gurdy chanter), `fiddle`, `recorder`, `lead` (synth, busier rhythms) |
| drums | `punchy`, `808`, `dusty` (swung, dark), `industrial` (distorted), `tavern` (frame drum, tambourine, claps) |
| bass | `saw`, `sub`, `acid` (squelchy, wanders), `folk` (plucked oom-pah), `reese` (wide, long notes) |
| stabs | `supersaw`, `organ` (house), `pluck`, `strum` (strummed lute chords), `brass` |
| riser | `sweep`, `noise`, `tonal`, `reverse` (reversed cymbal), `roll` (drum roll) |

Variants of the same layer are calibrated to the same loudness, so swapping one
doesn't change the mix balance.

Changing a variant crossfades (6 s by default). For a while the old instrument keeps
playing along with the new one, following the same chords, so a flute melody can
hand its phrase over to a fiddle mid-tune. Drones and pads start the new sound
immediately instead of waiting for the next chord. Pass `fade=` in seconds to change
the length; `fade=0` switches at once and lets sounding notes ring out.

```python
eng = Engine("battle", seed=7, variants={"drums": "808"})  # the rest use their defaults
eng.set_variant("bass", "reese")          # crossfades over 6 s
eng.set_variant("drone", "hurdy", fade=20)  # a slow morph, e.g. walking into a tavern
eng.load("tavern", fade=12)               # preset with instruments: params glide, instruments fade
eng.reroll_variants(keep=["drums"])       # random instruments, same music
eng.variants()                            # {'drone': 'sub', 'pad': 'glass', ...}
eng.default_variants()                    # each layer's first variant
```

## Saving a sound

```python
from drift import save_config, load_config
save_config(eng, "boss.json")             # seed, parameters, meter, variants, muted layers
eng = load_config("boss.json")            # or Engine.from_config(dict)
```

A config fully determines the music: the same config always plays the same piece,
from its beginning. It records where the sliders *are*, not how you moved them, so
it won't resume mid-session exactly where you saved.

## Parameters

| name | range | effect |
|---|---|---|
| `energy` | 0-1 | how many layers play and how busy they are |
| `drive` | 0-1 | drums, bass, stabs, risers and sidechain pumping on top; 0 = pure ambient |
| `tension` | 0-1 | mode: lydian, ionian, mixolydian, dorian, aeolian, phrygian; also more 7ths/sus |
| `brightness` | 0-1 | timbre, dark to airy |
| `space` | 0-1 | reverb length/mix, echo feedback |
| `tempo` | BPM | clock (16th-note steps; eighths in 6/8) |
| `volume` | 0-1 | master |

## How it works

- **Harmony** (`theory.py`) wanders: chords move by 4ths/5ths/3rds with a weak pull
  home, the mode follows `tension`, and every so often the key shifts by a fifth.
- **Layers** (`voices.py`) listen to the harmony and clock: `Drone`, `Pad`, `Swell`,
  `Arp` (Euclidean rhythms), `Bells`, `Melody` (motifs that return varied). Each fades
  in and out by itself (a slow random walk on top of `energy`), so the arrangement
  keeps shifting without a script.
- **Drive layers** follow `drive` instead: `Drums` (a sparse pulse, then boots-and-cats,
  then four-on-the-floor with ghost notes and fills; the kick is tuned to the key),
  `Bass` (distorted saw, offbeat then rolling 16ths), `Stabs` (supersaw chords),
  `Riser` (a two-bar sweep into the next 8-bar phrase, landing on a crash). Pads and
  bass duck under the kick in proportion to `drive`.
- **Sound** (`dsp.py`): band-limited wavetables (built per note by inverse FFT),
  additive inharmonic bells, drums and sweeps synthesised once into buffers, ping-pong echo, an 8-line FDN reverb, a slow auto-gain
  and a soft clip.

## Custom layers

```python
from drift import Engine, Layer, default_layers
from drift import dsp

class Heartbeat(Layer):
    name = "heartbeat"
    threshold = 0.5           # appears when energy rises above ~0.5
    def on_step(self, ctx, step, offset):
        if self.active and ctx.pos in (0, 3):    # ctx.pos: step within the bar
            midi = ctx.harmony.near(ctx.harmony.degree, 33)
            self.play(ctx, midi, 0.4 * self.level, dsp.Envelope(0.005, 0.25, 0.0, 0.2),
                      dsp.wave(3, 0.2), offset)

eng = Engine("night", layers=default_layers() + [Heartbeat()])
```

Give a custom layer its own `VARIANTS = {"name": {...}, ...}` and read settings from
`self.v`. A variant's `gain`, `reverb`, `echo`, `duck` and `saturate` are applied
automatically.

Class attributes tune a layer: `param` (which parameter brings it in), `threshold`,
`wander_depth`, `gain`, `reverb`/`echo` sends, `duck` (sidechain) and `saturate`.
Call `ctx.kick(offset)` from a layer to trigger the sidechain.
`ctx.pos` is the step in the bar (16ths), `ctx.bar_steps` the bar length, `ctx.beats` and
`ctx.beat_lens` the beat starts and lengths (7/8: `(0, 4, 8)`, `(4, 4, 6)`), and
`ctx.beat_at(ctx.pos)` gives (beat, step within the beat, beat length).
`self.ks(ctx, midi, decay=..., bright=...)` returns a cached plucked-string sample
for `self.play_sample(sample, amp, offset)`.

Hooks: `on_bar(ctx, bar, offset)`, `on_chord(ctx, offset)`, `on_step(ctx, step, offset)`.
`offset` is the sample position inside the current block; pass it to `play()`.

## Cost

Calm presets render at 15-25x realtime on one core, `tavern` at about 9x and `battle`
at about 7x (~15% of a core).
`PygamePlayer` renders on one thread and feeds the mixer from another, keeping
`ahead` chunks ready, so a slow block (a riser is built whole, ~0.5 s once per
8 bars) eats into that reserve instead of being heard. In a game, prefer `AudioServer`.

## License

MIT, see [LICENSE](LICENSE). Use it in anything, commercial included; keep the
copyright notice with copies of the code.

Music you render with drift is yours: no attribution, royalties or license notice
needed for the audio itself. Every sound is synthesized in code, with no samples or
third-party assets.
