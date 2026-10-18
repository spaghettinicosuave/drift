"""Named parameter sets. Any key may be omitted; `root` (0-11, C=0) pins the key,
`room` scales the reverb size (only used when an Engine is created).
Missing parameters fall back to engine.DEFAULTS (so presets without `drive` are pure ambient).
`meter` is any time signature ("4/4" if omitted; loading a preset without one keeps the
current meter). `variants` switch instruments when the preset is loaded; presets without
them keep whatever instruments are playing. A new Engine starts with DEFAULT_VARIANTS."""

# The instruments an Engine starts with, and what default_variants() returns to.
DEFAULT_VARIANTS = {"drone": "choir", "pad": "breath", "swell": "shimmer", "arp": "glass",
                    "bells": "gong", "melody": "reed", "drums": "808", "bass": "saw",
                    "stabs": "supersaw", "riser": "reverse"}

BATTLE = {"drone": "hurdy", "pad": "viols", "swell": "strings", "arp": "sequence", "bells": "gong",
          "melody": "lead", "drums": "punchy", "bass": "reese", "stabs": "brass", "riser": "roll"}

# Italo disco: four-on-the-floor, dense chorus pad, sequenced lead, cowbell offbeats, gated bass.
ITALO = {"drone": "brasspad", "pad": "italo", "swell": "rise", "arp": "sequence", "bells": "gong",
         "melody": "lead", "drums": "italo", "bass": "italo", "stabs": "italo", "riser": "sweep"}

TAVERN = {"drone": "hurdy", "pad": "viols", "swell": "flute", "arp": "lute", "bells": "dulcimer",
          "melody": "hurdy", "drums": "tavern", "bass": "folk", "stabs": "strum", "riser": "roll"}

PRESETS: dict[str, dict] = {
    # Open, major, unhurried.
    "meadow": {"tempo": 72, "energy": 0.45, "tension": 0.12, "brightness": 0.55, "space": 0.6,
               "variants": DEFAULT_VARIANTS},
    # Busier and brighter: the Spore-city feel.
    "city": {"tempo": 88, "energy": 0.75, "tension": 0.3, "brightness": 0.7, "space": 0.45},
    # Slow, darker modes, lots of air.
    "night": {"tempo": 60, "energy": 0.3, "tension": 0.55, "brightness": 0.3, "space": 0.8},
    # Sparse and murky, huge room.
    "cave": {"tempo": 52, "energy": 0.2, "tension": 0.75, "brightness": 0.2, "space": 0.95, "room": 1.4},
    # Almost nothing: drone and the odd chord.
    "still": {"tempo": 56, "energy": 0.08, "tension": 0.3, "brightness": 0.35, "space": 0.85},
    # --- with `drive`: drums, bass, stabs, risers
    # Light pulse under a meadow: exploring with somewhere to be.
    "stroll": {"tempo": 96, "energy": 0.45, "drive": 0.3, "tension": 0.15, "brightness": 0.55, "space": 0.55},
    # Boots and cats, offbeat bass, minor-ish.
    "chase": {"tempo": 118, "energy": 0.55, "drive": 0.62, "tension": 0.5, "brightness": 0.6, "space": 0.4},
    # Everything on: four-on-the-floor, rolling bass, stabs, risers.
    "battle": {"tempo": 128, "energy": 0.7, "drive": 0.95, "tension": 0.68, "brightness": 0.75, "space": 0.35,
               "variants": BATTLE},
    # --- acoustic: hurdy-gurdy, lute, fiddle, frame drum
    # A slow jig in 6/8 by the fire: hurdy-gurdy over drones, lute, frame drum. Raise drive
    # for strummed chords, folk bass and drum rolls.
    "tavern": {"tempo": 46, "meter": "6/8", "energy": 0.55, "drive": 0.35, "tension": 0.42,
               "brightness": 0.55, "space": 0.3, "room": 0.7, "variants": TAVERN},
    # Late evening waltz: recorder, soft lute, a quiet pulse on the frame drum.
    "hearth": {"tempo": 62, "meter": "3/4", "energy": 0.42, "drive": 0.2, "tension": 0.3,
               "brightness": 0.45, "space": 0.45, "room": 0.8,
               "variants": {**TAVERN, "melody": "recorder", "drone": "hurdy"}},
    # Italo disco: four-on-the-floor, supersaw pads, sequenced arp, punchy 808 + clap, driving saw bass.
    # Minor-leaning dorian (tension ~0.45) with airy brightness; raise drive to go full dancefloor.
    "italo": {"tempo": 126, "energy": 0.72, "drive": 0.85, "tension": 0.45, "brightness": 0.72, "space": 0.5,
              "variants": ITALO},
}
