# Vehicle make/model/color enrichment

How to enrich `passes` rows in `camwatch.db` with vehicle identification,
running from a Claude Code session on the Ubuntu 3060 Ti desktop.

## What we're doing

For each high-confidence pass, look at its `.jpg` thumbnail and write
back a guess at vehicle make, model, color bucket, and confidence. The
existing `passes` table already has the columns:

- `vehicle_make` TEXT
- `vehicle_model` TEXT
- `vehicle_year_range` TEXT
- `vehicle_color` TEXT (categorical: light, grey, dark, red, blue, green, brown, yellow)
- `vehicle_confidence` TEXT (high, medium, low)
- `vehicle_enriched_at` TEXT (ISO timestamp, NULL = unenriched)

A pass is "enrichable" iff its `.jpg` thumbnail exists on disk.

**Note on thumbnail naming (Mac → Ubuntu migration, 2026-05-12).** On the
old Mac host the sub-stream was processed for detection and a separate
`thumb_upgrade` worker re-extracted an HD thumbnail from the main stream,
saved as `<clip>_big.jpg`. On this Ubuntu/3060 Ti host the GPU processes
the main stream directly, so the regular `<clip>.jpg` is already HD
(~800×300) and no `_big.jpg` is produced. The enrichment pipeline now
reads the regular `.jpg`. The `thumb_upgrade_status` column is a
historical artifact and stays NULL for post-migration passes.

## Why we run via sub-agents

The naive approach — one long Claude Code session reads each thumbnail
via `Read`, identifies the vehicle, writes back via `Bash`+sqlite — does
not scale. Over 290 enrichments in a single session:

- 868 tool calls (368 Reads, 283 Bash)
- 292 thumbnail images read into the main conversation context
- Sum of `cache_read_input_tokens` across all turns: **659 M**
- Max single-turn context size: **857 K tokens** (approaching the 1 M ceiling)
- Each successive turn replayed every prior thumbnail because tool results
  stay in the conversation for the rest of the session
- Approx cost at Opus rates: $200 to $350 per 290-row session

The root cause is structural: every image lands in the main context and
never leaves, so the prompt grows linearly with passes processed and the
per-turn cache-read fee compounds.

The sub-agent pattern below has finished a 189-row run for **~$5–6 of
Opus tokens** end-to-end — roughly a 40× saving — because the parent
session never sees the images.

## The pattern

The main session never reads a thumbnail itself. It delegates each batch
of thumbnails to an Opus sub-agent via the `Agent` tool with
`model: "opus"`. The sub-agent reads the image, returns one line of JSON
per pass, then its entire context (image + tool schemas + reasoning) is
discarded. Only the JSON result lands in the parent context.

Concretely:

1. Main session queries the DB for pending passes that have a `.jpg`
   on disk. No image data leaves the DB query.
2. Main session groups them into batches of **10–12 passes per
   sub-agent call**. Per-call overhead is roughly 3–5 K tokens regardless
   of payload, so batching amortizes it. One image per sub-agent is wasteful.
3. Main session fires **4–6 sub-agents in parallel** by emitting multiple
   `Agent` tool calls in a single message. 189 passes finishes in three
   wall-clock minutes that way.
4. Each sub-agent returns JSON like:
   ```
   [
     {"pass_id": 2706, "make": "Honda", "model": "Civic",
      "year_range": "2016-2021", "color": "dark", "confidence": "medium"},
     {"pass_id": 2707, "make": null, "model": null,
      "year_range": null, "color": null, "confidence": "low",
      "note": "false positive: stroller"}
   ]
   ```
5. Main session parses each result and pipes it to
   `scripts/enrich_apply.py` which `UPDATE`s the rows and stamps
   `vehicle_enriched_at = now()`.

## Model choice: use Opus, not Haiku or Sonnet

A blind bench on 10 hard known samples (mix of sedans, EVs, SUVs from
10 different makes), scored against an earlier Opus pass treated as
ground truth, settled this:

| Model  | make+model correct |
|--------|-------------------:|
| Opus   | 10/10              |
| Sonnet | 9/10               |
| Haiku  | 0/10               |

Haiku does not look at the image carefully enough — it picks one
plausible-looking make/model from training-data priors and stamps every
dark SUV with it. Even when explicitly told "do not default," it just
defaulted to a different car. Unusable here.

Sonnet is a real option (5× cheaper than Opus per token) and missed only
on a Toyota Avalon → Lexus ES confusion. For most use cases Sonnet is
fine; we run Opus because accuracy matters more than the few-dollar
delta for a ~200-row daily batch.

## Sub-agent prompt template (silhouette-first)

Pass this shape to each first-pass sub-agent. The key wording is the
**silhouette-first** guidance: badges are usually not legible at 800px
side-on, and the agent will bail on every dark SUV otherwise.

```
You will be given paths to N vehicle thumbnail images, one per pass.
For each image, Read the image, identify the vehicle, and return one
JSON object per pass.

Rules:
- color MUST be one of: light, grey, dark, red, blue, green, brown, yellow
- confidence MUST be one of: high, medium, low
- If the image is not a vehicle (stroller, person, animal, delivery van
  with company markings), set make/model/year_range/color to null and
  confidence to "low", and include a "note" describing what you saw.
- year_range optional; "YYYY-YYYY" or "YYYY+" or null.
- These are side-profile shots from a fixed driveway camera. Badges are
  usually NOT legible at 800px wide — this is expected, not a failure.
  Identification comes from body silhouette, roofline shape, taillight
  signature, headlight cluster, grille/front-fascia outline, wheelbase
  length, fender flares, wheel design, and roof rail style. Commit on
  those. Return null make/model ONLY if the image is too blurred or
  occluded to extract any silhouette cues — not because "the badge isn't
  visible."
- Output ONLY a JSON array. No prose, no markdown. First char `[`,
  last char `]`.

Input passes:
- pass_id=2706  image=/abs/path/.../cal_<ts>_id<n>_<dir>.jpg
- pass_id=2707  image=...
...

Output schema for each entry:
{"pass_id": int, "make": str|null, "model": str|null,
 "year_range": str|null, "color": str|null,
 "confidence": "high"|"medium"|"low", "note": str|null}
```

## Two-pass: rescue low-confidence rows

After the first pass, ~10–25% of rows typically land as `confidence='low'`
or `make IS NULL`. Most are recoverable: the first-pass prompt is tuned
for throughput and the agent tends to bail when distinguishing features
are close (e.g. two large luxury SUVs from the same era).

Run a **rescue pass** that re-runs only those rows with a sharper prompt:

- Same silhouette-first framing.
- Smaller batches (5 passes per sub-agent) — rescue agents reason more
  and may also use web search.
- Explicitly enables `WebSearch` and `WebFetch`: the agent can shortlist
  2–3 candidates from the silhouette and search for distinguishing-feature
  descriptions ("Audi Q5 vs Mazda CX-9 side profile differences"). The
  tools are always available; the prompt tells the agent it's OK to use
  them.
- Pass the **first-pass guess** to the rescue agent as `prior_guess`
  context — useful as a hypothesis to confirm or overturn, but call it
  out as untrusted.
- Skip night-time rows. Motion blur from the low-light shutter is the
  real failure mode there; a better prompt cannot recover it. In
  practice we filter to rows captured before 19:00 local.

Rescue rescued **20/20 daytime+evening low-confidence rows** in the last
run for ~$1 of additional tokens, at ~15s per 5-pass batch. Worth doing
on every batch.

### Rescue prompt template

```
You are doing a second-pass rescue of vehicle thumbnails that the first
pass flagged "low confidence" or "could not identify."

Guide:
- Side-profile shots from a fixed driveway camera. Badges usually NOT
  legible — that is expected.
- Identify from silhouette, roofline, taillight signature, headlight
  cluster, grille outline, wheelbase, fender flares, wheel design, roof
  rails. Commit on those. Medium confidence on silhouette alone is fine.
  Return null make/model ONLY if the image is too blurred/occluded to
  extract silhouette cues.
- You have WebSearch + WebFetch. If you narrow to 2–3 candidates from the
  silhouette but cannot decide, search for distinguishing-feature
  descriptions. Use search judiciously, not by default.
- A "prior_guess" is provided. Treat it as a low-confidence hypothesis to
  confirm or overturn, not as truth.

Output: ONE JSON array. First char `[`, last char `]`. No prose outside.
Schema: same as first pass.

Input passes:
- pass_id=2933  image=/abs/path  prior_guess="Audi Q5 (low)"
...
```

## DB application helper

`scripts/enrich_apply.py` reads a JSON array on stdin and applies it:

```bash
echo '[{"pass_id": 2990, "make": "Audi", "model": "Q5", ...}]' \
  | python3 scripts/enrich_apply.py
# updated 1/1 rows
```

It validates `color` against the enum (drops invalid values to null) and
coerces `confidence` to `low` if outside the enum. Use `--dry-run` to
inspect without writing.

## Failure modes that look like bugs

- **Same-make stamping (Haiku, also early Opus).** If the first-pass
  output has the same `(make, model)` for >70% of rows, you're seeing a
  model bias. Re-run with Opus and the silhouette-first prompt.
- **"badge not visible" bail-outs.** Add the silhouette-first paragraph
  to the prompt; without it Opus also bails on roughly half the dark
  SUVs.
- **Opus confidently wrong.** Even at `medium`, the agent can pick the
  wrong make on a clear shot (a Mazda CX-9 / Audi Q5 silhouette swap was
  the worked example). Treat `medium` as "review later"; trust only
  `high` for downstream analytics.

## Steps for a fresh Claude session

1. `cd ~/git/camwatch`
2. Count pending: `vehicle_enriched_at IS NULL` rows that have a
   `.jpg` on disk
3. First pass: batches of 10–12, fire 4–6 parallel Opus sub-agents per
   round, pipe each batch result to `scripts/enrich_apply.py`
4. Identify residual `confidence='low'` OR `make IS NULL` rows captured
   before 19:00 local
5. Rescue pass: batches of 5, parallel Opus sub-agents with the rescue
   prompt; pipe to `scripts/enrich_apply.py`
6. Stop. Night-time motion-blurred rows stay as low; that's the camera,
   not the model.

Do not, in the main session, `Read` any of the `.jpg` files. Do not
pass image content directly in the main conversation. Image handling is
the sub-agent's job and dies with the sub-agent.
