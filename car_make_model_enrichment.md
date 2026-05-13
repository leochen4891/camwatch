# Vehicle make/model/color enrichment

How to enrich `passes` rows in `camwatch.db` with vehicle identification,
running from a Claude Code session on the Ubuntu 3060 Ti desktop.

## What we're doing

For each high-confidence pass, look at its `_big.jpg` thumbnail and write
back a guess at vehicle make, model, color bucket, and confidence. The
existing `passes` table already has the columns:

- `vehicle_make` TEXT
- `vehicle_model` TEXT
- `vehicle_year_range` TEXT
- `vehicle_color` TEXT (categorical: light, grey, dark, red, blue, green, brown, yellow)
- `vehicle_confidence` TEXT (high, medium, low)
- `vehicle_enriched_at` TEXT (ISO timestamp, NULL = unenriched)

A pass is "enrichable" iff its `_big.jpg` thumbnail exists on disk.
The list filter in `alert.enrich_offset_mph` controls which passes
the UI surfaces as candidates; enriching the whole table is fine.

## What we learned about the current approach

The prior approach was: one long Claude Code session reads each thumbnail
via `Read`, identifies the vehicle, writes back via `Bash`+sqlite, then
moves to the next. Over 290 enrichments in a single session:

- 868 tool calls (368 Reads, 283 Bash)
- 292 thumbnail images read into the main conversation context
- Sum of `cache_read_input_tokens` across all turns: **659 M**
- Max single-turn context size: **857 K tokens** (approaching the 1 M ceiling)
- Each successive turn replayed every prior thumbnail because tool results
  stay in the conversation for the rest of the session
- Approx cost at Opus rates: $200 to $350 per 290-row session
  (about $1 per enrichment), or equivalent allowance on the Max plan

The root cause is structural: every image lands in the main context and
never leaves, so the prompt grows linearly with passes processed and the
per-turn cache-read fee compounds.

## The pattern we're going to use

Run on Claude Code on the Max plan. The main session never reads a
thumbnail itself. It delegates each batch of thumbnails to a Haiku
sub-agent via the `Agent` tool, with `model: "haiku"`. The sub-agent
reads the image, returns a single line of JSON per pass, then its entire
context (image + tool schemas + reasoning) is discarded. Only the JSON
result lands in the parent context.

Concretely:

1. Main session queries the DB for pending passes that have a `_big.jpg`
   on disk. No image data leaves the DB query.
2. Main session groups them into batches of **10 to 15 passes per
   sub-agent call**. Per-call overhead (sub-agent system prompt + tool
   schemas) is roughly 3 to 5 K tokens regardless of payload, so
   batching amortizes it. One image per sub-agent is wasteful; 15 is a
   good ceiling for Haiku.
3. Main session fires **4 to 8 sub-agents in parallel** by emitting
   multiple `Agent` tool calls in a single message. 290 passes finishes
   in a couple of wall-clock minutes that way.
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
5. Main session parses each result and runs `UPDATE passes SET
   vehicle_make = ..., vehicle_enriched_at = ?` for each row.

Across 290 enrichments, the main session context grows by roughly 60 KB
of JSON lines instead of 12 MB of base64 plus 850 K-token replays.

### Sub-agent prompt template

Pass this exact shape to each sub-agent. Keep it tight; Haiku does best
with explicit schemas and no prose room.

```
You will be given paths to N vehicle thumbnail images, one per pass.
For each image, identify the vehicle and return one JSON object.

Rules:
- color MUST be one of: light, grey, dark, red, blue, green, brown, yellow
- confidence MUST be one of: high, medium, low
- If the image is not a vehicle (stroller, person, animal, occlusion),
  set make/model/year_range/color to null and confidence to low, and
  include a "note" field describing what you saw instead.
- year_range is optional; use "YYYY-YYYY" or null.
- Return ONLY a JSON array. No prose, no markdown.

Input passes:
- pass_id=2706  image=/path/to/events/pass_2706_big.jpg
- pass_id=2707  image=/path/to/events/pass_2707_big.jpg
...

Output schema for each entry:
{"pass_id": int, "make": str|null, "model": str|null,
 "year_range": str|null, "color": str|null,
 "confidence": "high"|"medium"|"low", "note": str|null}
```

## The 190 unenriched passes missing thumbnails

These are pre-cutover Mac-era passes whose `_big.jpg` did not get migrated
when the service moved to the 3060 Ti. Options, in increasing effort:

1. Leave them. Their rows stay with `vehicle_enriched_at IS NULL` and the
   UI just shows no vehicle metadata for those rows.
2. Re-extract a thumbnail from the corresponding `.mp4` clip in
   `recordings/` if it still exists and within retention. This is a
   one-time backfill script, not part of the regular enrichment loop.
3. Accept them as historical noise.

Recommend (1) for now; option (2) only if those days matter for analysis.

## Steps for the fresh Claude session

When starting the enrichment session on the Ubuntu desktop:

1. `cd ~/git/camwatch`
2. Verify DB and thumbnails layout:
   - `python -c "import sqlite3; ..."` to count `vehicle_enriched_at IS NULL`
     rows that have a `_big.jpg` on disk
3. Plan the run: how many to do, batch size, parallelism
4. Loop: for each batch of pending pass IDs, launch parallel `Agent`
   calls with `model: "haiku"`, prompt as above
5. As JSON returns, `UPDATE passes` with the results
6. Stop when no pending passes remain (or hit a self-imposed cap)

Do not, in the main session, `Read` any of the `_big.jpg` files. Do not
pass image content directly in the main conversation. Image handling is
the sub-agent's job and dies with the sub-agent.
