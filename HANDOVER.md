# HANDOVER — Camera provenance + registry adoption (ADR-013/015)

> For the implementing session: read `camwatch-system/SYSTEM.md` first, then
> ADR-013/014/015 in `camwatch-system/DECISIONS.md`. This file is the work
> order; the ADRs are the why. **Delete this file in the final commit.**
>
> This repo is public — camera placement details, IPs, and credentials live
> only in the private registry repo (`camwatch-cameras`) and must never
> appear here.

## Work items

1. **Adopt the camera registry as a dependency** (ADR-015). Add the private
   `camwatch-cameras` package (uv git/path dependency) and replace this
   repo's own camera facts with loader calls:
   - homography: `load_camera(main_id).homography()` (and its K/D fields)
     instead of `config/homography.yaml`;
   - cadence: `load_camera(main_id).cadence_fps()` instead of any local
     constant;
   - RTSP URL: `profile.rtsp_url(user, password)` with creds from env.
   The local `config/homography.yaml`-family files and the calibration
   scripts (`mark_points.py`, `build_homography_from_marks.py`,
   `render_homography_overlay.py`, `fit_distortion_from_scene.py`) are
   superseded by the registry repo's self-contained tooling — mark or remove
   them in a follow-up once the loader path is proven.
2. **Main-camera election** (ADR-013): a `main_camera_id` config key selects
   which registry camera creates passes (capability-gated by the loader —
   uncalibrated/non-speed cameras must refuse). Switching = config change +
   service restart; no hot-swap.
3. **Ingest payload**: send `camera` (the elected `camera_id`) with every
   pass, alongside the existing `speed_method`. The hub writes the initial
   `pass_speeds` measurement from these fields — coordinate with
   `camwatch-web`'s HANDOVER (its migration ships first).

## Acceptance

- Regression: with the loader-provided homography + cadence, a replayed pass
  produces the same speed as before the switch (the registry's cx810
  artifacts were migrated verbatim from this repo's config).
- Ingest payload carries `camera`; the engine still boots with the hub down.
- Election rejects a camera whose profile lacks a calibrated `speed`
  capability.

## Sequencing

Item 1 can start now. Item 3's hub side lands with `camwatch-web`'s
migration; sending the extra field early is harmless if the hub ignores
unknown fields — verify before relying on that.
