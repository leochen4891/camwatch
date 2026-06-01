# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System context

This repo is **one component of the CamWatch system**. See the whole-system map in
`camwatch-system/SYSTEM.md` (architecture, data flow, the web API contract, shared
glossary, ADRs) before making cross-component changes.

**This component's role:** Producer. Captures passes off the speed camera, computes
speed and direction, records media, stores them in local `camwatch.db`, and uploads
to the hub via `POST /api/ingest`. Also hosts the hourly Opus make/model enrichment
cron and calls the local enricher service. Owns the capture, speed, and ingest core
paths.

CamWatch is hub-and-spoke: `camwatch-web` (Cloudflare) is the hub and system of
record; `camwatch` (this repo, on the desktop) produces passes into it and also runs
Opus make/model enrichment locally; `camwatch-plate-reader` is a hub sidecar (reads
passes, writes plates back); `camwatch-enricher` is a desktop-only experimental local
model beside this service. Cross-machine traffic goes only through the hub's HTTP API.

## Where the detail lives

This repo predates a full CLAUDE.md; the authoritative guides are:

- `README.md` — architecture, capture pipeline, calibration, vehicle enrichment, commands.
- `BABYSITTER.md` and `babysitter-check.md` — the hourly observer/enrichment tick and its observer-only constraints.
- `car_make_model_enrichment.md` — the Opus make/model enrichment design and runbook.
