"""Local make/model enrichment service for camwatch.

Wraps a frozen DINOv2 vision encoder + KNN over labeled embeddings.
Exposes an HTTP API the capture worker calls per pass; high-confidence
matches auto-fill `vehicle_make`/`vehicle_model` on the row, low-confidence
ones are left for the existing Opus sub-agent workflow.
"""
