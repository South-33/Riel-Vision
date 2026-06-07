Scripts archived during the 2026-06-08 housekeeping pass.

Rule used for this pass: keep root `scripts/` for the active closure around
`model.md` canonical checks, current YOLO/WebGL/refiner/capture harnesses,
runtime helpers, and script dependencies found by imports or explicit
`scripts/<name>.py` calls. Move older one-off, browser-stack, fragment/two-stage,
P1/smoke, accepted-blend, bootstrap, and summarizer tools here when they were not
referenced by that active closure.

These files are preserved as reference material. Before restoring one, check
current imports, CLI references, data registry paths, and `model.md` direction.
