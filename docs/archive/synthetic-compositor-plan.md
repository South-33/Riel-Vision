# Synthetic Compositor Plan

Goal: create the best CashSnap OCR/detector for banknotes that can count overlapped bills on phone/browser. Synthetic data is the main engine, but only if the assets and compositor are controlled, mask-based, and visually audited.

## Core Principle

Do not train on bad synthetic data. Start from real-camera bill photos when available, remove the background, verify the cutout/mask visually, then composite those verified bill assets into new scenes.

For every composited scene, labels must be regenerated from the visible pixels after all occlusion. Never reuse the original full-note box when part of the bill is hidden.

## Asset Pipeline

1. Start from real camera/dataset photos with bill labels.
2. Crop or isolate each bill with context.
3. Run background removal so only the bill remains.
4. Keep the alpha mask and a tight transparent PNG.
5. Recompute the bill's local mask box from non-transparent pixels.
6. Visually audit the cutout:
   - keep clean single-bill masks
   - reject missing corners, eaten text, merged hands/background, bad transparency holes, or multi-bill blobs
   - tag denomination, currency, front/back, issue/version, source dataset, and quality
7. Use only verified cutouts for serious synthetic training.

## Compositor Rules

Each synthetic scene is a stack of bill instances placed on a real background. The compositor owns the scene graph:

- each bill has an RGBA image and binary mask
- each bill gets a transform: scale, rotation, perspective, shear, warp, x/y position, z-order
- top bills occlude bottom bills
- visible masks are computed after z-order compositing
- labels are exported from visible masks, not from original boxes

Label exports:

- detect: axis-aligned bounding box around visible pixels
- OBB: rotated/min-area box around visible pixels
- segmentation: visible mask polygon or raster-derived polygon when supported
- optional ignore/drop if visible area is too small or denomination is no longer human-identifiable

## Knobs

Asset selection:

- currency and denomination mix
- front/back ratio
- issue/version family
- pristine/worn/damaged ratio
- rare/common class balancing
- minimum asset quality threshold

Layout:

- non-overlap table spread
- light overlap
- dense pile
- row/grid shop display
- crossed notes
- radial fan
- hand-held fan with shared grip point
- partial off-frame notes
- mixed scale/distance
- z-order policy
- maximum occlusion per bill
- minimum visible area per bill
- avoid or allow touching edges

Geometry:

- x/y translation
- rotation
- scale
- anisotropic scale
- perspective tilt
- shear
- mild paper curl/warp
- folded/corner bend masks
- local crumple/crease displacement

Camera/photo:

- background image bank
- shadows under bills
- contact shadows between overlapping bills
- glare/specular wash
- motion blur
- defocus blur
- JPEG compression
- resolution/downsample/upsample
- sensor noise/grain
- exposure
- contrast
- brightness
- color temperature
- tint
- saturation
- white balance drift
- phone sharpening
- vignette

Occluders:

- fingers/hand masks near fan grip
- table objects
- wallet/receipt edges
- random crop by frame edge

Validation gates:

- save rendered preview sheets for every generated dataset
- report box counts per class
- report visible area distribution
- report overlap/occlusion distribution
- compare generated previews against real benchmark photos
- reject datasets where labels are huge full-note boxes for a slice/fan task

## Training Strategy

Use synthetic at scale, but keep stages explicit:

1. Clean recognition: 1-3 verified bills, little/no overlap.
2. Spread counting: many bills on a table, mostly non-overlap.
3. Moderate overlap: bills partially cover each other with visible-region labels.
4. Dense overlap/fan: strong occlusion, shared grip point, backs/fronts mixed.
5. Real benchmark check: never train on benchmark images; use them to tune the compositor and choose checkpoints.

The model can train mostly on synthetic. The real benchmark is the ruler, not the main data source.

## Fragment Recognition Branch

If detector-only training keeps finding bill regions but confusing denominations, add the detector-plus-classifier branch in `docs/fragment-classifier-plan.md`.

The compositor should export visible-instance crops as a side product, not just YOLO labels. Those crops become a denomination-fragment dataset for a tiny classifier that can run after YOLO proposals in browser/phone builds.
