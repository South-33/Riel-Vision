#!/usr/bin/env python
"""Compile full real + 2x scaled WebGL synthetic dataset mix for base training."""

import os
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parents[1]

FULL_REAL_TXT = ROOT / "configs" / "generated_lists" / "cashsnap_v1_full_real_only_seed_train.txt"
V2_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v2_hardneg_guard_train.txt"
V16_SCALED2X_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x_train.txt"
V16_SCALED2X_YAML = ROOT / "configs" / "webgl_ablation" / "cashsnap_production_pilot_v16_scaled2x.yaml"

CLASS_NAMES = [
    "USD_1", "USD_5", "USD_10", "USD_20", "USD_50", "USD_100",
    "KHR_500", "KHR_1000", "KHR_2000", "KHR_5000", "KHR_10000", "KHR_20000", "KHR_50000"
]

def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")

def label_class_ids(image: str) -> list[int]:
    label_path = ROOT / label_path_for_image(image)
    if not label_path.exists():
        return []
    class_ids = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 1:
            try:
                class_ids.append(int(float(parts[0])))
            except ValueError:
                pass
    return class_ids

def main():
    if not FULL_REAL_TXT.exists():
        print(f"Error: Full real list not found at {FULL_REAL_TXT}")
        return
    if not V2_TXT.exists():
        print(f"Error: V2 list not found at {V2_TXT}")
        return
    if not V16_SCALED2X_TXT.exists():
        print(f"Error: V16 Scaled2x list not found at {V16_SCALED2X_TXT}")
        return
    if not V16_SCALED2X_YAML.exists():
        print(f"Error: V16 Scaled2x YAML not found at {V16_SCALED2X_YAML}")
        return

    # 1. Load the 14,036 raw real images
    full_real_lines = [l.strip() for l in FULL_REAL_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    full_real_set = set(full_real_lines)
    print(f"Loaded {len(full_real_lines)} raw real images.")

    # 2. Extract extra real crop images from V2 that are not in the raw split
    v2_lines = [l.strip() for l in V2_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    v2_real = [l for l in v2_lines if "data/synthetic" not in l and "cashsnap_target_anchor_transplant" not in l]
    extra_real_crops = sorted(list(set(v2_real) - full_real_set))
    print(f"Found {len(extra_real_crops)} custom real crops/negatives in V2 not in Full Real list.")

    # Combined real images
    combined_real = full_real_lines + extra_real_crops
    print(f"Combined real images count (unique/exposures): {len(combined_real)}")

    # 3. Extract the 560 unique synthetic images (repeated 2x to get 1,120 exposures)
    v16_lines = [l.strip() for l in V16_SCALED2X_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    v16_synth = [l for l in v16_lines if "data/synthetic" in l or "cashsnap_target_anchor_transplant" in l]
    unique_synth = sorted(list(set(v16_synth)))
    print(f"Found {len(unique_synth)} unique synthetic images, total {len(v16_synth)} synthetic exposures.")

    # 4. Merge
    final_list = combined_real + v16_synth
    print(f"Total final images in full mix: {len(final_list)}")

    # Write text list file
    out_txt = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_full_v16_scaled2x_train.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(final_list) + "\n", encoding="utf-8")
    print(f"Wrote compiled training list to: {out_txt.relative_to(ROOT)}")

    # Write YAML config file
    out_yaml = ROOT / "configs" / "webgl_ablation" / "cashsnap_full_v16_scaled2x.yaml"
    
    # Load V16 Scaled2x YAML to preserve structure
    base_yaml_data = yaml.safe_load(V16_SCALED2X_YAML.read_text(encoding="utf-8"))
    
    yaml_data = dict(base_yaml_data)
    yaml_data["train"] = "configs/generated_lists/webgl_ablation/cashsnap_full_v16_scaled2x_train.txt"
    
    # Calculate summary counts for classes and empty rows
    class_counter = Counter()
    empty_count = 0
    for img in final_list:
        cls_ids = label_class_ids(img)
        if not cls_ids:
            empty_count += 1
        else:
            class_counter.update(cls_ids)
            
    class_counts_named = {CLASS_NAMES[cid]: count for cid, count in sorted(class_counter.items())}
    
    # Update pilot metadata
    yaml_data["cashsnap_production_pilot"] = {
        "schema": "cashsnap_production_pilot_config_v1",
        "tag": "cashsnap_production_pilot_full_v16_scaled2x_from_base",
        "base_reference_config": "configs/webgl_ablation/cashsnap_production_pilot_v16_scaled2x.yaml",
        "recommended_init_checkpoint": "yolo26n.pt",
        "component_sources": {
            "full_real_images": [
                "configs/generated_lists/cashsnap_v1_full_real_only_seed_train.txt"
            ],
            "custom_real_crops": [
                "configs/generated_lists/webgl_ablation/cashsnap_production_pilot_v2_hardneg_guard_train.txt"
            ],
            "strictbest_synth_replay": base_yaml_data["cashsnap_production_pilot"]["component_sources"]["strictbest_synth_replay"],
        },
        "summary": {
            "seed": 20260611,
            "rows": len(final_list),
            "unique_rows": len(set(final_list)),
            "duplicate_exposures": len(final_list) - len(set(final_list)),
            "empty_rows": empty_count,
            "class_counts": class_counts_named,
            "strict_synth_unique": len(unique_synth),
            "strict_synth_repeated_exposures": len(v16_synth)
        },
        "label_policy": base_yaml_data["cashsnap_production_pilot"]["label_policy"]
    }
    
    yaml_data["cashsnap_policy"] = {
        "intended_use": "Base training recipe: full 14,036 real + 264 custom real crops + WebGL synthetic candidates at scaled 2x from base yolo26n.pt.",
        "promotion_rule": base_yaml_data["cashsnap_policy"]["promotion_rule"]
    }
    
    out_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    print(f"Wrote YAML config to: {out_yaml.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
