#!/usr/bin/env python
"""Compile 2x and 4x dataset mixes for V16 training."""

import os
import re
import yaml
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

BASELINE_TXT = ROOT / "configs" / "generated_lists" / "webgl_ablation" / "cashsnap_production_pilot_v2_hardneg_guard_train.txt"
BASELINE_YAML = ROOT / "configs" / "webgl_ablation" / "cashsnap_production_pilot_v2_hardneg_guard.yaml"
MIX_YAML_PATH = ROOT / "configs" / "cashsnap_webgl_trainable_candidates_mix_scaled4x.yaml"

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
    class_ids: list[int] = []
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
    if not BASELINE_TXT.exists():
        print(f"Error: baseline list not found at {BASELINE_TXT}")
        return
    if not BASELINE_YAML.exists():
        print(f"Error: baseline YAML not found at {BASELINE_YAML}")
        return
    if not MIX_YAML_PATH.exists():
        print(f"Error: mix YAML not found at {MIX_YAML_PATH}")
        return

    # Load baseline lines
    lines = BASELINE_TXT.read_text(encoding="utf-8").splitlines()
    non_synth = [l.strip() for l in lines if l.strip() and "cashsnap_target_anchor_transplant" not in l]
    print(f"Loaded {len(lines)} baseline lines. Kept {len(non_synth)} non-synthetic lines.")

    # Load mix config to find recipe folders
    mix_data = yaml.safe_load(MIX_YAML_PATH.read_text(encoding="utf-8"))
    
    # Gather all synthetic image paths per directory
    recipe_imgs = {}
    for train_dir_rel in mix_data["train"]:
        train_dir = ROOT / train_dir_rel
        if not train_dir.exists():
            print(f"Warning: Directory {train_dir_rel} does not exist!")
            continue
        # Find all png files, sort them
        files = sorted(list(train_dir.glob("*.png")))
        recipe_imgs[train_dir_rel] = [f.relative_to(ROOT).as_posix() for f in files]
        print(f"Recipe {train_dir_rel}: found {len(files)} images.")

    # Compile for 2x and 4x
    for scale, fraction in [("2x", 0.5), ("4x", 1.0)]:
        print(f"\n--- COMPILING {scale} SCALE (fraction={fraction}) ---")
        selected_synth = []
        for train_dir_rel, imgs in recipe_imgs.items():
            num_select = int(len(imgs) * fraction)
            selected_synth.extend(imgs[:num_select])
            print(f"  {train_dir_rel}: selected {num_select} / {len(imgs)} images")

        print(f"Total unique synthetic images selected: {len(selected_synth)}")
        
        # Repeat 2x as instructed
        repeated_synth = selected_synth * 2
        print(f"Total repeated synthetic exposures: {len(repeated_synth)}")

        # Merge
        final_list = non_synth + repeated_synth
        print(f"Total final images in mix: {len(final_list)}")

        # Write text list file
        out_txt = ROOT / "configs" / "generated_lists" / "webgl_ablation" / f"cashsnap_production_pilot_v16_scaled{scale}_train.txt"
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(final_list) + "\n", encoding="utf-8")
        print(f"Wrote text list to: {out_txt.relative_to(ROOT)}")

        # Write YAML config file
        out_yaml = ROOT / "configs" / "webgl_ablation" / f"cashsnap_production_pilot_v16_scaled{scale}.yaml"
        
        # Read baseline yaml to clone structure
        base_yaml_data = yaml.safe_load(BASELINE_YAML.read_text(encoding="utf-8"))
        
        # Update config fields
        yaml_data = dict(base_yaml_data)
        yaml_data["train"] = f"configs/generated_lists/webgl_ablation/cashsnap_production_pilot_v16_scaled{scale}_train.txt"
        
        # Calculate summary counts
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
            "tag": f"cashsnap_production_pilot_v16_scaled{scale}",
            "base_reference_config": "configs/webgl_ablation/cashsnap_production_pilot_v2_hardneg_guard.yaml",
            "recommended_init_checkpoint": "runs/cashsnap/fixed_step_countsafe_vis70_p24_from_last_e50_i416_b2_w0_adamw_lr5e6_nowarmup_noamp_cachefalse_freeze22_steps318_seed0/weights/last.pt",
            "component_sources": {
                "clean_real_replay": [
                    "configs/generated_lists/webgl_ablation/cashsnap_v1_balanced_real_only_probe_train.txt"
                ],
                "strictbest_synth_replay": mix_data["train"],
                "countable_partial_replay": [
                    "configs/generated_lists/webgl_ablation/cashsnap_countsafe_vis70_p24_v1_extra.txt",
                    "configs/generated_lists/webgl_ablation/cashsnap_countsafe_vis70_plus_center50_p24_v1_extra.txt",
                    "configs/generated_lists/webgl_ablation/cashsnap_reviewed_countable_khr_borderpartial24_v1_extra.txt",
                    "configs/generated_lists/webgl_ablation/cashsnap_reviewed_visibleevidence_obvioussafe_v1_extra.txt",
                    "runs/cashsnap/visible_evidence_qa_mined_partialstress_cap8_v1/strict_partial_khr_codex_reviewed_v1_images.txt",
                    "runs/cashsnap/real_overlap_focus_materialized_reviewed_v1/train_anchor_candidate_images.txt"
                ],
                "train_safe_hard_negative_replay": [
                    "runs/cashsnap/visible_evidence_qa_lowrisk_empty_candidates_v1/coin_hardneg12_codex_reviewed_v1_images.txt",
                    "configs/generated_lists/webgl_ablation/cashsnap_reviewed_foreignhardneg_koreanwon24_v1_extra.txt",
                    "configs/generated_lists/audit/cashsnap_foreign_asian_currency_top33_from_bgfp_v1.txt",
                    "configs/generated_lists/audit/cashsnap_countable_center50_fpneg32_added_negative_v1.txt",
                    "runs/cashsnap/ve_v4_trainanchors_guard_v2/champion_train_empty_fp_analogs_allclasses_conf015_v1.json"
                ],
                "high_risk_class_protectors": [
                    "configs/generated_lists/webgl_ablation/cashsnap_v1_balanced_real_only_probe_train.txt"
                ]
            },
            "summary": {
                "seed": 20260611,
                "rows": len(final_list),
                "unique_rows": len(set(final_list)),
                "duplicate_exposures": len(final_list) - len(set(final_list)),
                "empty_rows": empty_count,
                "class_counts": class_counts_named,
                "strict_synth_unique": len(selected_synth),
                "strict_synth_repeated_exposures": len(repeated_synth)
            },
            "label_policy": base_yaml_data["cashsnap_production_pilot"]["label_policy"]
        }
        
        yaml_data["cashsnap_policy"] = {
            "intended_use": f"Production Pilot v16 scaled{scale} training recipe: clean real + WebGL synthetic candidates at scaled {scale}.",
            "promotion_rule": base_yaml_data["cashsnap_policy"]["promotion_rule"]
        }
        
        out_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
        print(f"Wrote YAML config to: {out_yaml.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
