import os
import shutil

def safe_delete_dir(path):
    if os.path.exists(path):
        print(f"Deleting obsolete directory: {path}")
        try:
            shutil.rmtree(path)
            print("  -> Deleted successfully.")
        except Exception as e:
            print(f"  -> Error deleting: {e}")

def safe_move_dir(src, dst):
    if os.path.exists(src):
        print(f"Moving to deprecated/archive: {src} -> {dst}")
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print("  -> Moved successfully.")
        except Exception as e:
            print(f"  -> Error moving: {e}")

def main():
    data_dir = r"d:\Project\KhmerCurrencyOCR\data"
    dep_dir = os.path.join(data_dir, "deprecated")
    
    # 1. Delete obsolete huge fragment classifier datasets (~2.7 GB)
    obsolete_frag_dirs = [
        "fragment_classifier_cashsnap_realfrag_plus_legacy_backfocus_v1",
        "fragment_classifier_cashsnap_realfrag_plus_legacy_v1",
        "fragment_classifier_cashsnap_realfrag_v1"
    ]
    for d in obsolete_frag_dirs:
        safe_delete_dir(os.path.join(data_dir, d))
        
    # 2. Move small/historical/inactive fragment classifier datasets to data/deprecated
    minor_frag_dirs = [
        "fragment_classifier_v1_candidate",
        "fragment_classifier_review_pack_smoke_v1",
        "fragment_classifier_roboflow_partial_khr_diag_v1",
        "fragment_classifier_roboflow_partial_khr_oldcommon_eval_v1",
        "fragment_classifier_smoke_v1",
        "fragment_classifier_p1_oldcommon_focus_unreviewed_diag_v1",
        "picwish_upload_batches_cashsnap_smoke",
        "picwish_upload_batches"
    ]
    for d in minor_frag_dirs:
        safe_move_dir(os.path.join(data_dir, d), os.path.join(dep_dir, d))
        
    # 3. Delete obsolete/old synthetic datasets (~1.2 GB)
    synth_dir = os.path.join(data_dir, "synthetic")
    obsolete_synth_dirs = [
        "khr_messy_v1",
        "khr_messy_v2",
        "khr_messy_v3",
        "khr_messy_v4",
        "khr_messy_v5_real_crops",
        "khr_fan_v1",
        "khr_old_common_focus_v1",
        "khr_radial_slice_probe_v1",
        "khr_messy_clean_smoke",
        "khr_fan_smoke",
        "khr_current_cutout_obb_smoke",
        "khr_current_cutout_detect_smoke"
    ]
    for d in obsolete_synth_dirs:
        safe_delete_dir(os.path.join(synth_dir, d))
        
    # 4. Move remaining historical synthetic datasets to data/deprecated/synthetic
    dep_synth_dir = os.path.join(dep_dir, "synthetic")
    minor_synth_dirs = [
        "khr_current_thin_radial_slice_probe_v1_obb",
        "khr_current_thin_radial_slice_probe_v1",
        "khr_p1_thin_hand_smoke_v1",
        "khr_p1_oldcommon_thin_hand_smoke_v1",
        "khr_p1_oldcommon_thin_only_smoke_v1",
        "khr_p1_oldcommon_microthin_smoke_v1",
        "khr_rare_gold_probe_v1",
        "khr_rare_gold_clean_v1"
    ]
    for d in minor_synth_dirs:
        safe_move_dir(os.path.join(synth_dir, d), os.path.join(dep_synth_dir, d))
        
    print("\nRepo hygiene cleanup complete!")

if __name__ == "__main__":
    main()
