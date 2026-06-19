import os
import shutil

def move_v6_source():
    project_root = r"d:\Project\KhmerCurrencyOCR"
    src_dir = os.path.join(project_root, "submission", "v6")
    dest_dir = os.path.join(project_root, "submission", "archive", "v6_src")

    print(f"Creating destination directory: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    items_to_move = ["js", "models", "autostart.bat", "design.html", "script.html", "threejs_scene.js"]

    for item in items_to_move:
        item_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        if os.path.exists(item_path):
            print(f"Moving {item_path} -> {dest_path}")
            if os.path.isdir(item_path):
                # If target directory already exists, remove it first
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(item_path, dest_path)
            else:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(item_path, dest_path)
        else:
            print(f"Warning: {item_path} does not exist, skipping.")

    print("Source files moved successfully!")

if __name__ == "__main__":
    move_v6_source()
