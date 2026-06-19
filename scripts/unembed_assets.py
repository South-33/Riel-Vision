import os
import re
import shutil
import base64
import hashlib
import subprocess

def extract_base64_image(b64_data, output_path):
    # Clean up whitespace if any
    b64_clean = re.sub(r'\s+', '', b64_data)
    if not b64_clean:
        return False
    try:
        img_data = base64.b64decode(b64_clean)
        with open(output_path, "wb") as f:
            f.write(img_data)
        return True
    except Exception as e:
        print(f"Error decoding image to {output_path}: {e}")
        return False

def unembed_assets():
    project_root = r"d:\Project\KhmerCurrencyOCR"
    v6_src_dir = os.path.join(project_root, "submission", "archive", "v6_src")
    v6_dest_dir = os.path.join(project_root, "submission", "v6")
    
    # Target directories
    js_dir = os.path.join(v6_dest_dir, "js")
    models_dir = os.path.join(v6_dest_dir, "models")
    images_dir = os.path.join(v6_dest_dir, "images")
    
    # 0. Revert RielVision.html to pre-compiled state first (from commit f73c516)
    html_path = os.path.join(v6_dest_dir, "RielVision.html")
    print(f"Checking out pre-compiled template of RielVision.html from commit f73c516...")
    try:
        subprocess.run(["git", "checkout", "f73c516", "--", html_path], check=True, shell=True)
    except Exception as e:
        print(f"Git checkout failed: {e}")

    # Recreate clean target folders
    print("Recreating clean directories...")
    for folder in [js_dir, models_dir, images_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # 1. Copy the ONNX model file
    src_model = os.path.join(v6_src_dir, "models", "cashsnap_core14_khr100_final_i640.onnx")
    dest_model = os.path.join(models_dir, "cashsnap_core14_khr100_final_i640.onnx")
    print(f"Copying model file to {dest_model}...")
    shutil.copy2(src_model, dest_model)

    # 2. Extract textures from source threejs_scene.js and write them as JPEG files
    src_scene_js = os.path.join(v6_src_dir, "threejs_scene.js")
    dest_scene_js = os.path.join(js_dir, "threejs_scene.js")
    print(f"Processing threejs_scene.js from {src_scene_js}...")
    
    with open(src_scene_js, "r", encoding="utf-8") as f:
        scene_content = f.read()

    # Find the textures inside BILL_TEXTURES
    texture_matches = re.findall(r"([A-Z0-9_]+):\s*'data:image/jpeg;base64,([^']*)'", scene_content)
    
    if texture_matches:
        print(f"Found {len(texture_matches)} 3D bill textures in source scene. Extracting...")
        textures_replacement = "const BILL_TEXTURES = {\n"
        for idx, (key, b64_data) in enumerate(texture_matches):
            filename = f"{key.lower()}.jpg"
            filepath = os.path.join(images_dir, filename)
            
            # Extract and decode
            if extract_base64_image(b64_data, filepath):
                print(f"Extracted bill texture: {filename}")
                
            comma = "," if idx < len(texture_matches) - 1 else ""
            textures_replacement += f"    {key}: 'images/{filename}'{comma}\n"
        textures_replacement += "};"
        
        # Replace textures block in JS
        bill_textures_pattern = r"const BILL_TEXTURES = \{[\s\S]*?\};"
        scene_content = re.sub(bill_textures_pattern, textures_replacement, scene_content)
    else:
        print("ERROR: Could not find BILL_TEXTURES in source JS file!")

    # Fix relative addon imports for standard browser mapping
    scene_content = scene_content.replace(
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';",
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';"
    )
    
    # Save the cleaned threejs_scene.js
    with open(dest_scene_js, "w", encoding="utf-8") as f:
        f.write(scene_content)
    print(f"Cleaned threejs_scene.js written to {dest_scene_js}")

    # 3. Clean up and copy js/demo.js
    src_demo_js = os.path.join(v6_src_dir, "js", "demo.js")
    dest_demo_js = os.path.join(js_dir, "demo.js")
    print(f"Processing demo.js from {src_demo_js}...")
    with open(src_demo_js, "r", encoding="utf-8") as f:
        demo_content = f.read()
        
    # Ensure it loads the model from the local relative models/ directory
    demo_content = demo_content.replace(
        "const MODEL_URL = 'models/cashsnap_core14_khr100_final_i640.onnx?v=core14-khr100-final-i640-v2';",
        "const MODEL_URL = 'models/cashsnap_core14_khr100_final_i640.onnx';"
    )
    with open(dest_demo_js, "w", encoding="utf-8") as f:
        f.write(demo_content)
    print(f"Cleaned demo.js written to {dest_demo_js}")

    # 4. Process RielVision.html
    print(f"Reading HTML template: {html_path}")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Extract all inline base64 images (screenshots/diagrams) to images folder
    print("Extracting inline Base64 slide images...")
    img_pattern = r'data:image/(png|jpeg|jpg|webp);base64,([^"\']*)'
    
    def replace_image(match):
        ext = match.group(1)
        b64_data = match.group(2)
        b64_clean = re.sub(r'\s+', '', b64_data)
        if not b64_clean:
            return match.group(0)
            
        md5_hash = hashlib.md5(b64_clean.encode('utf-8')).hexdigest()[:12]
        filename = f"img_{md5_hash}.{ext}"
        filepath = os.path.join(images_dir, filename)
        
        if not os.path.exists(filepath):
            extract_base64_image(b64_clean, filepath)
            print(f"Extracted slide image: {filename}")
            
        return f"images/{filename}"

    html_content = re.sub(img_pattern, replace_image, html_content)

    # Completely remove the white fade-in overlay element and script
    print("Removing the fade-in-overlay element...")
    overlay_pattern = r'<!-- Blank white page fade-in overlay -->[\s\S]*?window\.addEventListener\(\'DOMContentLoaded\', \(\) => \{[\s\S]*?\}\);\s*</script>'
    html_content = re.sub(overlay_pattern, "", html_content)

    # Replace the inline module script block with a link to external js/threejs_scene.js
    print("Replacing inline module script block with external link...")
    pattern_threejs = r'<script type="module">\s*import \* as THREE from \'three\';[\s\S]*?</script>'
    replacement_threejs = '<script type="module" src="js/threejs_scene.js"></script>'
    
    html_content, count = re.subn(pattern_threejs, replacement_threejs, html_content)
    if count > 0:
        print(f"Replaced inline module script block with external JS file.")
    else:
        # Fallback
        pattern_threejs_fallback = r'<script type="module">[\s\S]*?</script>'
        html_content, count = re.subn(pattern_threejs_fallback, replacement_threejs, html_content)
        if count > 0:
            print(f"Replaced inline module script block with fallback pattern.")
        else:
            print("ERROR: Failed to replace inline module script block!")

    # Replace the demo.js inline script with a link to external js/demo.js if inlined
    pattern_demo_inlined = r'<script>\s*const classNames =[\s\S]*?</script>'
    replacement_demo = '<script src="js/demo.js"></script>'
    html_content, count = re.subn(pattern_demo_inlined, replacement_demo, html_content)
    if count > 0:
        print("Replaced inlined demo.js block with external script link.")
    else:
        # Check if it was already an external link, if so do nothing
        if 'src="js/demo.js"' not in html_content:
            # Try plain fallback
            print("WARNING: Could not find exact demo script block. Attempting query-less script tag replacement...")
            html_content = html_content.replace('<script src="js/demo.js?v=core14-khr100-final-i640-v2"></script>', replacement_demo)

    # Save the cleaned and optimized RielVision.html
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Unembedded RielVision.html written successfully to {html_path}")
    print(f"Final RielVision.html size: {os.path.getsize(html_path)} bytes")

if __name__ == "__main__":
    unembed_assets()
