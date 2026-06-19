import os
import re
import subprocess
import urllib.request
import hashlib
import base64

def download_file(url):
    print(f"Downloading: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise e

def make_fully_standalone():
    project_root = r"d:\Project\KhmerCurrencyOCR"
    html_path = os.path.join(project_root, "submission", "v6", "RielVision.html")
    scene_src = os.path.join(project_root, "submission", "archive", "v6_src", "threejs_scene.js")
    scene_temp = os.path.join(project_root, "submission", "archive", "v6_src", "threejs_scene.temp.js")
    scene_bundle = os.path.join(project_root, "submission", "archive", "v6_src", "threejs_scene.bundle.js")

    # 0. Revert RielVision.html to the starting state of commit f73c516
    print(f"Reverting {html_path} to starting state of commit f73c516...")
    try:
        subprocess.run(["git", "checkout", "f73c516", "--", html_path], check=True, shell=True)
    except Exception as e:
        print(f"Git checkout failed: {e}")

    # 1. Read threejs_scene.js and extract texture Base64 strings
    print("Preparing threejs_scene.js for bundling...")
    with open(scene_src, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all properties: KHR_50000: 'data:image/jpeg;base64,...'
    texture_matches = re.findall(r"([A-Z0-9_]+):\s*'data:image/jpeg;base64,([^']*)'", content)
    texture_dom_tags = []
    
    if texture_matches:
        print(f"Found {len(texture_matches)} texture Base64 strings in threejs_scene.js.")
        # Replace the textures block with DOM queries in the JS code
        textures_js_replacement = "const BILL_TEXTURES = {\n"
        for idx, (key, base64_data) in enumerate(texture_matches):
            dom_id = f"texture-{key.lower()}"
            # Add to DOM tag list
            texture_dom_tags.append(f'<script type="text/plain" id="{dom_id}">{base64_data}</script>')
            # Add to JS replacement block
            comma = "," if idx < len(texture_matches) - 1 else ""
            textures_js_replacement += f"    {key}: 'data:image/jpeg;base64,' + document.getElementById('{dom_id}').textContent.trim(){comma}\n"
        textures_js_replacement += "};"
        
        # Replace in the JS content
        bill_textures_pattern = r"const BILL_TEXTURES = \{[\s\S]*?\};"
        content_modified = re.sub(bill_textures_pattern, textures_js_replacement, content)
    else:
        print("ERROR: Could not find BILL_TEXTURES in threejs_scene.js!")
        content_modified = content

    # Replace three/addons/ with three/examples/jsm/ so esbuild resolves OrbitControls
    content_modified = content_modified.replace(
        "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';",
        "import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';"
    )

    with open(scene_temp, "w", encoding="utf-8") as f:
        f.write(content_modified)

    # 2. Run esbuild to bundle Three.js and OrbitControls (now without the huge base64 strings!)
    print("Running esbuild to bundle Three.js + OrbitControls...")
    cmd = [
        "npx", "esbuild", scene_temp,
        "--bundle",
        "--minify",
        "--format=iife",
        f"--outfile={scene_bundle}"
    ]
    try:
        subprocess.run(cmd, check=True, shell=True)
        print("esbuild bundling complete.")
    except Exception as e:
        print(f"esbuild failed: {e}")
        raise e
    finally:
        if os.path.exists(scene_temp):
            os.remove(scene_temp)

    # Read bundled scene script
    with open(scene_bundle, "r", encoding="utf-8") as f:
        bundled_scene_code = f.read()
    os.remove(scene_bundle)

    # 3. Download CDN scripts
    tailwind_code = download_file("https://cdn.tailwindcss.com")
    gsap_code = download_file("https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js")
    gsap_trigger_code = download_file("https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js")
    ort_code = download_file("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js")

    # 4. Read HTML and perform script replacements
    print(f"Reading HTML file: {html_path}")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 4b. Extract inline base64 images to external files
    images_dir = os.path.join(project_root, "submission", "v6", "images")
    os.makedirs(images_dir, exist_ok=True)
    print("Extracting inline base64 images to external files...")

    # Pattern matches base64 image URIs inside quotes
    img_pattern = r'data:image/(png|jpeg|jpg|webp);base64,([^"\']*)'

    def replace_image(match):
        ext = match.group(1)
        b64_data = match.group(2)
        
        # Clean up whitespace if any
        b64_clean = re.sub(r'\s+', '', b64_data)
        if not b64_clean:
            return match.group(0)
            
        # Hash to make filename stable
        md5_hash = hashlib.md5(b64_clean.encode('utf-8')).hexdigest()[:12]
        filename = f"img_{md5_hash}.{ext}"
        filepath = os.path.join(images_dir, filename)
        
        # Write bytes if file doesn't exist
        if not os.path.exists(filepath):
            try:
                img_data = base64.b64decode(b64_clean)
                with open(filepath, "wb") as img_file:
                    img_file.write(img_data)
                print(f"Extracted image: {filename}")
            except Exception as e:
                print(f"Error decoding image {filename}: {e}")
                return match.group(0)
                
        return f"images/{filename}"

    # Perform the substitution
    html_content = re.sub(img_pattern, replace_image, html_content)

    # Inlining CDN scripts
    html_content = html_content.replace(
        '<script src="https://cdn.tailwindcss.com"></script>',
        f'<script>\n{tailwind_code}\n</script>'
    )
    html_content = html_content.replace(
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>',
        f'<script>\n{gsap_code}\n</script>'
    )
    html_content = html_content.replace(
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>',
        f'<script>\n{gsap_trigger_code}\n</script>'
    )
    html_content = html_content.replace(
        '<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>',
        f'<script>\n{ort_code}\n</script>'
    )

    # 5. Extract MODEL_BASE64 string literal and convert to type="text/plain" DOM loading
    print("Extracting model base64 string...")
    model_match = re.search(r'const MODEL_BASE64\s*=\s*["\']([^"\']+)["\'];', html_content)
    if model_match:
        model_base64_str = model_match.group(1)
        print(f"Found model base64 string ({len(model_base64_str)} chars).")
        # Replace the JS declaration with DOM retrieval
        html_content = html_content.replace(
            model_match.group(0),
            'const MODEL_BASE64 = document.getElementById("model-data").textContent.trim();'
        )
    else:
        print("ERROR: Could not find MODEL_BASE64 string in HTML!")
        model_base64_str = ""



    # 7. Replace the ES module threejs script block with the bundled scene code
    pattern = r'<script type="module">\s*import \* as THREE from \'three\';[\s\S]*?</script>'
    replacement = f'<script>\n{bundled_scene_code}\n</script>'

    match = re.search(pattern, html_content)
    if match:
        print("Successfully found type='module' script block.")
        start, end = match.span()
        new_html_content = html_content[:start] + replacement + html_content[end:]
    else:
        print("WARNING: Could not find type='module' script block pattern. Trying fallback...")
        pattern_fallback = r'<script type="module">[\s\S]*?</script>'
        match = re.search(pattern_fallback, html_content)
        if match:
            start, end = match.span()
            new_html_content = html_content[:start] + replacement + html_content[end:]
            print("Successfully replaced script type='module' block with fallback pattern.")
        else:
            print("ERROR: Failed to replace type='module' script block.")
            new_html_content = html_content

    # 8. Remove the white fade-in overlay
    print("Removing the fade-in-overlay element and script...")
    overlay_pattern = r'<!-- Blank white page fade-in overlay -->[\s\S]*?window\.addEventListener\(\'DOMContentLoaded\', \(\) => \{[\s\S]*?\}\);\s*</script>'
    new_html_content = re.sub(overlay_pattern, "", new_html_content)

    # 9. Inject the text/plain script elements at the end of the body tag
    print("Injecting model and texture plain text DOM nodes...")
    dom_nodes_html = "\n".join(texture_dom_tags)
    if model_base64_str:
        dom_nodes_html += f'\n<script type="text/plain" id="model-data">{model_base64_str}</script>'
    
    # Insert right before </body>
    new_html_content = new_html_content.replace("</body>", f"{dom_nodes_html}\n</body>")

    # 10. Write the final HTML file back
    print(f"Writing fully standalone optimized HTML file: {html_path}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_html_content)
    print("Standalone RielVision.html updated and optimized successfully!")

if __name__ == "__main__":
    make_fully_standalone()
