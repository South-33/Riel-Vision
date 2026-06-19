import os
import base64
import re

def make_standalone():
    project_root = r"d:\Project\KhmerCurrencyOCR"
    index_path = os.path.join(project_root, "submission", "v6", "RielVision.html")
    demo_js_path = os.path.join(project_root, "submission", "archive", "v6_src", "js", "demo.js")
    model_path = os.path.join(project_root, "submission", "archive", "v6_src", "models", "cashsnap_core14_khr100_final_i640.onnx")

    print(f"Reading HTML file: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    print(f"Reading JS file: {demo_js_path}")
    with open(demo_js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    print(f"Reading Model file and base64 encoding it: {model_path}")
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    model_base64 = base64.b64encode(model_bytes).decode("utf-8")
    print(f"Model base64 size: {len(model_base64)} characters")

    # We will modify the initModel function in the JS content.
    # First, let's define the MODEL_BASE64 constant at the top of the script.
    # Then we modify initModel to use the base64 string.
    
    # Let's inspect where MODEL_URL is defined and replace it/inject MODEL_BASE64
    js_content = js_content.replace(
        "const MODEL_URL = 'models/cashsnap_core14_khr100_final_i640.onnx?v=core14-khr100-final-i640-v2';",
        f"const MODEL_URL = 'models/cashsnap_core14_khr100_final_i640.onnx?v=core14-khr100-final-i640-v2';\nconst MODEL_BASE64 = \"{model_base64}\";"
    )

    # Let's find the initModel function and replace it with the base64 loading version
    target_init_model = """async function initModel() {
    if (session) return;
    isModelLoading = true;
    statusText.innerText = 'LOADING MODEL (9.4MB)...';
    statusText.className = 'text-k-red font-bold uppercase animate-pulse';
    
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        console.log("Model loaded successfully.");
        statusText.innerText = 'READY';
        statusText.className = 'text-green-600 font-bold uppercase';
    } catch (e) {
        console.error("Failed to load model:", e);
        statusText.innerText = 'ERROR LOADING MODEL';
        statusText.className = 'text-k-red font-bold uppercase';
    }
    isModelLoading = false;
}"""

    replacement_init_model = """async function initModel() {
    if (session) return;
    isModelLoading = true;
    statusText.innerText = 'LOADING MODEL (INLINE)...';
    statusText.className = 'text-k-red font-bold uppercase animate-pulse';
    
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        let modelInput;
        if (typeof MODEL_BASE64 !== 'undefined' && MODEL_BASE64) {
            console.log("Decoding embedded base64 model (approx 9.8MB)...");
            const binaryString = atob(MODEL_BASE64);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            modelInput = bytes.buffer;
        } else {
            console.log("Loading model from URL:", MODEL_URL);
            modelInput = MODEL_URL;
        }
        
        session = await ort.InferenceSession.create(modelInput, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        console.log("Model loaded successfully.");
        statusText.innerText = 'READY';
        statusText.className = 'text-green-600 font-bold uppercase';
    } catch (e) {
        console.error("Failed to load model:", e);
        statusText.innerText = 'ERROR LOADING MODEL';
        statusText.className = 'text-k-red font-bold uppercase';
    }
    isModelLoading = false;
}"""

    # Normalize line endings for replacement matching
    target_init_model_norm = target_init_model.replace("\r\n", "\n")
    js_content_norm = js_content.replace("\r\n", "\n")
    
    if target_init_model_norm in js_content_norm:
        print("Found target initModel, replacing...")
        js_content_norm = js_content_norm.replace(target_init_model_norm, replacement_init_model)
    else:
        # Fallback regex replacement if direct replacement doesn't match perfectly
        print("Target initModel not matched exactly, trying regex search...")
        # Search for initModel and replace it
        js_content_norm = re.sub(
            r"async function initModel\(\)\s*\{.*?isModelLoading = false;\s*\}",
            replacement_init_model,
            js_content_norm,
            flags=re.DOTALL
        )

    # Let's locate the demo.js script tag in HTML and replace it with the inline script
    pattern = r'<script src="js/demo.js\?v=[^"]+"></script>'
    replacement = f"<script>\n{js_content_norm}\n</script>"
    
    # We will write the output to submission/v6/RielVision.html (overwriting it)
    new_html_content = re.sub(pattern, replacement, html_content)
    
    if new_html_content == html_content:
        # try without query parameter
        print("Trying pattern without query param...")
        pattern_no_query = r'<script src="js/demo.js"></script>'
        new_html_content = re.sub(pattern_no_query, replacement, html_content)

    if new_html_content == html_content:
        # try replacing an already inlined block
        print("Trying pattern for already inlined script block...")
        pattern_inlined = r'<script>\s*const classNames =[\s\S]*?</script>'
        new_html_content = re.sub(pattern_inlined, replacement, html_content)
        
    if new_html_content == html_content:
        if "MODEL_BASE64" in html_content:
            print("Standalone script is already inlined and up to date (no changes needed).")
        else:
            print("WARNING: Script tag replacement failed. Checking RielVision.html content...")
    else:
        print("Successfully replaced script tag in RielVision.html.")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(new_html_content)
    print("Standalone RielVision.html written successfully!")

if __name__ == "__main__":
    make_standalone()
