import os
import re
import hashlib
import base64

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

def unembed_prd_assets():
    project_root = r"d:\Project\KhmerCurrencyOCR"
    v6_dest_dir = os.path.join(project_root, "submission", "v6")
    images_dir = os.path.join(v6_dest_dir, "images")
    prd_path = os.path.join(v6_dest_dir, "RielVisionPRD.html")
    
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Reading PRD: {prd_path}")
    with open(prd_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    print("Extracting inline Base64 images from PRD...")
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
            print(f"Extracted PRD image: {filename}")
            
        return f"images/{filename}"

    cleaned_content = re.sub(img_pattern, replace_image, html_content)
    
    with open(prd_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
        
    print(f"Unembedded RielVisionPRD.html written successfully.")
    print(f"Final RielVisionPRD.html size: {os.path.getsize(prd_path)} bytes")

if __name__ == "__main__":
    unembed_prd_assets()
