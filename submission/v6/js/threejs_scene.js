import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let ready = false;

window.initThreeScene = function () {
  if (ready) return;
  ready = true;

  const loading = document.getElementById('scene-loading');
  const canvas  = document.getElementById('threejs-canvas');

  const container = canvas.parentElement;
  const sceneBar = document.getElementById('scene-bar');
  const getRenderSize = () => {
    const barH = sceneBar ? sceneBar.offsetHeight : 0;
    return {
      w: Math.max(1, container.clientWidth),
      h: Math.max(1, container.clientHeight - barH)
    };
  };
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const initialSize = getRenderSize();
  renderer.setSize(initialSize.w, initialSize.h);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07070F);

  const camera = new THREE.PerspectiveCamera(48, initialSize.w / initialSize.h, 0.1, 80);
  camera.position.set(0, 4.8, 5.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.minDistance = 1.8;
  controls.maxDistance = 14;
  controls.target.set(0, 0, 0);

  /* Lighting */
  scene.add(new THREE.AmbientLight(0xfff0dd, 0.16));
  const key = new THREE.DirectionalLight(0xfff5e8, 2.35);
  key.position.set(3, 8, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { near: 0.5, far: 28, left: -9, right: 9, top: 9, bottom: -9 });
  scene.add(key);
  const fill = new THREE.PointLight(0xffffff, 0.12, 22);
  fill.position.set(-5, 4, -2);
  scene.add(fill);
  const rim = new THREE.PointLight(0xffffff, 0.28, 18);
  rim.position.set(0, 2.5, -6);
  scene.add(rim);
  const chromaBounce = new THREE.PointLight(0x38ff24, 0.0, 18);
  chromaBounce.position.set(0, 1.1, 3.8);
  scene.add(chromaBounce);

  /* Table */
  const tbl = new THREE.Mesh(
    new THREE.PlaneGeometry(16, 12),
    new THREE.MeshStandardMaterial({ color: 0x080814, roughness: 0.92, metalness: 0.05 })
  );
  tbl.rotation.x = -Math.PI / 2;
  tbl.receiveShadow = true;
  scene.add(tbl);
  const grid = new THREE.GridHelper(16, 32, 0x18182e, 0x18182e);
  grid.material.opacity = 0.4; grid.material.transparent = true;
  scene.add(grid);

const BILL_TEXTURES = {
    KHR_50000: '../images/khr_50000.jpg',
    KHR_20000: '../images/khr_20000.jpg',
    KHR_10000: '../images/khr_10000.jpg',
    KHR_5000: '../images/khr_5000.jpg',
    KHR_1000: '../images/khr_1000.jpg',
    USD_100: '../images/usd_100.jpg'
};

  const BILL_DEFS = [
    { src: BILL_TEXTURES.KHR_50000, aspect: 2.0 },
    { src: BILL_TEXTURES.KHR_20000, aspect: 2.0 },
    { src: BILL_TEXTURES.KHR_10000, aspect: 2.0 },
    { src: BILL_TEXTURES.KHR_5000,  aspect: 2.0 },
    { src: BILL_TEXTURES.KHR_1000,  aspect: 2.0 },
    { src: BILL_TEXTURES.USD_100,   aspect: 2.35 },
  ];

  const loader = new THREE.TextureLoader();

  /* BoxGeometry face order: +x, -x, +y(top), -y, +z, -z */
  const sideMat   = new THREE.MeshStandardMaterial({ color: 0xede8e0, roughness: 0.95 });
  const bottomMat = new THREE.MeshStandardMaterial({ color: 0xd8d3ca, roughness: 0.95 });

  const BH = 1.0; // bill height (short dim)
  const meshes = [];

  BILL_DEFS.forEach((def, i) => {
    const BW = BH * def.aspect;
    const geo = new THREE.BoxGeometry(BW, 0.014, BH);
    // Start with plain side on top, swap when texture loads
    const mats = [sideMat, sideMat, sideMat, bottomMat, sideMat, sideMat];
    const mesh = new THREE.Mesh(geo, mats);
    mesh.castShadow = true; mesh.receiveShadow = true;
    scene.add(mesh);
    meshes.push(mesh);

    loader.load(def.src, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.generateMipmaps = true;
      tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
      const faceMat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.FrontSide, toneMapped: false });
      const newMats = [...mesh.material];
      newMats[2] = faceMat; // +y = top face
      mesh.material = newMats;
    }, undefined, (err) => {
      console.error("Failed to load texture:", def.src, err);
    });
  });

  /* Scene presets */
  const CLEAN = [
    { x:-2.6, y:0.010, z:-0.7, ry: 0.28 },
    { x:-0.8, y:0.020, z:-1.4, ry:-0.16 },
    { x: 1.0, y:0.010, z:-0.5, ry: 0.38 },
    { x: 2.4, y:0.020, z: 0.7, ry:-0.22 },
    { x:-1.7, y:0.010, z: 1.2, ry: 0.10 },
    { x: 0.5, y:0.020, z: 1.6, ry:-0.34 },
  ];
  const FAN = [
    { x:-3.05, y:0.010, z: 0.55, ry:-0.68 },
    { x:-1.85, y:0.026, z: 0.10, ry:-0.42 },
    { x:-0.62, y:0.042, z:-0.14, ry:-0.16 },
    { x: 0.62, y:0.058, z:-0.12, ry: 0.10 },
    { x: 1.85, y:0.074, z: 0.16, ry: 0.38 },
    { x: 3.05, y:0.090, z: 0.58, ry: 0.66 },
  ];
  const STACK = [
    { x: 0.05, y:0.010, z: 0.08, ry: 0.22 },
    { x:-0.12, y:0.028, z:-0.05, ry:-0.38 },
    { x: 0.08, y:0.046, z:-0.11, ry: 0.55 },
    { x:-0.06, y:0.064, z: 0.14, ry:-0.18 },
    { x: 0.14, y:0.082, z: 0.03, ry: 0.70 },
    { x:-0.09, y:0.100, z:-0.14, ry:-0.44 },
  ];
  const PRESETS = { clean: CLEAN, fan: FAN, stack: STACK };

  const states  = CLEAN.map(p => ({ ...p }));
  let   targets = CLEAN.map(p => ({ ...p }));

  meshes.forEach((m, i) => { m.position.set(CLEAN[i].x, CLEAN[i].y, CLEAN[i].z); m.rotation.y = CLEAN[i].ry; });

  function setPreset(name) {
    if (!PRESETS[name]) return;
    targets = PRESETS[name].map(p => ({ ...p }));
    document.querySelectorAll('.scene-btn').forEach(b => {
      if (b.dataset.scene) {
        b.classList.toggle('active', b.dataset.scene === name);
      }
    });
  }

  document.querySelectorAll('.scene-btn').forEach(b => {
    if (b.dataset.scene) {
      b.addEventListener('click', () => setPreset(b.dataset.scene));
    }
  });

  let bgIndex = 0;
  const bgColors = [0x07070F, 0xE5E5E5, 0x00FF00];
  const tableColors = [0x080814, 0xBBBBBB, 0x00D000];
  const gridOpacities = [0.4, 0.0, 0.0];
  const fillColors = [0xffffff, 0xffffff, 0x72ff64];
  const fillIntensities = [0.12, 0.10, 0.42];
  const bounceIntensities = [0.0, 0.0, 0.65];
  function applyBackdropMode() {
    if (isBboxMode) {
      scene.userData.originalBackground = new THREE.Color(bgColors[bgIndex]);
      if (tbl.userData.originalMaterial) {
        tbl.userData.originalMaterial.color.setHex(tableColors[bgIndex]);
      }
      grid.material.opacity = gridOpacities[bgIndex];
    } else {
      scene.background.setHex(bgColors[bgIndex]);
      if (scene.fog) scene.fog.color.setHex(bgColors[bgIndex]);
      tbl.material.color.setHex(tableColors[bgIndex]);
      grid.material.opacity = gridOpacities[bgIndex];
    }
    fill.color.setHex(fillColors[bgIndex]);
    fill.intensity = fillIntensities[bgIndex];
    chromaBounce.intensity = bounceIntensities[bgIndex];
  }
  
  const bgBtn = document.getElementById('bg-btn');
  const bgBtnText = document.getElementById('bg-btn-text');
  if (bgBtn) {
    bgBtn.addEventListener('click', () => {
      bgIndex = (bgIndex + 1) % 3;
      applyBackdropMode();
      const labels = ["Bg: Dark", "Bg: Light", "Bg: Chroma"];
      bgBtnText.textContent = labels[bgIndex];
    });
  }

  document.getElementById('capture-btn').addEventListener('click', () => {
    const currentSize = getRenderSize();
    
    // Export a stable 3K 16:9 image. This gives the detector cleaner bill detail
    // without going all the way back to oversized 4K files.
    const targetW = 3072;
    const targetH = 1728;
    
    const originalRatio = renderer.getPixelRatio();
    
    // Temporarily upscale the current view without changing the visible layout.
    renderer.setPixelRatio(1);
    camera.aspect = targetW / targetH;
    controls.update();
    camera.updateProjectionMatrix();
    renderer.setSize(targetW, targetH, false);
    
    renderer.render(scene, camera);
    
    const a = document.createElement('a');
    a.download = 'rielvision-3k-demo-scene.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
    
    // Restore layout sizing
    renderer.setPixelRatio(originalRatio);
    camera.aspect = currentSize.w / currentSize.h;
    controls.update();
    camera.updateProjectionMatrix();
    renderer.setSize(currentSize.w, currentSize.h);
    renderer.render(scene, camera);
  });

  // Bbox Mode Variables and Methods
  const ID_COLORS = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0x00ffff, 0xffff00];
  const ID_COLORS_HEX = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ffff00'];
  const BILL_LABELS = [
      'KHR 50,000',
      'KHR 20,000',
      'KHR 10,000',
      'KHR 5,000',
      'KHR 1,000',
      'USD 100'
  ];

  let isBboxMode = false;
  let isFlickering = false;
  const bboxOverlay = document.getElementById('bbox-overlay');

  function getProjectedBBox(mesh, camera, containerWidth, containerHeight) {
      if (!mesh.geometry.boundingBox) {
          mesh.geometry.computeBoundingBox();
      }
      const bbox = mesh.geometry.boundingBox;
      
      const vertices = [
          new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z),
          new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z),
          new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.min.z),
          new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z),
          new THREE.Vector3(bbox.max.x, bbox.min.y, bbox.min.z),
          new THREE.Vector3(bbox.max.x, bbox.min.y, bbox.max.z),
          new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.min.z),
          new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z)
      ];
      
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      
      vertices.forEach(v => {
          v.applyMatrix4(mesh.matrixWorld);
          v.project(camera);
          
          const x = (v.x * 0.5 + 0.5) * containerWidth;
          const y = (-(v.y * 0.5) + 0.5) * containerHeight;
          
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
      });
      
      return {
          left: minX,
          top: minY,
          width: maxX - minX,
          height: maxY - minY
      };
  }

  function updateBboxes() {
      if (!isBboxMode || !bboxOverlay) return;
      
      const { w, h } = getRenderSize();
      
      let html = '';
      
      meshes.forEach((mesh, i) => {
          const rect = getProjectedBBox(mesh, camera, w, h);
          if (!rect) return;
          
          const left = Math.max(0, Math.min(w, rect.left));
          const top = Math.max(0, Math.min(h, rect.top));
          const right = Math.max(0, Math.min(w, rect.left + rect.width));
          const bottom = Math.max(0, Math.min(h, rect.top + rect.height));
          
          const boxWidth = right - left;
          const boxHeight = bottom - top;
          
          if (boxWidth <= 5 || boxHeight <= 5) return;
          
          const color = ID_COLORS_HEX[i];
          const label = BILL_LABELS[i];
          const labelStyle = top < 20 ? `background-color:${color};` : `background-color:${color}; margin-top:-16px;`;
          
          html += `
          <div class="absolute border-2 pointer-events-none flex flex-col justify-start items-start" style="left:${left}px; top:${top}px; width:${boxWidth}px; height:${boxHeight}px; border-color:${color};">
              <span class="text-k-white px-1 font-mono text-[9px] uppercase tracking-wider font-bold" style="${labelStyle}">${label}</span>
          </div>
          `;
      });
      
      bboxOverlay.innerHTML = html;
  }

  function toggleVisualState(targetState) {
      if (targetState) {
          meshes.forEach((mesh, i) => {
              if (!mesh.userData.originalMaterials) {
                  mesh.userData.originalMaterials = mesh.material;
              }
              mesh.material = new THREE.MeshBasicMaterial({ color: ID_COLORS[i] });
          });
          
          if (!tbl.userData.originalMaterial) {
              tbl.userData.originalMaterial = tbl.material;
          }
          tbl.material = new THREE.MeshBasicMaterial({ color: 0x000000 });
          grid.visible = false;
          
          if (!scene.userData.originalBackground) {
              scene.userData.originalBackground = scene.background.clone ? scene.background.clone() : new THREE.Color(0x07070F);
          }
          if (!scene.userData.originalFog) {
              scene.userData.originalFog = scene.fog ? (scene.fog.clone ? scene.fog.clone() : scene.fog) : null;
          }
          scene.background = new THREE.Color(0x000000);
          scene.fog = null;
          
          if (bboxOverlay) bboxOverlay.classList.remove('hidden');
      } else {
          meshes.forEach(mesh => {
              if (mesh.userData.originalMaterials) {
                  mesh.material = mesh.userData.originalMaterials;
              }
          });
          
          if (tbl.userData.originalMaterial) {
              tbl.material = tbl.userData.originalMaterial;
          }
          grid.visible = true;
          
          if (scene.userData.originalBackground) {
              scene.background = scene.userData.originalBackground;
          }
          if (scene.userData.originalFog) {
              scene.fog = scene.userData.originalFog;
          }
          
          if (bboxOverlay) {
              bboxOverlay.classList.add('hidden');
              bboxOverlay.innerHTML = '';
          }
      }
  }

  function startFlicker() {
      if (isFlickering) return;
      isFlickering = true;
      
      const finalState = !isBboxMode;
      let count = 0;
      const maxFlickers = 6;
      
      const interval = setInterval(() => {
          toggleVisualState(count % 2 === 0 ? !isBboxMode : isBboxMode);
          count++;
          if (count >= maxFlickers) {
              clearInterval(interval);
              isBboxMode = finalState;
              toggleVisualState(isBboxMode);
              isFlickering = false;
              
              // update toggle button style & state
              const btn = document.getElementById('bbox-toggle-btn');
              const indicator = document.getElementById('bbox-indicator');
              if (btn) {
                  if (isBboxMode) {
                      btn.classList.remove('text-k-white', 'bg-k-blue', 'border-k-white/30');
                      btn.classList.add('bg-k-red', 'text-k-white', 'border-k-red');
                      if (indicator) indicator.classList.add('hidden');
                  } else {
                      btn.classList.remove('bg-k-red', 'text-k-white', 'border-k-red');
                      btn.classList.add('text-k-white', 'bg-k-blue', 'border-k-white/30');
                      if (indicator) indicator.classList.remove('hidden');
                  }
              }
          }
      }, 50);
  }

  const bboxToggleBtn = document.getElementById('bbox-toggle-btn');
  if (bboxToggleBtn) {
      bboxToggleBtn.addEventListener('click', startFlicker);
  }

  const LERP = 0.065;
  let raf;
  function animate() {
    raf = requestAnimationFrame(animate);
    meshes.forEach((m, i) => {
      states[i].x  += (targets[i].x  - states[i].x)  * LERP;
      states[i].y  += (targets[i].y  - states[i].y)  * LERP;
      states[i].z  += (targets[i].z  - states[i].z)  * LERP;
      states[i].ry += (targets[i].ry - states[i].ry) * LERP;
      m.position.set(states[i].x, states[i].y, states[i].z);
      m.rotation.y = states[i].ry;
    });
    controls.update();
    renderer.render(scene, camera);
    if (typeof updateBboxes === 'function') {
      updateBboxes();
    }
  }
  animate();

  loading.style.opacity = '0';
  setTimeout(() => { loading.style.display = 'none'; }, 500);

  window.addEventListener('resize', () => {
    const { w, h } = getRenderSize();
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  window._stopThree = () => { cancelAnimationFrame(raf); controls.dispose(); };
};

// Automatically run initialization
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (window.initThreeScene) window.initThreeScene();
  });
} else {
  if (window.initThreeScene) window.initThreeScene();
}
