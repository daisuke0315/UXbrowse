import { volumeCenter, volumeDisplaySpan, volumeVoxelCenterSpan } from "../VolumeGeometry.js";
import { ModelVisibilityShaderController } from "./ModelVisibilityShaderController.js";
import { MprPlaneOverlay } from "./MprPlaneOverlay.js";

export class ThreeSceneManager {
  constructor(container, config, logger) {
    this.container = container;
    this.config = config;
    this.logger = logger;
    this.initialized = false;
  }

  async init() {
    const start = performance.now();
    this.THREE = await import("../../vendor/three/three.module.js");
    const { OrbitControls } = await import("../../vendor/three/OrbitControls.js");
    const { OBJLoader } = await import("../../vendor/three/OBJLoader.js");
    const { GLTFLoader } = await import("../../vendor/three/GLTFLoader.js");
    this.OBJLoader = OBJLoader;
    this.GLTFLoader = GLTFLoader;
    const THREE = this.THREE;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.config.background);
    this.camera = new THREE.PerspectiveCamera(this.config.fov, 1, this.config.near, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(this.container.clientWidth || 640, this.container.clientHeight || 420);
    this.container.replaceChildren(this.renderer.domElement);
    this.renderer.domElement.className = "three-canvas";

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = this.config.enableDamping;
    this.controls.dampingFactor = this.config.dampingFactor;
    this.controls.rotateSpeed = this.config.rotateSpeed;
    this.controls.panSpeed = this.config.panSpeed;
    this.controls.zoomSpeed = this.config.zoomSpeed;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x304050, 1.8));
    const light = new THREE.DirectionalLight(0xffffff, 1.5);
    light.position.set(1, -1, 2);
    this.scene.add(light);
    this.axes = new THREE.AxesHelper(80);
    this.scene.add(this.axes);
    this.referenceCube = new THREE.Mesh(
      new THREE.BoxGeometry(40, 40, 40),
      new THREE.MeshStandardMaterial({ color: 0x7dd3fc, roughness: 0.45, metalness: 0.05 })
    );
    this.scene.add(this.referenceCube);
    this.targetMarker = new THREE.Mesh(
      new THREE.SphereGeometry(4, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xffd166, emissive: 0x342000 })
    );
    this.scene.add(this.targetMarker);
    this.sightLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x64d2ff })
    );
    this.scene.add(this.sightLine);
    this.mprPlaneOverlay = new MprPlaneOverlay(THREE, this.scene);
    this.modelVisibility = new ModelVisibilityShaderController(THREE);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    window.addEventListener("resize", () => this.resize());
    this.renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.logger.error("WebGLコンテキストが失われました");
    });

    this.initialized = true;
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(160, -220, 140);
    this.camera.lookAt(this.controls.target);
    this.controls.update();
    this.running = true;
    this.animate();
    this.logger.info("Three.js初期化完了", { ms: Math.round(performance.now() - start) });
    return this;
  }

  configureVolume(metadata, mapper, { preserveCamera = false } = {}) {
    if (!this.initialized) return;
    const THREE = this.THREE;
    this.mapper = mapper;
    if (this.box) {
      this.scene.remove(this.box);
      this.box.geometry.dispose();
      this.box.material.dispose();
    }
    const displaySize = volumeDisplaySpan(metadata);
    const voxelCenterSize = volumeVoxelCenterSpan(metadata);
    const centerDicom = volumeCenter(metadata);
    const centerThree = mapper.dicomPointToThree(centerDicom);
    const rowThree = mapper.dicomVectorToThree([metadata.direction[0], metadata.direction[3], metadata.direction[6]]);
    const columnThree = mapper.dicomVectorToThree([metadata.direction[1], metadata.direction[4], metadata.direction[7]]);
    const sliceThree = mapper.dicomVectorToThree([metadata.direction[2], metadata.direction[5], metadata.direction[8]]);

    const geometry = new THREE.BoxGeometry(displaySize[0], displaySize[1], displaySize[2]);
    const material = new THREE.MeshStandardMaterial({
      color: 0x6ee7b7,
      transparent: true,
      opacity: 0.18,
      wireframe: false,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.box = new THREE.Mesh(geometry, material);
    this.box.renderOrder = 0;
    const basis = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(...rowThree),
      new THREE.Vector3(...columnThree),
      new THREE.Vector3(...sliceThree)
    );
    this.box.setRotationFromMatrix(basis);
    this.box.position.set(...centerThree);
    this.scene.add(this.box);

    const diagonal = Math.hypot(displaySize[0], displaySize[1], displaySize[2]);
    this.camera.near = Math.max(0.1, diagonal / 10000);
    this.camera.far = Math.max(5000, diagonal * this.config.farPaddingFactor);
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = diagonal * 0.05;
    this.controls.maxDistance = diagonal * 5;
    if (!preserveCamera) {
      this.controls.target.set(...centerThree);
      this.camera.position.set(centerThree[0], centerThree[1] - diagonal * 0.9, centerThree[2] + diagonal * 0.5);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(this.controls.target);
    }
    this.controls.update();
    this.updateHelpers();
    this.logger.info("Volume範囲BoxをThree.jsへ配置", {
      voxelCenterSpan: voxelCenterSize.map((v) => Number(v.toFixed(3))),
      displaySize: displaySize.map((v) => Number(v.toFixed(3))),
      center: centerThree.map((v) => Number(v.toFixed(2))),
    });
  }

  async loadObjFile(file, options = {}) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    const text = await file.text();
    return this.loadObjText(text, file.name, options);
  }

  loadObjText(text, fileName = "model.obj", options = {}) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    const THREE = this.THREE;
    const start = performance.now();
    const loader = new this.OBJLoader();
    const object = loader.parse(text);
    this.prepareLoadedModel(object, {
      fileName,
      sourceFormat: "OBJ",
      opacity: options.opacity,
      forceMaterial: true,
    });
    this.replaceModel(object, options);
    const summary = this.summarizeObject(object);
    this.logger.info("OBJモデルを読み込みました", {
      fileName,
      meshes: summary.meshes,
      vertices: summary.vertices,
      size: summary.size.map((v) => Number(v.toFixed(2))),
      ms: Math.round(performance.now() - start),
    });
    return summary;
  }

  async loadGlbFile(file, options = {}) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    const arrayBuffer = await file.arrayBuffer();
    return this.loadGlbArrayBuffer(arrayBuffer, file.name, options);
  }

  loadGlbArrayBuffer(arrayBuffer, fileName = "model.glb", options = {}) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    const start = performance.now();
    const loader = new this.GLTFLoader();

    // GLTFLoader.parseはcallback APIなのでPromiseで包む。pathは空文字にして、
    // 初期対応範囲を「単一.glb内に埋め込まれたgeometry/material/texture」に限定する。
    return new Promise((resolve, reject) => {
      loader.parse(
        arrayBuffer,
        "",
        (gltf) => {
          const object = gltf.scene || gltf.scenes?.[0];
          if (!object) {
            reject(new Error("GLB内にsceneがありません"));
            return;
          }
          this.prepareLoadedModel(object, {
            fileName,
            sourceFormat: "GLB",
            opacity: options.opacity,
            forceMaterial: false,
          });
          this.replaceModel(object, options);
          const summary = this.summarizeObject(object);
          if (summary.meshes === 0) {
            this.clearObjModel();
            reject(new Error("GLB内にmeshがありません"));
            return;
          }
          this.logger.info("GLBモデルを読み込みました", {
            fileName,
            meshes: summary.meshes,
            vertices: summary.vertices,
            size: summary.size.map((v) => Number(v.toFixed(2))),
            ms: Math.round(performance.now() - start),
          });
          resolve(summary);
        },
        (error) => reject(error instanceof Error ? error : new Error(String(error)))
      );
    });
  }

  async applyTextureFile(file) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    if (!this.objModel) throw new Error("先にOBJまたはGLBモデルを読み込んでください");
    const url = URL.createObjectURL(file);
    try {
      const texture = await this.loadTextureUrl(url);
      texture.name = file.name;
      texture.colorSpace = this.THREE.SRGBColorSpace;
      texture.flipY = this.objModel.userData.sourceFormat !== "GLB";
      texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy?.() ?? 1);
      texture.needsUpdate = true;
      this.modelTexture?.dispose();
      this.modelTexture = texture;
      let materials = 0;
      let meshes = 0;
      this.objModel.traverse((child) => {
        if (!child.isMesh || !child.material) return;
        meshes += 1;
        for (const material of asMaterialArray(child.material)) {
          material.map = texture;
          material.color?.set?.(0xffffff);
          material.needsUpdate = true;
          materials += 1;
        }
      });
      return { fileName: file.name, sourceFormat: this.objModel.userData.sourceFormat || null, flipY: texture.flipY, meshes, materials };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  loadTextureUrl(url) {
    const loader = new this.THREE.TextureLoader();
    return new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, (error) => reject(error instanceof Error ? error : new Error(String(error))));
    });
  }

  prepareLoadedModel(object, { fileName, sourceFormat, opacity = 0.75, forceMaterial = false }) {
    const THREE = this.THREE;
    object.name = fileName;
    object.userData.sourceFormat = sourceFormat;
    object.userData.fileName = fileName;
    object.traverse((child) => {
      if (!child.isMesh) return;
      // OBJは材質情報なしの単体ファイルを想定するため確認しやすい標準材質を付与する。
      // GLBは埋め込みmaterial/textureを尊重し、透明度だけ後段の共通処理で調整する。
      if (forceMaterial || !child.material) {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xf4a261,
          roughness: 0.55,
          metalness: 0.05,
          transparent: opacity < 1,
          opacity,
          side: THREE.DoubleSide,
        });
      }
      child.castShadow = false;
      child.receiveShadow = false;
      child.renderOrder = 1;
    });
  }

  replaceModel(object, options) {
    if (this.objModel) this.clearObjModel();
    this.objModel = object;
    this.scene.add(object);
    this.applyObjTransform(options);
    this.modelVisibility.applyToObject(this.objModel, this.foregroundOptions);
  }

  applyObjTransform({ centerToTarget = false, scale = 1, opacity = 0.75, position = [0, 0, 0], rotation = [0, 0, 0], controlPosition = position, controlRotation = rotation, controlAxisFlip = [1, 1, 1] } = {}) {
    if (!this.objModel) return null;
    const THREE = this.THREE;
    this.lastModelOptions = {
      centerToTarget,
      scale,
      opacity,
      position: [...position],
      rotation: [...rotation],
      controlPosition: [...controlPosition],
      controlRotation: [...controlRotation],
      controlAxisFlip: [...controlAxisFlip],
    };
    this.objModel.scale.setScalar(Number.isFinite(scale) ? scale : 1);
    this.objModel.rotation.set(degToRad(rotation[0] || 0), degToRad(rotation[1] || 0), degToRad(rotation[2] || 0));
    this.objModel.position.set(0, 0, 0);
    this.objModel.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.objModel);
    const center = new THREE.Vector3();
    box.getCenter(center);
    if (centerToTarget) {
      // 座標整合済みモデルはThree.js座標=Volume world座標として扱うので移動しない。
      // この分岐は、座標を持たない確認用モデルを現在の注視点へ仮配置するためだけに使う。
      this.objModel.position.copy(this.controls.target).sub(center);
    }
    this.objModel.position.add(new THREE.Vector3(position[0] || 0, position[1] || 0, position[2] || 0));
    this.objModel.traverse((child) => {
      if (child.isMesh && child.material) {
        for (const material of asMaterialArray(child.material)) {
          material.opacity = opacity;
          material.transparent = opacity < 1 || this.foregroundOptions?.enabled === true;
          material.needsUpdate = true;
        }
      }
    });
    this.objModel.updateMatrixWorld(true);
    return this.summarizeObject(this.objModel);
  }

  clearObjModel() {
    if (!this.objModel) return;
    const sourceFormat = this.objModel.userData.sourceFormat || "3D";
    this.scene.remove(this.objModel);
    this.modelVisibility.restore();
    this.objModel.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
        else child.material.dispose();
      }
    });
    this.modelTexture?.dispose();
    this.modelTexture = null;
    this.objModel = null;
    this.logger.info(`${sourceFormat}モデルを消去しました`);
  }

  updateMprPlaneOverlay(canvas, result, request, options = {}) {
    if (!this.initialized || !this.mapper) return null;
    const planeInfo = this.mprPlaneOverlay.update({
      canvas,
      result,
      request,
      mapper: this.mapper,
      camera: this.camera,
      visible: options.visible,
      opacity: options.opacity,
      displayMode: options.displayMode,
      cameraPlaneDistance: options.cameraPlaneDistance,
      flipU: options.flipU,
      flipV: options.flipV,
      showBoundary: options.showBoundary,
      thresholdAlphaEnabled: options.thresholdAlphaEnabled,
      transparentThreshold: options.transparentThreshold,
    });
    if (planeInfo) {
      this.modelVisibility.updatePlane({
        centerThree: planeInfo.centerThree,
        nThree: planeInfo.nThree,
      });
    }
    this.updateForegroundTransparency(options);
    return planeInfo;
  }

  updateOverlayOptions(options = {}) {
    this.mprPlaneOverlay?.setOptions(options);
    this.updateForegroundTransparency(options);
    return this.mprPlaneOverlay?.getPlaneInfo?.() ?? null;
  }

  updateForegroundTransparency({ foregroundEnabled = true, frontAlpha = 0.15, foregroundDistance = 0, fadeWidth = 10, cutMode = false } = {}) {
    this.foregroundOptions = {
      enabled: foregroundEnabled,
      frontAlpha: cutMode ? 0 : frontAlpha,
      foregroundDistance,
      fadeWidth,
    };
    this.modelVisibility.setOptions(this.foregroundOptions);
    if (this.objModel) this.modelVisibility.applyToObject(this.objModel, this.foregroundOptions);
  }

  summarizeObject(object) {
    const THREE = this.THREE;
    let meshes = 0;
    let vertices = 0;
    let meshesWithoutUv = 0;
    let materialsWithMap = 0;
    object.traverse((child) => {
      if (!child.isMesh) return;
      meshes += 1;
      vertices += child.geometry?.attributes?.position?.count ?? 0;
      if (!child.geometry?.attributes?.uv) meshesWithoutUv += 1;
      if (child.material) {
        for (const material of asMaterialArray(child.material)) {
          if (material.map) materialsWithMap += 1;
        }
      }
    });
    const box = new THREE.Box3().setFromObject(object);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    return {
      sourceFormat: object.userData.sourceFormat || "OBJ",
      fileName: object.userData.fileName || object.name || null,
      meshes,
      vertices,
      meshesWithoutUv,
      materialsWithMap,
      size: [size.x, size.y, size.z],
      center: [center.x, center.y, center.z],
      transform: this.lastModelOptions ?? null,
      coordinateMode: this.lastModelOptions?.centerToTarget ? "centered to target" : "raw model coordinate",
      texture: this.modelTexture ? { name: this.modelTexture.name || null } : null,
    };
  }

  onControlsChange(callback) {
    this.controls.addEventListener("change", callback);
  }

  setDynamicFocusPoint(point) {
    if (!point) {
      this.dynamicFocusPoint = null;
      return;
    }
    this.dynamicFocusPoint ??= new this.THREE.Vector3();
    this.dynamicFocusPoint.set(point[0], point[1], point[2]);
  }

  resize() {
    if (!this.renderer || !this.camera) return;
    const rect = this.container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  animate() {
    if (!this.running) return;
    requestAnimationFrame(() => this.animate());
    if (this.controls) this.controls.update();
    this.updateHelpers();
    this.renderer.render(this.scene, this.camera);
  }

  updateHelpers() {
    if (!this.targetMarker || !this.sightLine) return;
    const focusPoint = this.dynamicFocusPoint ?? this.controls.target;
    this.targetMarker.position.copy(focusPoint);
    const points = [this.camera.position.clone(), focusPoint.clone()];
    this.sightLine.geometry.setFromPoints(points);
  }

  getPixelStats(sampleSize = 64) {
    if (!this.renderer) return null;
    const gl = this.renderer.getContext();
    const width = this.renderer.domElement.width;
    const height = this.renderer.domElement.height;
    const size = Math.max(1, Math.min(sampleSize, width, height));
    const x = Math.max(0, Math.floor((width - size) / 2));
    const y = Math.max(0, Math.floor((height - size) / 2));
    const pixels = new Uint8Array(size * size * 4);
    gl.readPixels(x, y, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let nonBlack = 0;
    let min = 765;
    let max = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const brightness = pixels[i] + pixels[i + 1] + pixels[i + 2];
      min = Math.min(min, brightness);
      max = Math.max(max, brightness);
      if (brightness > 20) nonBlack += 1;
    }
    return { width, height, sampleSize: size, nonBlack, min, max };
  }

  getObjSummary() {
    return this.objModel ? this.summarizeObject(this.objModel) : null;
  }

  getModelSummary() {
    return this.getObjSummary();
  }

  getMprPlaneOverlaySummary() {
    return this.mprPlaneOverlay?.getPlaneInfo?.() ?? null;
  }

  getForegroundTransparencySummary() {
    return this.modelVisibility?.getSummary?.() ?? null;
  }
}

function degToRad(degrees) {
  return degrees * Math.PI / 180;
}

function asMaterialArray(material) {
  return Array.isArray(material) ? material : [material];
}
