import { volumeCenter, volumeDisplaySpan, volumeVoxelCenterSpan } from "../VolumeGeometry.js";
import { ModelVisibilityShaderController } from "./ModelVisibilityShaderController.js";
import { MprPlaneOverlay } from "./MprPlaneOverlay.js";

export class ThreeSceneManager {
  constructor(container, config, logger) {
    this.container = container;
    this.config = config;
    this.logger = logger;
    this.initialized = false;
    this.foregroundOptions = {
      enabled: false,
      frontAlpha: 0.15,
      foregroundDistance: 0,
      fadeWidth: 10,
    };
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
    this.sceneContentBounds = new THREE.Box3();
    this.sceneContentSphere = new THREE.Sphere();
    this.hasSceneContentBounds = false;
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      // EXT_clip_control対応環境では通常Zより高精度。未対応時はThree.jsが通常Zへ戻す。
      reversedDepthBuffer: true,
    });
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
    this.logger.info("Three.js初期化完了", {
      revision: THREE.REVISION,
      reversedDepthBuffer: this.renderer.capabilities.reversedDepthBuffer === true,
      ms: Math.round(performance.now() - start),
    });
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
    this.refreshSceneContentBounds();
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

  async loadGlbFileSet(fileList, options = {}) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    const files = [...fileList];
    const glbFiles = files.filter((file) => file.name.toLowerCase().endsWith(".glb"));
    if (glbFiles.length === 0) throw new Error("選択したフォルダにGLBファイルがありません");
    if (glbFiles.length > 1) throw new Error(`GLBファイルが${glbFiles.length}個あります。1つだけ含むフォルダを選択してください`);
    const modelFile = glbFiles[0];
    const arrayBuffer = await modelFile.arrayBuffer();
    return this.loadGlbArrayBuffer(arrayBuffer, modelFile.name, options, { resourceFiles: files, modelFile });
  }

  loadGlbArrayBuffer(arrayBuffer, fileName = "model.glb", options = {}, { resourceFiles = [], modelFile = null } = {}) {
    if (!this.initialized) throw new Error("Three.js is not initialized");
    const start = performance.now();
    const objectUrls = [];
    const resolvedResources = new Set();
    const manager = new this.THREE.LoadingManager();
    if (resourceFiles.length && modelFile) {
      manager.setURLModifier((url) => {
        const resourceFile = resolveLocalGltfResource(resourceFiles, modelFile, url);
        if (!resourceFile) return url;
        const objectUrl = URL.createObjectURL(resourceFile);
        objectUrls.push(objectUrl);
        resolvedResources.add(resourceFile.webkitRelativePath || resourceFile.name);
        return objectUrl;
      });
    }
    const loader = new this.GLTFLoader(manager);

    // GLTFLoader.parseはcallback APIなのでPromiseで包む。pathは空文字にして、
    // フォルダ読込時はGLBから参照された同階層のtexture等をObject URLへ解決する。
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
          summary.fileName = fileName;
          summary.resolvedResources = [...resolvedResources];
          if (summary.meshes === 0) {
            this.clearObjModel();
            reject(new Error("GLB内にmeshがありません"));
            return;
          }
          this.logger.info("GLBモデルを読み込みました", {
            fileName,
            meshes: summary.meshes,
            vertices: summary.vertices,
            resolvedResources: summary.resolvedResources,
            size: summary.size.map((v) => Number(v.toFixed(2))),
            ms: Math.round(performance.now() - start),
          });
          resolve(summary);
        },
        (error) => reject(error instanceof Error ? error : new Error(String(error)))
      );
    }).finally(() => {
      for (const objectUrl of objectUrls) URL.revokeObjectURL(objectUrl);
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

  prepareLoadedModel(object, { fileName, sourceFormat, opacity = 1, forceMaterial = false }) {
    const THREE = this.THREE;
    const unlitMaterials = new Map();
    object.name = fileName;
    object.userData.sourceFormat = sourceFormat;
    object.userData.fileName = fileName;
    object.traverse((child) => {
      if (!child.isMesh) return;
      // 外観参照モデルは撮影済みの色を照明で変化させず表示するため、Unlitを正規仕様とする。
      // GLBでは元materialの色・texture・alpha設定を引き継ぎ、照明依存のPBR要素だけを外す。
      if (forceMaterial || !child.material) {
        child.material = new THREE.MeshBasicMaterial({
          color: 0xf4a261,
          transparent: opacity < 1,
          opacity,
          // フォトグラメトリmeshは薄い開放面や局所的に反転した三角形を含み得る。
          // FrontSideでcullすると手前面が角度依存で消え、奥の面が透けたように見える。
          side: THREE.DoubleSide,
          depthTest: true,
          depthWrite: true,
        });
      } else {
        const sourceMaterials = asMaterialArray(child.material);
        const converted = sourceMaterials.map((material) => {
          if (unlitMaterials.has(material)) return unlitMaterials.get(material);
          const unlit = new THREE.MeshBasicMaterial({
            name: material.name,
            color: material.color?.clone?.() ?? new THREE.Color(0xffffff),
            map: material.map ?? null,
            alphaMap: material.alphaMap ?? null,
            vertexColors: material.vertexColors === true || child.geometry?.hasAttribute?.("color") === true,
            transparent: material.transparent || opacity < 1,
            opacity,
            alphaTest: material.alphaTest ?? 0,
            // 両面をdepth test/write付きで描くことで、裏向きの手前面もdepthへ参加させる。
            side: THREE.DoubleSide,
            depthTest: material.depthTest,
            depthWrite: material.depthWrite,
            blending: material.blending,
            fog: material.fog,
          });
          unlitMaterials.set(material, unlit);
          return unlit;
        });
        child.material = Array.isArray(child.material) ? converted : converted[0];
      }
      for (const material of asMaterialArray(child.material)) {
        material.userData.viewSyncBaseTransparent = material.transparent;
        material.userData.viewSyncBaseDepthWrite = material.depthWrite;
        material.userData.viewSyncBaseAlphaHash = material.alphaHash === true;
      }
      child.castShadow = false;
      child.receiveShadow = false;
      child.renderOrder = 1;
    });
    for (const material of unlitMaterials.keys()) material.dispose?.();
  }

  replaceModel(object, options) {
    if (this.objModel) this.clearObjModel();
    this.objModel = object;
    this.scene.add(object);
    this.applyObjTransform(options);
    this.modelVisibility.applyToObject(this.objModel, this.foregroundOptions);
  }

  applyObjTransform({ centerToTarget = false, scale = 1, opacity = 1, position = [0, 0, 0], rotation = [0, 0, 0], controlPosition = position, controlRotation = rotation, controlAxisFlip = [1, 1, 1] } = {}) {
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
          const useHashedForeground = this.foregroundOptions?.enabled === true
            && opacity >= 1
            && material.userData.viewSyncBaseTransparent !== true;
          material.alphaHash = material.userData.viewSyncBaseAlphaHash === true || useHashedForeground;
          material.transparent = material.userData.viewSyncBaseTransparent === true || opacity < 1;
          material.depthWrite = material.transparent
            ? false
            : material.userData.viewSyncBaseDepthWrite !== false;
          material.depthTest = true;
          material.needsUpdate = true;
        }
      }
    });
    this.objModel.updateMatrixWorld(true);
    const summary = this.summarizeObject(this.objModel);
    this.refreshSceneContentBounds();
    return summary;
  }

  getObjCameraInput(distance = 1500) {
    if (!this.objModel) return null;
    const THREE = this.THREE;
    this.objModel.updateMatrixWorld(true);
    this.camera.updateMatrixWorld();
    const box = new THREE.Box3().setFromObject(this.objModel);
    const modelCenter = new THREE.Vector3();
    const eye = new THREE.Vector3();
    const target = new THREE.Vector3();
    const n = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion).normalize();
    box.getCenter(modelCenter);
    this.camera.getWorldPosition(eye);
    // ViewSync MPRの正規仕様:
    // モデル連動中でもMPR注視点はモデル中心固定にしない。
    // Unity版と同じくカメラ前方の固定距離点を使い、ズーム時にMPR断面を移動させる。
    // modelCenterは位置合わせ確認用の参照値として返すだけで、MPR中心には使わない。
    target.copy(eye).addScaledVector(n, Math.max(0.001, distance));
    return {
      eye: [eye.x, eye.y, eye.z],
      target: [target.x, target.y, target.z],
      up: [up.x, up.y, up.z],
      n: [n.x, n.y, n.z],
      distance,
      orbitDistance: this.camera.position.distanceTo(this.controls.target),
      modelCenter: [modelCenter.x, modelCenter.y, modelCenter.z],
      modelCenterDistance: eye.distanceTo(modelCenter),
      targetMode: "camera-forward-model-linked",
    };
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
    this.refreshSceneContentBounds();
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

  updateForegroundTransparency({ foregroundEnabled = false, frontAlpha = 0.15, foregroundDistance = 0, fadeWidth = 10, cutMode = false } = {}) {
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
    let transparentMaterials = 0;
    let depthWriteMaterials = 0;
    let doubleSidedMaterials = 0;
    let alphaHashMaterials = 0;
    object.traverse((child) => {
      if (!child.isMesh) return;
      meshes += 1;
      vertices += child.geometry?.attributes?.position?.count ?? 0;
      if (!child.geometry?.attributes?.uv) meshesWithoutUv += 1;
      if (child.material) {
        for (const material of asMaterialArray(child.material)) {
          if (material.map) materialsWithMap += 1;
          if (material.transparent) transparentMaterials += 1;
          if (material.depthWrite) depthWriteMaterials += 1;
          if (material.side === THREE.DoubleSide) doubleSidedMaterials += 1;
          if (material.alphaHash) alphaHashMaterials += 1;
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
      transparentMaterials,
      depthWriteMaterials,
      doubleSidedMaterials,
      alphaHashMaterials,
      reversedDepthBuffer: this.renderer?.capabilities?.reversedDepthBuffer === true,
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

  refreshSceneContentBounds() {
    if (!this.THREE || !this.sceneContentBounds || !this.sceneContentSphere) return;
    this.sceneContentBounds.makeEmpty();
    for (const object of [this.box, this.objModel]) {
      if (!object) continue;
      object.updateMatrixWorld(true);
      this.sceneContentBounds.expandByObject(object);
    }
    this.hasSceneContentBounds = !this.sceneContentBounds.isEmpty();
    if (this.hasSceneContentBounds) this.sceneContentBounds.getBoundingSphere(this.sceneContentSphere);
    this.updateCameraClipping(true);
  }

  updateCameraClipping(force = false) {
    if (!this.camera) return;
    let requiredFar = 5000;
    let requiredNear = this.config.near ?? 0.1;
    if (this.hasSceneContentBounds) {
      requiredFar = calculateRequiredCameraFar(
        [this.camera.position.x, this.camera.position.y, this.camera.position.z],
        [this.sceneContentSphere.center.x, this.sceneContentSphere.center.y, this.sceneContentSphere.center.z],
        this.sceneContentSphere.radius,
      );
      let protectedDistance = this.controls
        ? this.camera.position.distanceTo(this.controls.target)
        : Number.POSITIVE_INFINITY;
      const planeInfo = this.mprPlaneOverlay?.getPlaneInfo?.();
      if (planeInfo?.visible && planeInfo.centerThree) {
        protectedDistance = Math.min(
          protectedDistance,
          this.camera.position.distanceTo(new this.THREE.Vector3(...planeInfo.centerThree)),
        );
      }
      requiredNear = calculateRequiredCameraNear(
        [this.camera.position.x, this.camera.position.y, this.camera.position.z],
        [this.sceneContentSphere.center.x, this.sceneContentSphere.center.y, this.sceneContentSphere.center.z],
        this.sceneContentSphere.radius,
        this.config.near ?? 0.1,
        protectedDistance,
      );
    } else if (this.controls) {
      requiredFar = Math.max(requiredFar, this.camera.position.distanceTo(this.controls.target) * 2);
    }
    const farChange = Math.abs(requiredFar - this.camera.far) / Math.max(1, this.camera.far);
    const nearChange = Math.abs(requiredNear - this.camera.near) / Math.max(0.001, this.camera.near);
    if (!force && farChange < 0.02 && nearChange < 0.02) return;
    this.camera.near = requiredNear;
    this.camera.far = requiredFar;
    this.camera.updateProjectionMatrix();
  }

  animate() {
    if (!this.running) return;
    requestAnimationFrame(() => this.animate());
    if (this.controls) this.controls.update();
    this.updateCameraClipping();
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

export function resolveLocalGltfResource(files, modelFile, requestedUrl) {
  const requestedPath = normalizeLocalResourcePath(requestedUrl);
  if (!requestedPath || requestedPath.startsWith("data:") || requestedPath.startsWith("blob:")) return null;
  const modelPath = normalizeLocalResourcePath(modelFile.webkitRelativePath || modelFile.name);
  const slash = modelPath.lastIndexOf("/");
  const modelDirectory = slash >= 0 ? modelPath.slice(0, slash) : "";
  const expectedPath = normalizeLocalResourcePath(modelDirectory ? `${modelDirectory}/${requestedPath}` : requestedPath);
  const normalizedFiles = files.map((file) => ({
    file,
    path: normalizeLocalResourcePath(file.webkitRelativePath || file.name),
  }));
  const exact = normalizedFiles.find((entry) => entry.path === expectedPath);
  if (exact) return exact.file;
  const expectedLower = expectedPath.toLowerCase();
  return normalizedFiles.find((entry) => entry.path.toLowerCase() === expectedLower)?.file ?? null;
}

export function calculateRequiredCameraFar(cameraPosition, contentCenter, contentRadius, minimumFar = 5000, padding = 1.1) {
  const dx = cameraPosition[0] - contentCenter[0];
  const dy = cameraPosition[1] - contentCenter[1];
  const dz = cameraPosition[2] - contentCenter[2];
  const distance = Math.hypot(dx, dy, dz);
  const radius = Math.max(0, Number.isFinite(contentRadius) ? contentRadius : 0);
  return Math.max(minimumFar, (distance + radius) * padding);
}

export function calculateRequiredCameraNear(cameraPosition, contentCenter, contentRadius, minimumNear = 0.1, protectedDistance = Number.POSITIVE_INFINITY) {
  const dx = cameraPosition[0] - contentCenter[0];
  const dy = cameraPosition[1] - contentCenter[1];
  const dz = cameraPosition[2] - contentCenter[2];
  const nearestContentDistance = Math.max(0, Math.hypot(dx, dy, dz) - Math.max(0, contentRadius));
  const nearestProtectedDistance = Number.isFinite(protectedDistance)
    ? Math.max(0, protectedDistance)
    : Number.POSITIVE_INFINITY;
  // 最も近い表示対象の25%までnearを前進させ、手前を切らずに深度分解能を確保する。
  return Math.max(minimumNear, Math.min(nearestContentDistance, nearestProtectedDistance) * 0.25);
}

function normalizeLocalResourcePath(value) {
  let path = String(value ?? "").split(/[?#]/, 1)[0].replace(/\\/g, "/");
  try {
    path = decodeURIComponent(path);
  } catch {
    // 不正なpercent encodingは元文字列のまま比較する。
  }
  return path.replace(/^\.\//, "").replace(/^\/+/, "").replace(/\/\.\//g, "/");
}
