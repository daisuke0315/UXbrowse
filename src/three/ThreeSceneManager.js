import { volumeCenter, volumeDisplaySpan, volumeVoxelCenterSpan } from "../VolumeGeometry.js";
import { ModelVisibilityShaderController } from "./ModelVisibilityShaderController.js?v=locked-foreground-side-1";
import { MprPlaneOverlay } from "./MprPlaneOverlay.js?v=proposal-target-mpr-transparency-1";
import { WeightedBlendedOitRenderer } from "./WeightedBlendedOitRenderer.js?v=locked-foreground-side-1";

export class ThreeSceneManager {
  constructor(container, config, logger) {
    this.container = container;
    this.config = config;
    this.logger = logger;
    this.initialized = false;
    this.running = false;
    this.renderFrameId = null;
    this.renderCount = 0;
    this.qualityRestoreTimer = null;
    this.controlsInteracting = false;
    this.inertiaFramesRemaining = 0;
    this.currentPixelRatio = 1;
    this.foregroundSideSign = 0;
    this.validationResponseMode = false;
    this.foregroundOptions = {
      enabled: false,
      frontAlpha: 0.15,
      foregroundDistance: 0,
      fadeWidth: 10,
      fixedFrontSide: 0,
      transparencyMode: "auto",
    };
  }

  async init() {
    const start = performance.now();
    this.THREE = await import("../../vendor/three/three.module.js");
    const { TrackballControls } = await import("../../vendor/three/TrackballControls.js?v=screen-space-pan-1");
    const { OBJLoader } = await import("../../vendor/three/OBJLoader.js");
    const { GLTFLoader } = await import("../../vendor/three/GLTFLoader.js");
    this.OBJLoader = OBJLoader;
    this.GLTFLoader = GLTFLoader;
    const THREE = this.THREE;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.config.background);
    this.validationOverlayScene = new THREE.Scene();
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
    this.idlePixelRatio = Math.min(window.devicePixelRatio || 1, this.config.maxPixelRatio ?? 2);
    this.interactivePixelRatio = Math.min(this.idlePixelRatio, this.config.interactivePixelRatio ?? 1);
    this.currentPixelRatio = this.idlePixelRatio;
    this.renderer.setPixelRatio(this.currentPixelRatio);
    // 描画バッファだけを初期化し、表示サイズは親要素へ100%追従させる。
    // setSizeの既定動作で固定pxのinline styleを残すと、レイアウト変更後に
    // Canvasの表示比率とcamera.aspectがずれてモデルが横へ伸びて見える。
    this.renderer.setSize(this.container.clientWidth || 640, this.container.clientHeight || 420, false);
    this.container.replaceChildren(this.renderer.domElement);
    this.renderer.domElement.className = "three-canvas";
    this.renderer.domElement.dataset.pixelRatio = String(this.currentPixelRatio);
    this.renderer.domElement.dataset.renderCount = "0";
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";

    // TrackballControlsは縦横とも現在の画面姿勢を基準に回転するため、
    // カメラが横・裏向きでもドラッグ方向と回転方向がずれない。
    this.controls = new TrackballControls(this.camera, this.renderer.domElement);
    this.controls.rotateSpeed = this.config.rotateSpeed;
    this.controls.zoomSpeed = this.config.zoomSpeed;
    this.setInteractionOptions({ panSpeed: this.config.panSpeed, inertia: this.config.inertia });
    this.controls.addEventListener("start", () => {
      this.controlsInteracting = true;
      this.inertiaFramesRemaining = 0;
      this.useInteractivePixelRatio();
      this.requestRender();
    });
    this.controls.addEventListener("change", () => {
      this.useInteractivePixelRatio();
      this.scheduleIdlePixelRatio();
      this.requestRender();
    });
    this.controls.addEventListener("end", () => {
      this.controlsInteracting = false;
      const inertia = this.interactionOptions?.inertia ?? 0;
      this.inertiaFramesRemaining = inertia > 0 ? Math.ceil(6 + inertia * 40) : 0;
      if (this.inertiaFramesRemaining > 0) this.requestRender();
      this.scheduleIdlePixelRatio();
    });

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
    // 初期化確認用の水色Cube。実モデルと混同するため通常表示では非表示にする。
    this.referenceCube.visible = false;
    this.scene.add(this.referenceCube);
    this.mprPlaneOverlay = new MprPlaneOverlay(THREE, this.scene);
    this.modelVisibility = new ModelVisibilityShaderController(THREE);
    this.weightedOit = new WeightedBlendedOitRenderer(THREE, this.renderer, this.scene, this.camera);

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
    this.requestRender();
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
    // Volume範囲は位置合わせ用の補助表示。通常観察ではモデルを隠すため既定は非表示にする。
    this.box.visible = false;
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
    this.refreshSceneContentBounds();
    this.logger.info("Volume範囲BoxをThree.jsへ配置", {
      voxelCenterSpan: voxelCenterSize.map((v) => Number(v.toFixed(3))),
      displaySize: displaySize.map((v) => Number(v.toFixed(3))),
      center: centerThree.map((v) => Number(v.toFixed(2))),
    });
    this.requestRender();
  }

  clearVolume() {
    if (this.box) {
      this.scene.remove(this.box);
      this.box.geometry.dispose();
      this.box.material.dispose();
      this.box = null;
    }
    this.mapper = null;
    this.mprPlaneOverlay?.setOptions({ visible: false });
    this.foregroundOptions.enabled = false;
    this.modelVisibility?.setOptions(this.foregroundOptions);
    this.refreshSceneContentBounds();
    this.requestRender();
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
      this.requestRender();
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
    this.requestRender();
    return summary;
  }

  getObjCameraInput(distance = 600) {
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

  focusObjModelAtDistance(distance = 600) {
    if (!this.objModel || !this.camera || !this.controls) return null;
    this.objModel.updateMatrixWorld(true);
    const box = new this.THREE.Box3().setFromObject(this.objModel);
    if (box.isEmpty()) return null;
    const center = new this.THREE.Vector3();
    const cameraDirection = new this.THREE.Vector3();
    box.getCenter(center);
    cameraDirection.copy(this.camera.position).sub(this.controls.target);
    if (cameraDirection.lengthSq() < 1e-8) cameraDirection.set(0, -1, 0.5);
    cameraDirection.normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(cameraDirection, Math.max(1, distance));
    this.camera.lookAt(center);
    this.controls.update();
    this.refreshSceneContentBounds();
    this.requestRender();
    return center.toArray();
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
    this.requestRender();
    this.logger.info(`${sourceFormat}モデルを消去しました`);
  }

  pickModelPoint(clientX, clientY) {
    if (!this.objModel || !this.camera || !this.renderer) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const pointer = new this.THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.scene.updateMatrixWorld(true);
    const raycaster = new this.THREE.Raycaster();
    raycaster.setFromCamera(pointer, this.camera);
    const hit = raycaster.intersectObject(this.objModel, true)[0];
    return hit ? hit.point.toArray() : null;
  }

  showValidationTarget(dicomPoint, { attachToMpr = false } = {}) {
    this.clearValidationTarget();
    // グレースケールCTと黄系の可視光モデルの両方に埋もれにくい高彩度マゼンタ。
    this.validationTargetMarker = this.createValidationSphereMarker(dicomPoint, 0xff4fd8, {
      opacity: 1,
      depthTest: true,
      name: "validation-target-sphere",
    });
    if (attachToMpr) this.attachValidationTargetToMpr();
  }

  attachValidationTargetToMpr() {
    if (!this.validationTargetMarker || !this.camera || !this.controls) return;
    const targetWorld = this.validationTargetMarker.position.clone();
    const pose = calculateFocusAlignedCameraPose(
      this.camera.position.toArray(),
      this.controls.target.toArray(),
      targetWorld.toArray(),
    );
    this.camera.position.set(...pose.camera);
    this.controls.target.set(...pose.target);
    this.validationTargetMarker.material.transparent = false;
    this.validationTargetMarker.material.opacity = 1;
    // メインシーンの深度バッファを使い、球をピクセル単位でモデルに隠す。
    // 球全体の表示/非表示を切り替えないため、断面通過時も連続的に縮小する。
    this.validationTargetMarker.material.depthTest = true;
    this.validationTargetMarker.material.depthWrite = false;
    this.validationTargetMarker.material.side = this.THREE.FrontSide;
    this.validationTargetMarker.material.needsUpdate = true;
    this.validationTargetMarker.userData.usesMprSphereSection = true;
    this.ensureValidationOverlayLights();
    this.validationOverlayScene.add(this.validationTargetMarker);
    this.validationTargetMarker.position.copy(targetWorld);
    this.camera.updateMatrixWorld();
    this.controls.update();
    this.requestRender();
  }

  ensureValidationOverlayLights() {
    if (this.validationOverlayLights?.length) return;
    // 環境光を抑え、方向光の勾配で小さい球でも立体形状を判別できるようにする。
    const hemisphere = new this.THREE.HemisphereLight(0xffffff, 0x182632, 0.6);
    const directional = new this.THREE.DirectionalLight(0xffffff, 1.7);
    directional.position.set(-1.2, -1.5, 2.4);
    this.validationOverlayScene.add(hemisphere, directional);
    this.validationOverlayLights = [hemisphere, directional];
  }

  getValidationTargetRadiusDicom() {
    const radius = this.validationTargetMarker?.userData?.validationRadius;
    if (!this.mapper || !Number.isFinite(radius) || radius <= 0) return null;
    const axisRadii = [
      [radius, 0, 0],
      [0, radius, 0],
      [0, 0, radius],
    ].map((axis) => {
      const vector = this.mapper.threeVectorToDicom(axis);
      return Math.hypot(vector[0], vector[1], vector[2]);
    }).filter((value) => Number.isFinite(value) && value > 0);
    if (!axisRadii.length) return null;
    return axisRadii.reduce((sum, value) => sum + value, 0) / axisRadii.length;
  }

  showValidationAnswer(dicomPoint) {
    this.clearValidationAnswer();
    this.validationAnswerMarker = this.createValidationMarker(dicomPoint, 0x7ee7ef, 1, { opacity: 0.52 });
    if (this.validationAnswerMarker) {
      this.validationGizmo = this.createValidationGizmo(this.validationAnswerMarker.position);
    }
  }

  showValidationResults(trials) {
    this.clearValidationMarkers();
    this.validationResultMarkers = [];
    const styles = {
      mpr: {
        target: { color: 0x62f0a7, scale: 1.3, label: "M正", labelOffset: [-62, -48] },
        answer: { color: 0xffb45e, scale: 1.3, label: "M答", labelOffset: [-62, 48] },
      },
      proposal: {
        target: { color: 0x59a8ff, scale: 1.3, label: "P正", labelOffset: [62, -48] },
        answer: { color: 0xf38bdc, scale: 1.3, label: "P答", labelOffset: [62, 48] },
      },
    };
    for (const trial of trials ?? []) {
      const style = styles[trial.condition];
      if (!style) continue;
      this.addValidationResultPoint(trial.target_xyz_mm, style.target, `${trial.condition}-target`);
      this.addValidationResultPoint(trial.answer_xyz_mm, style.answer, `${trial.condition}-answer`);
    }
    this.requestRender();
  }

  addValidationResultPoint(dicomPoint, style, name) {
    const point = this.mapper?.dicomPointToThree?.(dicomPoint);
    if (!point) return;
    this.addValidationResultAnchor(point, style.color, name);
    const visible = this.createValidationMarker(dicomPoint, style.color, style.scale, {
      opacity: 0.96,
      depthTest: true,
      label: style.label,
      labelOffset: style.labelOffset,
      name: `validation-result-${name}`,
    });
    const xray = this.createValidationMarker(dicomPoint, style.color, style.scale, {
      opacity: 0.3,
      depthTest: false,
      destinationScene: this.validationOverlayScene,
      label: style.label,
      labelOffset: style.labelOffset,
      name: `validation-result-${name}-xray`,
    });
    if (visible) this.validationResultMarkers.push(visible);
    if (xray) this.validationResultMarkers.push(xray);
  }

  addValidationResultAnchor(point, color, name) {
    const radius = this.hasSceneContentBounds
      ? Math.max(2.2, this.sceneContentSphere.radius * 0.018)
      : 4;
    const layers = [
      {
        scene: this.scene,
        opacity: 0.98,
        depthTest: true,
        renderOrder: 998,
        suffix: "",
      },
      {
        scene: this.validationOverlayScene,
        opacity: 0.28,
        depthTest: false,
        renderOrder: 999,
        suffix: "-xray",
      },
    ];
    for (const layer of layers) {
      const geometry = new this.THREE.SphereGeometry(radius, 20, 14);
      const material = new this.THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: layer.opacity,
        depthTest: layer.depthTest,
        depthWrite: false,
        toneMapped: false,
      });
      const anchor = new this.THREE.Mesh(geometry, material);
      anchor.position.set(...point);
      anchor.renderOrder = layer.renderOrder;
      anchor.name = `validation-result-${name}-anchor${layer.suffix}`;
      layer.scene.add(anchor);
      this.validationResultMarkers.push(anchor);
    }
  }

  getValidationInitialAnswerPoint() {
    if (!this.objModel || !this.THREE) return null;
    this.scene.updateMatrixWorld(true);
    const bounds = new this.THREE.Box3().setFromObject(this.objModel);
    if (bounds.isEmpty()) return this.hasSceneContentBounds ? this.sceneContentSphere.center.toArray() : null;
    return bounds.getCenter(new this.THREE.Vector3()).toArray();
  }

  createValidationMarker(dicomPoint, color, scale = 1, {
    opacity = 1,
    depthTest = false,
    destinationScene = this.scene,
    name = "validation-marker",
    label = "",
    labelOffset = null,
  } = {}) {
    if (!this.mapper || !this.scene) return null;
    const point = this.mapper.dicomPointToThree(dicomPoint);
    const markerCanvas = document.createElement("canvas");
    const hasCallout = Array.isArray(labelOffset);
    const canvasSize = hasCallout ? 256 : 128;
    const center = canvasSize / 2;
    markerCanvas.width = canvasSize;
    markerCanvas.height = canvasSize;
    const context = markerCanvas.getContext("2d");
    const cssColor = `#${color.toString(16).padStart(6, "0")}`;
    context.strokeStyle = cssColor;
    context.fillStyle = "rgba(4, 12, 16, 0.7)";
    context.lineWidth = 8;
    context.beginPath();
    context.arc(center, center, hasCallout ? 18 : 30, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    if (label && hasCallout) {
      const labelX = center + labelOffset[0];
      const labelY = center + labelOffset[1];
      context.lineWidth = 5;
      context.beginPath();
      context.moveTo(center, center);
      context.lineTo(labelX, labelY);
      context.stroke();
      context.fillStyle = "rgba(4, 12, 16, 0.88)";
      context.strokeStyle = cssColor;
      context.lineWidth = 5;
      context.beginPath();
      context.roundRect(labelX - 34, labelY - 23, 68, 46, 12);
      context.fill();
      context.stroke();
      context.fillStyle = cssColor;
      context.font = "800 29px system-ui, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(label, labelX, labelY + 1);
    } else if (label) {
      context.fillStyle = cssColor;
      context.font = `800 ${label.length > 1 ? 34 : 54}px system-ui, sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(label, center, center + 3);
    } else {
      context.beginPath();
      context.moveTo(12, center);
      context.lineTo(canvasSize - 12, center);
      context.moveTo(center, 12);
      context.lineTo(center, canvasSize - 12);
      context.stroke();
    }
    const texture = new this.THREE.CanvasTexture(markerCanvas);
    const marker = new this.THREE.Sprite(new this.THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity,
      depthTest,
      depthWrite: false,
      toneMapped: false,
    }));
    const markerSize = (this.hasSceneContentBounds ? Math.max(10, this.sceneContentSphere.radius * 0.1) : 24) * scale;
    const canvasScale = hasCallout ? 2 : 1;
    marker.scale.set(markerSize * canvasScale, markerSize * canvasScale, 1);
    marker.position.set(...point);
    marker.renderOrder = 1000;
    marker.name = name;
    destinationScene.add(marker);
    this.requestRender();
    return marker;
  }

  createValidationSphereMarker(dicomPoint, color, {
    opacity = 1,
    depthTest = true,
    destinationScene = this.scene,
    name = "validation-sphere-marker",
  } = {}) {
    if (!this.mapper || !this.scene) return null;
    const point = this.mapper.dicomPointToThree(dicomPoint);
    const radius = this.hasSceneContentBounds
      ? Math.max(2.8, this.sceneContentSphere.radius * 0.022)
      : 5;
    const geometry = new this.THREE.SphereGeometry(radius, 24, 16);
    const material = depthTest
      ? new this.THREE.MeshStandardMaterial({
        color,
        roughness: 0.42,
        metalness: 0.03,
        emissive: new this.THREE.Color(color),
        emissiveIntensity: 0.12,
        transparent: false,
        opacity: 1,
        depthTest: true,
        depthWrite: true,
      })
      : new this.THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      });
    const sphere = new this.THREE.Mesh(geometry, material);
    sphere.position.set(...point);
    sphere.renderOrder = 1000;
    sphere.name = name;
    sphere.userData.validationRadius = radius;
    destinationScene.add(sphere);
    this.requestRender();
    return sphere;
  }

  createValidationGizmo(position) {
    const THREE = this.THREE;
    const group = new THREE.Group();
    const length = this.hasSceneContentBounds ? Math.max(24, this.sceneContentSphere.radius * 0.2) : 44;
    const shaftRadius = Math.max(1.4, length * 0.04);
    const coneRadius = shaftRadius * 2.5;
    const coneLength = length * 0.22;
    const axes = [
      { name: "X", vector: new THREE.Vector3(1, 0, 0), color: 0xff5b64, rotate: ["z", -Math.PI / 2] },
      { name: "Y", vector: new THREE.Vector3(0, 1, 0), color: 0x62f0a7, rotate: null },
      { name: "Z", vector: new THREE.Vector3(0, 0, 1), color: 0x59a8ff, rotate: ["x", Math.PI / 2] },
    ];
    for (const axis of axes) {
      const layers = [
        { name: "xray", depthTest: false, opacity: 0.32, renderOrder: 1001 },
        { name: "visible", depthTest: true, opacity: 0.92, renderOrder: 1002 },
      ];
      for (const layer of layers) {
        const material = new THREE.MeshBasicMaterial({
          color: axis.color,
          depthTest: layer.depthTest,
          depthWrite: false,
          transparent: true,
          opacity: layer.opacity,
          toneMapped: false,
        });
        const shaft = new THREE.Mesh(
          new THREE.CylinderGeometry(shaftRadius, shaftRadius, length - coneLength, 12),
          material
        );
        shaft.position.copy(axis.vector).multiplyScalar((length - coneLength) / 2);
        const cone = new THREE.Mesh(
          new THREE.ConeGeometry(coneRadius, coneLength, 16),
          material.clone()
        );
        cone.position.copy(axis.vector).multiplyScalar(length - coneLength / 2);
        for (const mesh of [shaft, cone]) {
          if (axis.rotate) mesh.rotation[axis.rotate[0]] = axis.rotate[1];
          mesh.userData.validationAxis = axis.name;
          mesh.userData.validationLayer = layer.name;
          mesh.renderOrder = layer.renderOrder;
          group.add(mesh);
        }
      }
      const xrayLabel = this.createValidationAxisLabel(axis.name, axis.color, length * 0.22, {
        depthTest: false,
        opacity: 0.38,
        renderOrder: 1001,
        layerName: "xray",
      });
      const visibleLabel = this.createValidationAxisLabel(axis.name, axis.color, length * 0.22, {
        depthTest: true,
        opacity: 0.95,
        renderOrder: 1003,
        layerName: "visible",
      });
      for (const label of [xrayLabel, visibleLabel]) {
        label.position.copy(axis.vector).multiplyScalar(length * 1.12);
        group.add(label);
      }
    }
    group.position.copy(position);
    group.userData.axisLength = length;
    group.name = "validation-translation-gizmo";
    this.scene.add(group);
    this.requestRender();
    return group;
  }

  createValidationAxisLabel(text, color, size, {
    depthTest = false,
    opacity = 0.92,
    renderOrder = 1003,
    layerName = "visible",
  } = {}) {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 96;
    const context = canvas.getContext("2d");
    const cssColor = `#${color.toString(16).padStart(6, "0")}`;
    context.fillStyle = "rgba(5, 14, 19, 0.72)";
    context.strokeStyle = cssColor;
    context.lineWidth = 6;
    context.beginPath();
    context.arc(48, 48, 34, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = cssColor;
    context.font = "700 48px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, 48, 50);
    const texture = new this.THREE.CanvasTexture(canvas);
    const label = new this.THREE.Sprite(new this.THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity,
      depthTest,
      depthWrite: false,
      toneMapped: false,
    }));
    label.scale.set(size, size, 1);
    label.renderOrder = renderOrder;
    label.name = `validation-axis-${text}-${layerName}`;
    return label;
  }

  beginValidationMarkerDrag(clientX, clientY) {
    if (!this.validationGizmo || !this.validationAnswerMarker) return false;
    const raycaster = this.createPointerRaycaster(clientX, clientY);
    if (!raycaster) return false;
    const hit = raycaster.intersectObject(this.validationGizmo, true)
      .find((candidate) => candidate.object?.userData?.validationAxis);
    const axisName = hit?.object?.userData?.validationAxis;
    if (!axisName) return false;
    const axis = axisName === "X"
      ? new this.THREE.Vector3(1, 0, 0)
      : axisName === "Y"
        ? new this.THREE.Vector3(0, 1, 0)
        : new this.THREE.Vector3(0, 0, 1);
    const originScreen = this.worldToClient(this.validationGizmo.position);
    const endpointScreen = this.worldToClient(
      this.validationGizmo.position.clone().addScaledVector(axis, this.validationGizmo.userData.axisLength)
    );
    const screenAxis = new this.THREE.Vector2(
      endpointScreen.x - originScreen.x,
      endpointScreen.y - originScreen.y
    );
    const screenLength = Math.max(1, screenAxis.length());
    screenAxis.normalize();
    this.validationMarkerDrag = {
      axis,
      screenAxis,
      worldPerPixel: this.validationGizmo.userData.axisLength / screenLength,
      startPointer: new this.THREE.Vector2(clientX, clientY),
      startPosition: this.validationGizmo.position.clone(),
    };
    if (this.controls) this.controls.enabled = false;
    return true;
  }

  updateValidationMarkerDrag(clientX, clientY) {
    const drag = this.validationMarkerDrag;
    if (!drag) return null;
    const pointerDelta = new this.THREE.Vector2(clientX, clientY).sub(drag.startPointer);
    const distance = pointerDelta.dot(drag.screenAxis) * drag.worldPerPixel;
    const position = drag.startPosition.clone().addScaledVector(drag.axis, distance);
    this.validationGizmo.position.copy(position);
    this.validationAnswerMarker.position.copy(position);
    this.requestRender();
    return position.toArray();
  }

  endValidationMarkerDrag() {
    if (!this.validationMarkerDrag) return null;
    this.validationMarkerDrag = null;
    if (this.controls) this.controls.enabled = true;
    this.requestRender();
    return this.validationAnswerMarker?.position.toArray() ?? null;
  }

  createPointerRaycaster(clientX, clientY) {
    if (!this.camera || !this.renderer) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const pointer = new this.THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new this.THREE.Raycaster();
    raycaster.setFromCamera(pointer, this.camera);
    return raycaster;
  }

  worldToClient(point) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const projected = point.clone().project(this.camera);
    return {
      x: rect.left + (projected.x + 1) * rect.width / 2,
      y: rect.top + (1 - projected.y) * rect.height / 2,
    };
  }

  clearValidationTarget() {
    this.disposeValidationMarker("validationTargetMarker");
    this.disposeValidationMarker("validationTargetXrayMarker");
  }

  clearValidationAnswer() {
    this.endValidationMarkerDrag();
    this.disposeValidationMarker("validationAnswerMarker");
    this.disposeValidationMarker("validationAnswerXrayMarker");
    if (this.validationGizmo) {
      this.scene.remove(this.validationGizmo);
      this.validationGizmo.traverse((object) => {
        object.geometry?.dispose?.();
        object.material?.map?.dispose?.();
        object.material?.dispose?.();
      });
      this.validationGizmo = null;
      this.requestRender();
    }
  }

  clearValidationMarkers() {
    this.clearValidationTarget();
    this.clearValidationAnswer();
    this.clearValidationResultMarkers();
  }

  clearValidationResultMarkers() {
    for (const marker of this.validationResultMarkers ?? []) {
      marker.parent?.remove(marker);
      marker.geometry?.dispose?.();
      marker.material.map?.dispose?.();
      marker.material.dispose();
    }
    this.validationResultMarkers = [];
    this.requestRender();
  }

  disposeValidationMarker(property) {
    const marker = this[property];
    if (!marker) return;
    marker.parent?.remove(marker);
    marker.geometry?.dispose?.();
    marker.material.map?.dispose?.();
    marker.material.dispose();
    this[property] = null;
    this.requestRender();
  }

  setValidationResponseMode(enabled) {
    this.validationResponseMode = Boolean(enabled);
    if (enabled) {
      this.mprPlaneOverlay?.setOptions({ visible: false });
      this.foregroundOptions.enabled = false;
      this.modelVisibility?.setOptions(this.foregroundOptions);
    }
    this.requestRender();
  }

  getCameraPose() {
    if (!this.camera) return null;
    return {
      position: this.camera.position.toArray(),
      target: this.controls?.target?.toArray?.() ?? null,
      up: this.camera.up.toArray(),
      zoom: this.camera.zoom,
    };
  }

  updateMprPlaneOverlay(canvas, result, request, options = {}) {
    if (!this.initialized || !this.mapper) return null;
    if (this.validationResponseMode) {
      this.mprPlaneOverlay?.setOptions({ visible: false });
      this.foregroundOptions.enabled = false;
      this.modelVisibility?.setOptions(this.foregroundOptions);
      this.requestRender();
      return this.mprPlaneOverlay?.getPlaneInfo?.() ?? null;
    }
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
      const visibilityPlane = {
        centerThree: planeInfo.centerThree,
        uThree: planeInfo.uThree,
        vThree: planeInfo.vThree,
        nThree: planeInfo.nThree,
        widthMm: planeInfo.widthMm,
        heightMm: planeInfo.heightMm,
      };
      this.modelVisibility.updatePlane(visibilityPlane);
    }
    this.updateForegroundTransparency(options);
    this.requestRender();
    return planeInfo;
  }

  updateOverlayOptions(options = {}) {
    if (this.validationResponseMode) {
      this.mprPlaneOverlay?.setOptions({ ...options, visible: false });
      this.foregroundOptions.enabled = false;
      this.modelVisibility?.setOptions(this.foregroundOptions);
      this.requestRender();
      return this.mprPlaneOverlay?.getPlaneInfo?.() ?? null;
    }
    this.mprPlaneOverlay?.setOptions(options);
    this.updateForegroundTransparency(options);
    this.requestRender();
    return this.mprPlaneOverlay?.getPlaneInfo?.() ?? null;
  }

  updateForegroundTransparency({ foregroundEnabled = false, frontAlpha = 0, foregroundDistance = 0, fadeWidth = 10 } = {}) {
    this.foregroundOptions = {
      enabled: foregroundEnabled,
      frontAlpha,
      foregroundDistance,
      fadeWidth,
      fixedFrontSide: this.foregroundSideSign,
      transparencyMode: "auto",
    };
    this.modelVisibility.setOptions(this.foregroundOptions);
    if (this.objModel) this.modelVisibility.applyToObject(this.objModel, this.foregroundOptions);
    this.requestRender();
  }

  setForegroundSideLocked(locked) {
    if (locked && this.camera && this.modelVisibility) {
      const cameraOffset = this.camera.position.clone().sub(this.modelVisibility.planeCenter);
      const cameraSide = cameraOffset.dot(this.modelVisibility.planeNormal);
      this.foregroundSideSign = cameraSide < 0 ? -1 : 1;
    } else {
      this.foregroundSideSign = 0;
    }
    this.foregroundOptions.fixedFrontSide = this.foregroundSideSign;
    this.modelVisibility?.setOptions(this.foregroundOptions);
    this.requestRender();
    return this.foregroundSideSign;
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

  setInteractionOptions({ panSpeed = this.controls?.panSpeed ?? 0.4, inertia = 0 } = {}) {
    if (!this.controls) return null;
    const nextPanSpeed = this.THREE.MathUtils.clamp(Number(panSpeed) || 0.4, 0.1, 1);
    const nextInertia = this.THREE.MathUtils.clamp(Number(inertia) || 0, 0, 1);
    this.controls.panSpeed = nextPanSpeed;
    this.controls.staticMoving = nextInertia <= 0;
    // TrackballControlsは値が大きいほど早く停止するため、UIの慣性値を反転する。
    this.controls.dynamicDampingFactor = Math.max(0.05, 1 - nextInertia);
    this.interactionOptions = { panSpeed: nextPanSpeed, inertia: nextInertia };
    this.requestRender();
    return { ...this.interactionOptions };
  }

  getInteractionSummary() {
    return this.interactionOptions ? { ...this.interactionOptions } : null;
  }

  resize() {
    if (!this.renderer || !this.camera) return;
    const rect = this.container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.controls?.handleResize?.();
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.requestRender();
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
    this.requestRender();
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

  requestRender() {
    if (!this.running || this.renderFrameId !== null) return;
    this.renderFrameId = requestAnimationFrame(() => this.renderFrame());
  }

  renderFrame() {
    this.renderFrameId = null;
    if (!this.running) return;
    if (this.controls) this.controls.update();
    this.updateValidationTargetPlaneOcclusion();
    this.updateCameraClipping();
    const useOit = this.foregroundOptions.enabled
      && this.foregroundOptions.frontAlpha > 0
      && this.foregroundOptions.frontAlpha < 1
      && this.weightedOit?.supported
      && this.objModel;
    if (useOit) {
      const frontAlpha = this.foregroundOptions.frontAlpha;
      this.modelVisibility.setRuntimeFrontAlpha(0);
      try {
        this.renderer.render(this.scene, this.camera);
      } finally {
        this.modelVisibility.setRuntimeFrontAlpha(frontAlpha);
      }
      this.weightedOit.render(this.objModel, {
        frontAlpha,
        fixedFrontSide: this.foregroundSideSign,
        planeCenter: this.modelVisibility.planeCenter.toArray(),
        planeNormal: this.modelVisibility.planeNormal.toArray(),
        planeU: this.modelVisibility.planeU.toArray(),
        planeV: this.modelVisibility.planeV.toArray(),
        planeHalfSize: this.modelVisibility.planeHalfSize.toArray(),
      });
    } else {
      this.renderer.render(this.scene, this.camera);
    }
    this.renderValidationOverlay();
    this.renderCount += 1;
    this.renderer.domElement.dataset.renderCount = String(this.renderCount);
    // TrackballControlsはpointermove自体では再描画を要求しない。
    // 慣性OFFでもドラッグ中は更新し、慣性ONでは指を離した後も
    // TrackballControlsの減衰が完了するまで描画を継続する。
    if (!this.controlsInteracting && this.inertiaFramesRemaining > 0) {
      this.inertiaFramesRemaining -= 1;
    }
    if (this.controlsInteracting || this.inertiaFramesRemaining > 0) this.requestRender();
  }

  renderValidationOverlay() {
    if (!this.validationOverlayScene?.children.length) return;
    const originalAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = false;
    try {
      this.renderer.render(this.validationOverlayScene, this.camera);
    } finally {
      this.renderer.autoClear = originalAutoClear;
    }
  }

  updateValidationTargetPlaneOcclusion() {
    if (!this.validationTargetMarker || !this.camera) return;
    const planeInfo = this.mprPlaneOverlay?.getPlaneInfo?.();
    if (this.validationTargetMarker.userData.usesMprSphereSection) {
      const radius = this.validationTargetMarker.userData.validationRadius ?? 0;
      const sectionScale = calculateValidationSphereSectionScale(
        signedPointToPlaneDistance(this.getValidationTargetWorldPosition(), planeInfo),
        radius,
      );
      this.validationTargetMarker.scale.setScalar(sectionScale);
      // 可視光モデルとの前後関係はdepthTestで球面ごとに反映する。
      this.validationTargetMarker.visible = sectionScale > 0.001;
      return;
    }
    const modelIsInFront = this.isModelInFrontOfValidationTarget();
    const planeIsInFront = isPointBehindDisplayPlane(
      this.camera.position.toArray(),
      this.getValidationTargetWorldPosition(),
      planeInfo,
    );
    const isOccluded = planeIsInFront || modelIsInFront;
    // 通常時は陰影付きの不透明球を表示し、
    // モデルまたはMPRに隠れた場合は完全に非表示にする。
    this.validationTargetMarker.visible = !isOccluded;
  }

  isModelInFrontOfValidationTarget() {
    if (!this.objModel || !this.validationTargetMarker || !this.camera) return false;
    const markerPosition = new this.THREE.Vector3(...this.getValidationTargetWorldPosition());
    const direction = markerPosition.clone().sub(this.camera.position);
    const targetDistance = direction.length();
    const radius = this.validationTargetMarker.userData.validationRadius ?? 0;
    if (targetDistance <= Math.max(1e-4, radius)) return false;
    direction.normalize();
    // 正解点がモデル表面にある場合の自己交差だけを除外する。
    // 球半径の大部分を除外すると、表面直下へ入った球まで前面扱いになる。
    const surfaceBias = Math.max(0.15, Math.min(1, radius * 0.08));
    const raycaster = new this.THREE.Raycaster(
      this.camera.position,
      direction,
      0,
      Math.max(0, targetDistance - surfaceBias),
    );
    this.objModel.updateMatrixWorld(true);
    return raycaster
      .intersectObject(this.objModel, true)
      .some((intersection) => !this.isModelPointTransparentAtMpr(intersection.point));
  }

  getValidationTargetWorldPosition() {
    if (!this.validationTargetMarker) return [0, 0, 0];
    this.validationTargetMarker.updateMatrixWorld(true);
    return this.validationTargetMarker.getWorldPosition(new this.THREE.Vector3()).toArray();
  }

  isModelPointTransparentAtMpr(point) {
    const visibility = this.modelVisibility;
    if (!visibility?.enabled || !this.camera) return false;
    const relative = point.clone().sub(visibility.planeCenter);
    const viewRay = point.clone().sub(this.camera.position);
    const denominator = viewRay.dot(visibility.planeNormal);
    if (Math.abs(denominator) <= 1e-6) return false;
    const rayT = visibility.planeCenter.clone().sub(this.camera.position).dot(visibility.planeNormal) / denominator;
    if (rayT <= 0) return false;
    const planeIntersection = this.camera.position.clone().addScaledVector(viewRay, rayT);
    const projected = planeIntersection.sub(visibility.planeCenter);
    const insideFov = Math.abs(projected.dot(visibility.planeU)) <= visibility.planeHalfSize.x
      && Math.abs(projected.dot(visibility.planeV)) <= visibility.planeHalfSize.y;
    if (!insideFov) return false;

    const cameraDistance = this.camera.position.clone().sub(visibility.planeCenter).dot(visibility.planeNormal);
    const frontSide = Math.abs(visibility.fixedFrontSide) > 0.5
      ? visibility.fixedFrontSide
      : (cameraDistance < 0 ? -1 : 1);
    const signedDistance = relative.dot(visibility.planeNormal);
    if (signedDistance * frontSide <= 0) return false;

    const distanceFromPlane = Math.abs(signedDistance);
    const foregroundDistance = Math.max(0, visibility.foregroundDistance);
    const fadeWidth = Math.max(1e-4, visibility.fadeWidth);
    const frontMask = foregroundDistance <= 1e-4
      ? 1
      : 1 - smoothstep(foregroundDistance, foregroundDistance + fadeWidth, distanceFromPlane);
    const alpha = 1 + (visibility.frontAlpha - 1) * frontMask;
    // 可視光モデルが少しでも見えている交点は遮蔽物として扱う。
    // マーカーを透過させるのは、MPRで完全切断（Alpha 0）された場所だけ。
    return alpha <= 0.001;
  }

  useInteractivePixelRatio() {
    if (!this.renderer || this.currentPixelRatio === this.interactivePixelRatio) return;
    this.currentPixelRatio = this.interactivePixelRatio;
    this.renderer.setPixelRatio(this.currentPixelRatio);
    this.renderer.domElement.dataset.pixelRatio = String(this.currentPixelRatio);
  }

  scheduleIdlePixelRatio() {
    if (this.qualityRestoreTimer !== null) clearTimeout(this.qualityRestoreTimer);
    this.qualityRestoreTimer = setTimeout(() => {
      this.qualityRestoreTimer = null;
      if (this.controlsInteracting || !this.renderer || this.currentPixelRatio === this.idlePixelRatio) return;
      this.currentPixelRatio = this.idlePixelRatio;
      this.renderer.setPixelRatio(this.currentPixelRatio);
      this.renderer.domElement.dataset.pixelRatio = String(this.currentPixelRatio);
      this.requestRender();
    }, this.config.qualityRestoreDelayMs ?? 180);
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
    return {
      width,
      height,
      pixelRatio: this.renderer.getPixelRatio(),
      renderScheduled: this.renderFrameId !== null,
      renderCount: this.renderCount,
      sampleSize: size,
      nonBlack,
      min,
      max,
    };
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
    const summary = this.modelVisibility?.getSummary?.() ?? null;
    return summary ? {
      ...summary,
      requestedMode: "oit",
      resolvedMode: this.weightedOit?.supported === true ? "oit" : "alphaHash",
      oitSupported: this.weightedOit?.supported === true,
      oitActive: this.foregroundOptions?.enabled === true
        && this.foregroundOptions?.frontAlpha > 0
        && this.foregroundOptions?.frontAlpha < 1
        && this.weightedOit?.supported === true
        && Boolean(this.objModel),
    } : null;
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

export function calculateFocusAlignedCameraPose(cameraPosition, currentTarget, nextTarget) {
  const delta = nextTarget.map((value, index) => value - currentTarget[index]);
  return {
    camera: cameraPosition.map((value, index) => value + delta[index]),
    target: [...nextTarget],
  };
}

export function calculateValidationSphereSectionScale(distanceFromPlane, sphereRadius) {
  const radius = Number(sphereRadius);
  const distance = Math.abs(Number(distanceFromPlane));
  if (!(radius > 0) || !Number.isFinite(distance) || distance >= radius) return 0;
  const normalizedDistance = distance / radius;
  return Math.sqrt(Math.max(0, 1 - normalizedDistance * normalizedDistance));
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

export function isPointBehindDisplayPlane(cameraPosition, pointPosition, planeInfo, epsilon = 1e-4) {
  if (!planeInfo?.visible || !Array.isArray(cameraPosition) || !Array.isArray(pointPosition)) return false;
  const center = planeInfo.centerThree;
  const normal = planeInfo.nThree;
  const u = planeInfo.uThree;
  const v = planeInfo.vThree;
  const width = Number(planeInfo.widthMm);
  const height = Number(planeInfo.heightMm);
  if (![center, normal, u, v].every((value) => Array.isArray(value) && value.length === 3)) return false;
  if (!(width > 0) || !(height > 0)) return false;

  const direction = pointPosition.map((value, index) => value - cameraPosition[index]);
  const cameraToPlane = center.map((value, index) => value - cameraPosition[index]);
  const denominator = dot3(direction, normal);
  if (Math.abs(denominator) <= epsilon) return false;
  const intersectionRatio = dot3(cameraToPlane, normal) / denominator;
  if (intersectionRatio <= epsilon || intersectionRatio >= 1 - epsilon) return false;

  const intersectionOffset = direction.map((value, index) =>
    cameraPosition[index] + value * intersectionRatio - center[index]
  );
  return Math.abs(dot3(intersectionOffset, u)) <= width / 2 + epsilon
    && Math.abs(dot3(intersectionOffset, v)) <= height / 2 + epsilon;
}

function signedPointToPlaneDistance(pointPosition, planeInfo) {
  const center = planeInfo?.centerThree;
  const normal = planeInfo?.nThree;
  if (![pointPosition, center, normal].every((value) => Array.isArray(value) && value.length === 3)) return 0;
  return pointPosition.reduce(
    (sum, value, index) => sum + (value - center[index]) * normal[index],
    0,
  );
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function smoothstep(edge0, edge1, value) {
  const range = edge1 - edge0;
  if (range <= 0) return value < edge0 ? 0 : 1;
  const t = Math.max(0, Math.min(1, (value - edge0) / range));
  return t * t * (3 - 2 * t);
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
