export class MprPlaneOverlay {
  constructor(THREE, scene) {
    this.THREE = THREE;
    this.scene = scene;
    this.visible = true;
    this.opacity = 1;
    this.showBoundary = true;
  }

  update({ canvas, result, request, mapper, camera = null, visible = true, opacity = 1, showBoundary = true, displayMode = "world", cameraPlaneDistance = 0, flipU = false, flipV = false, thresholdAlphaEnabled = false, transparentThreshold = -500 }) {
    if (!canvas || !result || !request || !mapper) return null;
    this.visible = visible;
    this.opacity = opacity;
    this.showBoundary = showBoundary;
    this.ensureObjects(canvas);
    this.updateTextureImage(result, { thresholdAlphaEnabled, transparentThreshold });
    const sourcePose = this.computeSourcePose(result.plane, mapper);
    const displayPose = displayMode === "camera" && camera
      ? this.computeCameraPose(sourcePose, camera, cameraPlaneDistance)
      : { ...sourcePose, displayMode: "world", widthScale: 1, depth: null };
    this.updateGeometry(result.width * request.spacingU * displayPose.widthScale, result.height * request.spacingV * displayPose.widthScale, { flipU, flipV });
    this.applyPose(displayPose);
    this.lastPlaneInfo = this.createPlaneInfo(sourcePose, displayPose);
    this.texture.needsUpdate = true;
    this.mesh.visible = visible;
    this.material.opacity = opacity;
    this.material.depthTest = displayPose.displayMode !== "camera";
    this.boundary.visible = visible && showBoundary;
    this.boundary.material.depthTest = displayPose.displayMode !== "camera";
    return this.getPlaneInfo();
  }

  setOptions({ visible = this.visible, opacity = this.opacity, showBoundary = this.showBoundary } = {}) {
    this.visible = visible;
    this.opacity = opacity;
    this.showBoundary = showBoundary;
    if (this.mesh) {
      this.mesh.visible = visible;
      this.material.opacity = opacity;
    }
    if (this.boundary) this.boundary.visible = visible && showBoundary;
  }

  ensureObjects(canvas) {
    if (this.mesh) return;
    const THREE = this.THREE;
    this.textureCanvas = document.createElement("canvas");
    this.textureCanvas.width = canvas.width;
    this.textureCanvas.height = canvas.height;
    this.textureContext = this.textureCanvas.getContext("2d");
    this.texture = new THREE.CanvasTexture(this.textureCanvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.needsUpdate = true;
    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      opacity: this.opacity,
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    this.geometry = new THREE.PlaneGeometry(1, 1);
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = "mpr-plane-overlay";
    this.mesh.renderOrder = 2;
    this.scene.add(this.mesh);

    const boundaryGeometry = new THREE.BufferGeometry();
    const boundaryMaterial = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.95, depthTest: true, depthWrite: false });
    this.boundary = new THREE.LineLoop(boundaryGeometry, boundaryMaterial);
    this.boundary.name = "mpr-plane-boundary";
    this.boundary.renderOrder = 3;
    this.scene.add(this.boundary);
  }

  updateTextureImage(result, { thresholdAlphaEnabled, transparentThreshold }) {
    if (this.textureCanvas.width !== result.width) this.textureCanvas.width = result.width;
    if (this.textureCanvas.height !== result.height) this.textureCanvas.height = result.height;
    this.textureContext = this.textureCanvas.getContext("2d");
    const rgba = new Uint8ClampedArray(result.rgba);
    let transparentCount = 0;
    if (thresholdAlphaEnabled) {
      for (let i = 0; i < result.pixels.length; i += 1) {
        if (result.pixels[i] <= transparentThreshold) {
          rgba[i * 4 + 3] = 0;
          transparentCount += 1;
        }
      }
    }
    this.textureContext.putImageData(new ImageData(rgba, result.width, result.height), 0, 0);
    this.lastThresholdInfo = {
      thresholdAlphaEnabled,
      transparentThreshold,
      transparentCount,
      totalPixels: result.pixels.length,
      transparentRatio: result.pixels.length > 0 ? transparentCount / result.pixels.length : 0,
    };
  }

  updateGeometry(widthMm, heightMm, { flipU = false, flipV = false } = {}) {
    const THREE = this.THREE;
    const geometryChanged = this.widthMm !== widthMm || this.heightMm !== heightMm;
    if (geometryChanged) {
      this.widthMm = widthMm;
      this.heightMm = heightMm;
      this.geometry?.dispose();
      this.geometry = new THREE.PlaneGeometry(widthMm, heightMm);
      this.mesh.geometry = this.geometry;
      this.baseUv = Array.from(this.geometry.attributes.uv.array);
    }
    if (geometryChanged || this.flipU !== flipU || this.flipV !== flipV) {
      this.flipU = flipU;
      this.flipV = flipV;
      this.updateTextureUv(flipU, flipV);
    }
    if (!geometryChanged) return;
    const halfW = widthMm / 2;
    const halfH = heightMm / 2;
    this.boundary.geometry.dispose();
    this.boundary.geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-halfW, -halfH, 0),
      new THREE.Vector3(halfW, -halfH, 0),
      new THREE.Vector3(halfW, halfH, 0),
      new THREE.Vector3(-halfW, halfH, 0),
    ]);
  }

  updateTextureUv(flipU, flipV) {
    const uv = this.geometry.attributes.uv;
    for (let i = 0; i < uv.count; i += 1) {
      const u = this.baseUv[i * 2];
      const v = this.baseUv[i * 2 + 1];
      uv.setXY(i, flipU ? 1 - u : u, flipV ? 1 - v : v);
    }
    uv.needsUpdate = true;
  }

  computeSourcePose(plane, mapper) {
    const THREE = this.THREE;
    const centerThree = mapper.dicomPointToThree(plane.center);
    const uThree = mapper.dicomVectorToThree(plane.u);
    const vThree = mapper.dicomVectorToThree(plane.v);
    const nThree = mapper.dicomVectorToThree(plane.n);
    return {
      center: new THREE.Vector3(...centerThree),
      u: new THREE.Vector3(...uThree).normalize(),
      v: new THREE.Vector3(...vThree).normalize(),
      n: new THREE.Vector3(...nThree).normalize(),
      centerVolume: [...plane.center],
      nVolume: [...plane.n],
    };
  }

  computeCameraPose(sourcePose, camera, requestedDistance) {
    const THREE = this.THREE;
    camera.updateMatrixWorld();
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    const cameraPosition = new THREE.Vector3();
    camera.getWorldPosition(cameraPosition);
    const cameraForward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize();
    const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
    const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize();
    const sourceLocal = sourcePose.center.clone().applyMatrix4(camera.matrixWorldInverse);
    const sourceDepth = Math.max(0.001, -sourceLocal.z);
    const depth = requestedDistance > 0 ? requestedDistance : sourceDepth;
    const widthScale = depth / sourceDepth;
    const displayLocal = new THREE.Vector3(sourceLocal.x * widthScale, sourceLocal.y * widthScale, -depth);
    const displayCenter = displayLocal.applyMatrix4(camera.matrixWorld);
    return {
      center: displayCenter,
      u: cameraRight,
      v: cameraUp,
      n: cameraForward,
      displayMode: "camera",
      widthScale,
      depth,
    };
  }

  applyPose(pose) {
    const THREE = this.THREE;
    const basis = new THREE.Matrix4().makeBasis(
      pose.u,
      pose.v,
      pose.n
    );
    this.mesh.position.copy(pose.center);
    this.mesh.setRotationFromMatrix(basis);
    this.boundary.position.copy(this.mesh.position);
    this.boundary.quaternion.copy(this.mesh.quaternion);
  }

  createPlaneInfo(sourcePose, displayPose) {
    return {
      centerThree: displayPose.center.toArray(),
      uThree: displayPose.u.toArray(),
      vThree: displayPose.v.toArray(),
      nThree: displayPose.n.toArray(),
      sourceCenterThree: sourcePose.center.toArray(),
      sourceUThree: sourcePose.u.toArray(),
      sourceVThree: sourcePose.v.toArray(),
      sourceNThree: sourcePose.n.toArray(),
      centerVolume: sourcePose.centerVolume,
      nVolume: sourcePose.nVolume,
      displayMode: displayPose.displayMode,
      cameraPlaneDepth: displayPose.depth,
      cameraPlaneScale: displayPose.widthScale,
    };
  }

  getPlaneInfo() {
    if (!this.lastPlaneInfo) return null;
    return {
      ...this.lastPlaneInfo,
      visible: this.visible,
      opacity: this.opacity,
      showBoundary: this.showBoundary,
      thresholdAlpha: this.lastThresholdInfo ?? null,
      flipU: Boolean(this.flipU),
      flipV: Boolean(this.flipV),
      widthMm: this.widthMm,
      heightMm: this.heightMm,
    };
  }

  dispose() {
    if (this.mesh) this.scene.remove(this.mesh);
    if (this.boundary) this.scene.remove(this.boundary);
    this.geometry?.dispose();
    this.material?.dispose();
    this.texture?.dispose();
    this.boundary?.geometry?.dispose();
    this.boundary?.material?.dispose();
    this.mesh = null;
    this.boundary = null;
  }
}
