export class ModelVisibilityShaderController {
  constructor(THREE) {
    this.THREE = THREE;
    this.records = new Map();
    this.enabled = false;
    this.frontAlpha = 0.15;
    this.foregroundDistance = 0;
    this.fadeWidth = 10;
    this.planeCenter = new THREE.Vector3();
    this.planeNormal = new THREE.Vector3(0, 0, 1);
    this.planeU = new THREE.Vector3(1, 0, 0);
    this.planeV = new THREE.Vector3(0, 1, 0);
    this.planeHalfSize = new THREE.Vector2(1e20, 1e20);
  }

  setOptions({ enabled = this.enabled, frontAlpha = this.frontAlpha, foregroundDistance = this.foregroundDistance, fadeWidth = this.fadeWidth } = {}) {
    this.enabled = enabled;
    this.frontAlpha = frontAlpha;
    this.foregroundDistance = foregroundDistance;
    this.fadeWidth = fadeWidth;
    for (const record of this.records.values()) {
      this.updateMaterialRenderState(record);
      this.updateRecordUniforms(record);
    }
  }

  updatePlane({ centerThree, uThree, vThree, nThree, widthMm, heightMm } = {}) {
    if (centerThree) this.planeCenter.set(...centerThree);
    if (uThree) this.planeU.set(...uThree).normalize();
    if (vThree) this.planeV.set(...vThree).normalize();
    if (nThree) this.planeNormal.set(...nThree).normalize();
    if (Number.isFinite(widthMm) && Number.isFinite(heightMm)) {
      this.planeHalfSize.set(Math.max(0, widthMm / 2), Math.max(0, heightMm / 2));
    }
    for (const record of this.records.values()) this.updateRecordUniforms(record);
  }

  applyToObject(object, options = {}) {
    if (options) this.setOptions(options);
    if (!object) return;
    object.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) this.patchMaterial(material);
    });
  }

  patchMaterial(material) {
    if (this.records.has(material)) return;
    const record = {
      material,
      originalOnBeforeCompile: material.onBeforeCompile,
      originalTransparent: material.transparent,
      originalDepthWrite: material.depthWrite,
      originalAlphaHash: material.alphaHash === true,
      shaders: [],
      patchStatus: [],
    };
    this.records.set(material, record);
    this.updateMaterialRenderState(record);
    material.onBeforeCompile = (shader) => {
      record.originalOnBeforeCompile?.(shader);
      shader.uniforms.mprClipEnabled = { value: this.enabled };
      shader.uniforms.mprPlaneCenter = { value: this.planeCenter.clone() };
      shader.uniforms.mprPlaneNormal = { value: this.planeNormal.clone() };
      shader.uniforms.mprPlaneU = { value: this.planeU.clone() };
      shader.uniforms.mprPlaneV = { value: this.planeV.clone() };
      shader.uniforms.mprPlaneHalfSize = { value: this.planeHalfSize.clone() };
      shader.uniforms.mprFrontAlpha = { value: this.frontAlpha };
      shader.uniforms.mprForegroundDistance = { value: this.foregroundDistance };
      shader.uniforms.mprFadeWidth = { value: this.fadeWidth };
      const status = { vertexWorldPosition: false, fragmentAlpha: false };
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vMprWorldPosition;")
        .replace("#include <worldpos_vertex>", () => {
          status.vertexWorldPosition = true;
          return [
            "vec4 mprWorldPosition = vec4( transformed, 1.0 );",
            "#ifdef USE_BATCHING",
            "  mprWorldPosition = batchingMatrix * mprWorldPosition;",
            "#endif",
            "#ifdef USE_INSTANCING",
            "  mprWorldPosition = instanceMatrix * mprWorldPosition;",
            "#endif",
            "mprWorldPosition = modelMatrix * mprWorldPosition;",
            "vMprWorldPosition = mprWorldPosition.xyz;",
            "#include <worldpos_vertex>",
          ].join("\n");
        });
      const foregroundAlphaCode = [
        "float mprForegroundAlphaFactor = 1.0;",
        "if (mprClipEnabled) {",
        "  vec3 mprN = normalize(mprPlaneNormal);",
        "  vec3 mprRelative = vMprWorldPosition - mprPlaneCenter;",
        "  vec3 mprViewRay = vMprWorldPosition - cameraPosition;",
        "  float mprRayDenominator = dot(mprViewRay, mprN);",
        "  bool mprRayHitsPlane = abs(mprRayDenominator) > 0.000001;",
        "  float mprRayT = mprRayHitsPlane ? dot(mprPlaneCenter - cameraPosition, mprN) / mprRayDenominator : -1.0;",
        "  vec3 mprPlaneIntersection = cameraPosition + mprViewRay * mprRayT;",
        "  vec3 mprProjected = mprPlaneIntersection - mprPlaneCenter;",
        "  float mprU = abs(dot(mprProjected, normalize(mprPlaneU)));",
        "  float mprV = abs(dot(mprProjected, normalize(mprPlaneV)));",
        "  bool insideMprFov = mprRayHitsPlane && mprRayT > 0.0 && mprU <= mprPlaneHalfSize.x && mprV <= mprPlaneHalfSize.y;",
        "  float d = dot(mprRelative, mprN);",
        "  float cameraD = dot(cameraPosition - mprPlaneCenter, mprN);",
        "  if (insideMprFov && d * cameraD > 0.0) {",
        "    float distanceFromPlane = abs(d);",
        "    float frontMask = mprForegroundDistance <= 0.0001 ? 1.0 : 1.0 - smoothstep(mprForegroundDistance, mprForegroundDistance + max(mprFadeWidth, 0.0001), distanceFromPlane);",
        "    float foregroundAlpha = mix(1.0, mprFrontAlpha, frontMask);",
        "    mprForegroundAlphaFactor = foregroundAlpha;",
        "  }",
        "}",
      ].join("\n");
      const applyForegroundAlphaCode = [
        "diffuseColor.a *= clamp(mprForegroundAlphaFactor, 0.0, 1.0);",
        "#include <alphahash_fragment>",
      ].join("\n");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", [
          "#include <common>",
          "varying vec3 vMprWorldPosition;",
          "uniform bool mprClipEnabled;",
          "uniform vec3 mprPlaneCenter;",
          "uniform vec3 mprPlaneNormal;",
          "uniform vec3 mprPlaneU;",
          "uniform vec3 mprPlaneV;",
          "uniform vec2 mprPlaneHalfSize;",
          "uniform float mprFrontAlpha;",
          "uniform float mprForegroundDistance;",
          "uniform float mprFadeWidth;",
        ].join("\n"))
        .replace("#include <alphahash_fragment>", () => {
          status.fragmentAlpha = true;
          return [
            foregroundAlphaCode,
            applyForegroundAlphaCode,
          ].join("\n");
        });
      if (!status.fragmentAlpha && shader.fragmentShader.includes("#include <opaque_fragment>")) {
        shader.fragmentShader = shader.fragmentShader.replace("#include <opaque_fragment>", () => {
          status.fragmentAlpha = true;
          return `${foregroundAlphaCode}\nif (mprForegroundAlphaFactor <= 0.0001) discard;\n#include <opaque_fragment>`;
        });
      }
      record.patchStatus.push(status);
      record.shaders.push(shader);
      this.updateRecordUniforms(record);
    };
    material.needsUpdate = true;
  }

  updateRecordUniforms(record) {
    for (const shader of record.shaders) {
      shader.uniforms.mprClipEnabled.value = this.enabled;
      shader.uniforms.mprPlaneCenter.value.copy(this.planeCenter);
      shader.uniforms.mprPlaneNormal.value.copy(this.planeNormal);
      shader.uniforms.mprPlaneU.value.copy(this.planeU);
      shader.uniforms.mprPlaneV.value.copy(this.planeV);
      shader.uniforms.mprPlaneHalfSize.value.copy(this.planeHalfSize);
      shader.uniforms.mprFrontAlpha.value = this.frontAlpha;
      shader.uniforms.mprForegroundDistance.value = this.foregroundDistance;
      shader.uniforms.mprFadeWidth.value = this.fadeWidth;
    }
  }

  updateMaterialRenderState(record) {
    const { material } = record;
    const useHashedForeground = this.enabled && material.opacity >= 1 && !record.originalTransparent;
    // 単一mesh内の三角形ソート破綻を避けるため、前景透過では深度を書けるAlpha Hashを使う。
    material.alphaHash = record.originalAlphaHash || useHashedForeground;
    material.transparent = record.originalTransparent || material.opacity < 1;
    material.depthWrite = material.transparent ? false : record.originalDepthWrite;
    material.needsUpdate = true;
  }

  restore() {
    for (const record of this.records.values()) {
      const { material } = record;
      material.onBeforeCompile = record.originalOnBeforeCompile;
      material.transparent = record.originalTransparent;
      material.depthWrite = record.originalDepthWrite;
      material.alphaHash = record.originalAlphaHash;
      material.needsUpdate = true;
    }
    this.records.clear();
  }

  dispose() {
    this.restore();
  }

  getSummary() {
    let shaderCount = 0;
    let vertexPatched = 0;
    let fragmentPatched = 0;
    for (const record of this.records.values()) {
      shaderCount += record.shaders.length;
      for (const status of record.patchStatus) {
        if (status.vertexWorldPosition) vertexPatched += 1;
        if (status.fragmentAlpha) fragmentPatched += 1;
      }
    }
    return {
      enabled: this.enabled,
      frontAlpha: this.frontAlpha,
      transparencyMode: this.enabled ? "alpha-hash" : "opaque",
      foregroundDistance: this.foregroundDistance,
      fadeWidth: this.fadeWidth,
      materialCount: this.records.size,
      shaderCount,
      vertexPatched,
      fragmentPatched,
      planeCenter: [this.planeCenter.x, this.planeCenter.y, this.planeCenter.z],
      planeNormal: [this.planeNormal.x, this.planeNormal.y, this.planeNormal.z],
      planeU: [this.planeU.x, this.planeU.y, this.planeU.z],
      planeV: [this.planeV.x, this.planeV.y, this.planeV.z],
      planeHalfSize: [this.planeHalfSize.x, this.planeHalfSize.y],
    };
  }
}
