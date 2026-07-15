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

  updatePlane({ centerThree, nThree } = {}) {
    if (centerThree) this.planeCenter.set(...centerThree);
    if (nThree) this.planeNormal.set(...nThree).normalize();
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
        "  float d = dot(vMprWorldPosition - mprPlaneCenter, mprN);",
        "  float cameraD = dot(cameraPosition - mprPlaneCenter, mprN);",
        "  if (d * cameraD > 0.0) {",
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
      shader.uniforms.mprFrontAlpha.value = this.frontAlpha;
      shader.uniforms.mprForegroundDistance.value = this.foregroundDistance;
      shader.uniforms.mprFadeWidth.value = this.fadeWidth;
    }
  }

  updateMaterialRenderState(record) {
    const { material } = record;
    const useHashedForeground = this.enabled && material.opacity >= 1 && !record.originalTransparent;
    // GLB全体をtransparent/depthWrite=falseにすると、単一mesh内の三角形はソートされず
    // 内部面が外表面より前に合成される。Alpha Hashならdiscard後も通常の深度を書ける。
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
      foregroundDistance: this.foregroundDistance,
      fadeWidth: this.fadeWidth,
      materialCount: this.records.size,
      shaderCount,
      vertexPatched,
      fragmentPatched,
      planeCenter: [this.planeCenter.x, this.planeCenter.y, this.planeCenter.z],
      planeNormal: [this.planeNormal.x, this.planeNormal.y, this.planeNormal.z],
    };
  }
}
