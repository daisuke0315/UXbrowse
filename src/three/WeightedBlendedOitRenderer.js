export class WeightedBlendedOitRenderer {
  constructor(THREE, renderer, scene, camera) {
    this.THREE = THREE;
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.materialCache = new Map();
    this.size = new THREE.Vector2();
    this.supported = renderer.capabilities.isWebGL2
      && renderer.extensions.has("EXT_color_buffer_float")
      && renderer.extensions.has("EXT_float_blend");
    if (!this.supported) return;
    this.accumTarget = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.revealTarget = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.compositeScene = new THREE.Scene();
    this.compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.compositeMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: true,
      uniforms: {
        accumTexture: { value: this.accumTarget.texture },
        revealTexture: { value: this.revealTarget.texture },
      },
      vertexShader: "varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.0,1.0); }",
      fragmentShader: [
        "uniform sampler2D accumTexture;",
        "uniform sampler2D revealTexture;",
        "varying vec2 vUv;",
        "void main(){",
        "  vec4 accum=texture2D(accumTexture,vUv);",
        "  float reveal=texture2D(revealTexture,vUv).r;",
        "  float alpha=clamp(1.0-reveal,0.0,1.0);",
        "  vec3 color=accum.rgb/max(accum.a,0.00001);",
        "  gl_FragColor=vec4(color,alpha);",
        "  #include <tonemapping_fragment>",
        "  #include <colorspace_fragment>",
        "}",
      ].join("\n"),
    });
    this.compositeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial));
  }

  render(model, options = {}) {
    if (!this.supported || !model || !(options.frontAlpha > 0 && options.frontAlpha < 1)) return false;
    this.ensureSize();
    const modelObjects = new Set();
    model.traverse((object) => modelObjects.add(object));
    const hidden = [];
    this.scene.traverse((object) => {
      if (object === this.scene || object.isLight || modelObjects.has(object) || !object.visible) return;
      hidden.push(object);
      object.visible = false;
    });
    const originalBackground = this.scene.background;
    const originalAutoClear = this.renderer.autoClear;
    const originalClearColor = this.renderer.getClearColor(new this.THREE.Color()).clone();
    const originalClearAlpha = this.renderer.getClearAlpha();
    this.scene.background = null;
    try {
      this.renderPass(model, "accum", this.accumTarget, options, 0x000000, 0);
      this.renderPass(model, "reveal", this.revealTarget, options, 0xffffff, 1);
      this.renderer.setRenderTarget(null);
      this.renderer.autoClear = false;
      this.renderer.render(this.compositeScene, this.compositeCamera);
    } finally {
      this.renderer.setRenderTarget(null);
      this.renderer.autoClear = originalAutoClear;
      this.renderer.setClearColor(originalClearColor, originalClearAlpha);
      this.scene.background = originalBackground;
      for (const object of hidden) object.visible = true;
    }
    return true;
  }

  renderPass(model, pass, target, options, clearColor, clearAlpha) {
    const swaps = [];
    model.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const originals = Array.isArray(child.material) ? child.material : [child.material];
      const replacements = originals.map((material) => this.getPassMaterial(material, pass, options));
      swaps.push([child, child.material]);
      child.material = Array.isArray(child.material) ? replacements : replacements[0];
    });
    this.renderer.setRenderTarget(target);
    this.renderer.setClearColor(clearColor, clearAlpha);
    this.renderer.clear(true, false, false);
    try {
      this.renderer.render(this.scene, this.camera);
    } finally {
      for (const [child, material] of swaps) child.material = material;
    }
  }

  getPassMaterial(source, pass, options) {
    let record = this.materialCache.get(source);
    if (!record) {
      record = {
        accum: this.createPassMaterial(source, "accum"),
        reveal: this.createPassMaterial(source, "reveal"),
      };
      this.materialCache.set(source, record);
    }
    const material = record[pass];
    material.userData.oitUniforms.frontAlpha.value = options.frontAlpha;
    material.userData.oitUniforms.fixedFrontSide.value = options.fixedFrontSide ?? 0;
    material.userData.oitUniforms.planeCenter.value.set(...options.planeCenter);
    material.userData.oitUniforms.planeNormal.value.set(...options.planeNormal);
    material.userData.oitUniforms.planeU.value.set(...options.planeU);
    material.userData.oitUniforms.planeV.value.set(...options.planeV);
    material.userData.oitUniforms.planeHalfSize.value.set(...options.planeHalfSize);
    return material;
  }

  createPassMaterial(source, pass) {
    const THREE = this.THREE;
    const material = source.clone();
    material.alphaHash = false;
    material.transparent = true;
    material.depthTest = false;
    material.depthWrite = false;
    material.blending = THREE.CustomBlending;
    material.blendEquation = THREE.AddEquation;
    if (pass === "accum") {
      material.blendSrc = THREE.OneFactor;
      material.blendDst = THREE.OneFactor;
    } else {
      material.blendSrc = THREE.ZeroFactor;
      material.blendDst = THREE.OneMinusSrcAlphaFactor;
    }
    const uniforms = {
      frontAlpha: { value: 0.5 },
      fixedFrontSide: { value: 0 },
      planeCenter: { value: new THREE.Vector3() },
      planeNormal: { value: new THREE.Vector3(0, 0, 1) },
      planeU: { value: new THREE.Vector3(1, 0, 0) },
      planeV: { value: new THREE.Vector3(0, 1, 0) },
      planeHalfSize: { value: new THREE.Vector2(1, 1) },
    };
    material.userData.oitUniforms = uniforms;
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vOitWorldPosition;")
        .replace("#include <worldpos_vertex>", [
          "vec4 oitWorldPosition=vec4(transformed,1.0);",
          "#ifdef USE_BATCHING",
          "oitWorldPosition=batchingMatrix*oitWorldPosition;",
          "#endif",
          "#ifdef USE_INSTANCING",
          "oitWorldPosition=instanceMatrix*oitWorldPosition;",
          "#endif",
          "oitWorldPosition=modelMatrix*oitWorldPosition;",
          "vOitWorldPosition=oitWorldPosition.xyz;",
          "#include <worldpos_vertex>",
        ].join("\n"));
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", [
          "#include <common>",
          "varying vec3 vOitWorldPosition;",
          "uniform float frontAlpha;",
          "uniform float fixedFrontSide;",
          "uniform vec3 planeCenter;",
          "uniform vec3 planeNormal;",
          "uniform vec3 planeU;",
          "uniform vec3 planeV;",
          "uniform vec2 planeHalfSize;",
        ].join("\n"))
        .replace("#include <opaque_fragment>", this.getOitFragmentCode(pass))
        // Accum/reveal targets must retain linear values. Display transforms are
        // applied once by the fullscreen composite pass.
        .replace("#include <tonemapping_fragment>", "")
        .replace("#include <colorspace_fragment>", "")
        .replace("#include <fog_fragment>", "")
        .replace("#include <premultiplied_alpha_fragment>", "")
        .replace("#include <dithering_fragment>", "");
    };
    material.needsUpdate = true;
    return material;
  }

  getOitFragmentCode(pass) {
    const common = [
      "vec3 oitN=normalize(planeNormal);",
      "vec3 oitRay=vOitWorldPosition-cameraPosition;",
      "float oitDenom=dot(oitRay,oitN);",
      "if(abs(oitDenom)<=0.000001) discard;",
      "float oitT=dot(planeCenter-cameraPosition,oitN)/oitDenom;",
      "vec3 oitHit=cameraPosition+oitRay*oitT-planeCenter;",
      "bool oitInside=oitT>0.0 && abs(dot(oitHit,normalize(planeU)))<=planeHalfSize.x && abs(dot(oitHit,normalize(planeV)))<=planeHalfSize.y;",
      "float oitD=dot(vOitWorldPosition-planeCenter,oitN);",
      "float oitCameraD=dot(cameraPosition-planeCenter,oitN);",
      "float oitFrontSide=abs(fixedFrontSide)>0.5?fixedFrontSide:(oitCameraD<0.0?-1.0:1.0);",
      "if(!oitInside || oitD*oitFrontSide<=0.0) discard;",
      "float oitAlpha=clamp(frontAlpha*diffuseColor.a,0.0,1.0);",
      "float oitWeight=max(0.01,oitAlpha);",
    ];
    if (pass === "accum") common.push("gl_FragColor=vec4(outgoingLight*oitAlpha*oitWeight,oitAlpha*oitWeight);");
    else common.push("gl_FragColor=vec4(0.0,0.0,0.0,oitAlpha);");
    return common.join("\n");
  }

  ensureSize() {
    this.renderer.getDrawingBufferSize(this.size);
    const width = Math.max(1, Math.floor(this.size.x));
    const height = Math.max(1, Math.floor(this.size.y));
    if (this.accumTarget.width === width && this.accumTarget.height === height) return;
    this.accumTarget.setSize(width, height);
    this.revealTarget.setSize(width, height);
  }

  dispose() {
    for (const record of this.materialCache.values()) {
      record.accum.dispose();
      record.reveal.dispose();
    }
    this.materialCache.clear();
    this.accumTarget?.dispose();
    this.revealTarget?.dispose();
    this.compositeMaterial?.dispose();
    this.compositeScene?.traverse((object) => object.geometry?.dispose?.());
  }
}
