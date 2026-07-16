import { DicomLoader } from "./DicomLoader.js";
import { MhaLoader } from "./MhaLoader.js";
import { VolumeBuilder } from "./VolumeBuilder.js";
import { MprPlaneCalculator } from "./MprPlaneCalculator.js";
import { MprRenderer } from "./MprRenderer.js?v=performance-pass-1";
import { ValidationLogger } from "./ValidationLogger.js";
import { formatVector } from "./math3d.js";
import { volumeCenter } from "./VolumeGeometry.js";
import { viewerConfig } from "./config/viewerConfig.js?v=interaction-defaults-1";
import { ThreeSceneManager } from "./three/ThreeSceneManager.js?v=restore-alpha-hash-1";
import { ThreeCameraController } from "./three/ThreeCameraController.js";
import { ThreeDicomCoordinateMapper } from "./three/ThreeDicomCoordinateMapper.js";
import { DynamicMprController } from "./three/DynamicMprController.js";
import { MprUpdateScheduler } from "./three/MprUpdateScheduler.js";

const logger = new ValidationLogger();
const loader = new DicomLoader(logger);
const mhaLoader = new MhaLoader(logger);
const builder = new VolumeBuilder(logger);
const renderer = new MprRenderer(logger);
const planeCalculator = new MprPlaneCalculator();
const DEFAULT_MPR_DISTANCE = 600;
// MPRキャンバスは読影用の正表示を維持し、Three.js重畳面で生じる
// CanvasTexture／裏面表示のU/V差だけをtexture座標で補正する。
// The PiP monitor previews the actual overlay texture; only V is flipped by default.
const DEFAULT_MPR_UV_FLIP = [1, -1];

const state = {
  volume: null,
  result: null,
  threeScene: null,
  mapper: new ThreeDicomCoordinateMapper(viewerConfig.mapper.threeToDicomMatrix),
  cameraController: null,
  dynamicController: null,
  scheduler: null,
  threeReady: false,
  dynamicInfo: null,
  volumeAxisFlip: [1, 1, 1],
  mprUvFlip: [...DEFAULT_MPR_UV_FLIP],
  defaults: {
    eye: [0, 0, -500],
    target: [0, 0, 0],
    up: [0, 1, 0],
    wl: 40,
    ww: 400,
  },
};

const els = {
  fileInput: document.querySelector("#fileInput"),
  folderInput: document.querySelector("#folderInput"),
  mhaInput: document.querySelector("#mhaInput"),
  objInput: document.querySelector("#objInput"),
  glbInput: document.querySelector("#glbInput"),
  glbFolderInput: document.querySelector("#glbFolderInput"),
  textureInput: document.querySelector("#textureInput"),
  demoVolumeBtn: document.querySelector("#demoVolumeBtn"),
  statusText: document.querySelector("#statusText"),
  fileMenu: document.querySelector(".file-menu"),
  seriesUid: document.querySelector("#seriesUid"),
  volumeDimensions: document.querySelector("#volumeDimensions"),
  volumeSpacing: document.querySelector("#volumeSpacing"),
  huRange: document.querySelector("#huRange"),
  threeViewport: document.querySelector("#threeViewport"),
  workspace: document.querySelector(".workspace"),
  controlsToggleBtn: document.querySelector("#controlsToggleBtn"),
  canvas: document.querySelector("#mprCanvas"),
  mprPane: document.querySelector(".mpr-pane"),
  mprCollapseBtn: document.querySelector("#mprCollapseBtn"),
  mprExpandBtn: document.querySelector("#mprExpandBtn"),
  dynamicToggle: document.querySelector("#dynamicToggle"),
  updateInterval: document.querySelector("#updateInterval"),
  panSensitivity: document.querySelector("#panSensitivity"),
  panSensitivityValue: document.querySelector("#panSensitivityValue"),
  controlInertia: document.querySelector("#controlInertia"),
  controlInertiaValue: document.querySelector("#controlInertiaValue"),
  objCenterToggle: document.querySelector("#objCenterToggle"),
  objMprToggle: document.querySelector("#objMprToggle"),
  objScale: document.querySelector("#objScale"),
  objOpacity: document.querySelector("#objOpacity"),
  objOpacityValue: document.querySelector("#objOpacityValue"),
  objMprDistance: document.querySelector("#objMprDistance"),
  objPosX: document.querySelector("#objPosX"),
  objPosY: document.querySelector("#objPosY"),
  objPosZ: document.querySelector("#objPosZ"),
  objRotX: document.querySelector("#objRotX"),
  objRotY: document.querySelector("#objRotY"),
  objRotZ: document.querySelector("#objRotZ"),
  demoObjBtn: document.querySelector("#demoObjBtn"),
  clearObjBtn: document.querySelector("#clearObjBtn"),
  flipVolumeXBtn: document.querySelector("#flipVolumeXBtn"),
  flipVolumeYBtn: document.querySelector("#flipVolumeYBtn"),
  flipVolumeZBtn: document.querySelector("#flipVolumeZBtn"),
  resetVolumeFlipBtn: document.querySelector("#resetVolumeFlipBtn"),
  windowLevel: document.querySelector("#windowLevel"),
  windowWidth: document.querySelector("#windowWidth"),
  windowPreset: document.querySelector("#windowPreset"),
  mprWidth: document.querySelector("#mprWidth"),
  mprHeight: document.querySelector("#mprHeight"),
  spacingU: document.querySelector("#spacingU"),
  spacingV: document.querySelector("#spacingV"),
  normalOffset: document.querySelector("#normalOffset"),
  interpolation: document.querySelector("#interpolation"),
  renderBtn: document.querySelector("#renderBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  mprPlaneToggle: document.querySelector("#mprPlaneToggle"),
  mprFovWidth: document.querySelector("#mprFovWidth"),
  mprFovWidthValue: document.querySelector("#mprFovWidthValue"),
  mprFovHeight: document.querySelector("#mprFovHeight"),
  mprFovHeightValue: document.querySelector("#mprFovHeightValue"),
  mprPlaneOpacity: document.querySelector("#mprPlaneOpacity"),
  mprPlaneOpacityValue: document.querySelector("#mprPlaneOpacityValue"),
  mprPlaneMode: document.querySelector("#mprPlaneMode"),
  mprCameraPlaneDistance: document.querySelector("#mprCameraPlaneDistance"),
  mprFlipUBtn: document.querySelector("#mprFlipUBtn"),
  mprFlipVBtn: document.querySelector("#mprFlipVBtn"),
  mprFlipResetBtn: document.querySelector("#mprFlipResetBtn"),
  mprPlaneBoundaryToggle: document.querySelector("#mprPlaneBoundaryToggle"),
  foregroundFadeToggle: document.querySelector("#foregroundFadeToggle"),
  foregroundAlpha: document.querySelector("#foregroundAlpha"),
  foregroundAlphaValue: document.querySelector("#foregroundAlphaValue"),
  foregroundDistance: document.querySelector("#foregroundDistance"),
  foregroundDistanceValue: document.querySelector("#foregroundDistanceValue"),
  foregroundFadeWidth: document.querySelector("#foregroundFadeWidth"),
  foregroundFadeWidthValue: document.querySelector("#foregroundFadeWidthValue"),
  mprThresholdAlphaToggle: document.querySelector("#mprThresholdAlphaToggle"),
  mprTransparentThreshold: document.querySelector("#mprTransparentThreshold"),
  axialBtn: document.querySelector("#axialBtn"),
  coronalBtn: document.querySelector("#coronalBtn"),
  sagittalBtn: document.querySelector("#sagittalBtn"),
  debugInfo: document.querySelector("#debugInfo"),
  logOutput: document.querySelector("#logOutput"),
  diagnosticsOutput: document.querySelector("#diagnosticsOutput"),
};

createVectorInputs("eye", state.defaults.eye);
createVectorInputs("target", state.defaults.target);
createVectorInputs("up", state.defaults.up);

logger.subscribe(() => {
  els.logOutput.textContent = logger.toText();
  els.logOutput.scrollTop = els.logOutput.scrollHeight;
});

els.fileInput.addEventListener("change", () => runFromFileMenu(() => handleFiles(els.fileInput.files)));
els.folderInput.addEventListener("change", () => runFromFileMenu(() => handleFiles(els.folderInput.files)));
els.mhaInput.addEventListener("change", () => runFromFileMenu(() => handleMhaFile(els.mhaInput.files?.[0])));
els.objInput.addEventListener("change", () => runFromFileMenu(() => handleObjFile(els.objInput.files?.[0])));
els.glbInput.addEventListener("change", () => runFromFileMenu(() => handleGlbFile(els.glbInput.files?.[0])));
els.glbFolderInput.addEventListener("change", () => runFromFileMenu(() => handleGlbFolder(els.glbFolderInput.files)));
els.textureInput.addEventListener("change", () => runFromFileMenu(() => handleTextureFile(els.textureInput.files?.[0])));
els.demoVolumeBtn.addEventListener("click", () => runFromFileMenu(loadDemoVolume));
document.addEventListener("pointerdown", (event) => {
  if (els.fileMenu.open && !els.fileMenu.contains(event.target)) {
    els.fileMenu.open = false;
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !els.fileMenu.open) return;
  els.fileMenu.open = false;
  els.fileMenu.querySelector("summary")?.focus();
  event.preventDefault();
});
els.demoObjBtn.addEventListener("click", () => loadDemoObj());
els.controlsToggleBtn.addEventListener("click", toggleControlsPane);
els.mprCollapseBtn.addEventListener("click", toggleMprPaneCollapsed);
els.mprExpandBtn.addEventListener("click", toggleMprPaneExpanded);
els.renderBtn.addEventListener("click", render);
els.resetBtn.addEventListener("click", resetView);
els.clearObjBtn.addEventListener("click", () => {
  state.threeScene?.clearObjModel();
  writeDiagnostics();
});
els.dynamicToggle.addEventListener("change", () => {
  state.dynamicController?.setEnabled(els.dynamicToggle.checked);
  if (state.volume) requestLinkedMpr("toggle");
});
els.updateInterval.addEventListener("change", () => installScheduler());
for (const input of [els.panSensitivity, els.controlInertia]) {
  input.addEventListener("input", applyInteractionControls);
}
els.objMprToggle.addEventListener("change", () => {
  if (els.objMprToggle.checked) requestObjectLinkedMpr("toggle-on");
  else requestLinkedMpr("model-toggle-off");
});
els.objCenterToggle.addEventListener("change", applyObjControls);
els.objScale.addEventListener("change", applyObjControls);
els.objOpacity.addEventListener("change", applyObjControls);
els.objOpacity.addEventListener("input", applyVisualRangeControls);
els.objMprDistance.addEventListener("change", () => requestObjectLinkedMpr("obj-distance"));
for (const input of [els.objPosX, els.objPosY, els.objPosZ, els.objRotX, els.objRotY, els.objRotZ]) {
  input.addEventListener("input", applyObjControls);
  input.addEventListener("change", applyObjControls);
}
els.flipVolumeXBtn.addEventListener("click", () => toggleVolumeAxisFlip(0));
els.flipVolumeYBtn.addEventListener("click", () => toggleVolumeAxisFlip(1));
els.flipVolumeZBtn.addEventListener("click", () => toggleVolumeAxisFlip(2));
els.resetVolumeFlipBtn.addEventListener("click", resetVolumeAxisFlip);
els.mprFlipUBtn.addEventListener("click", () => toggleMprUvFlip(0));
els.mprFlipVBtn.addEventListener("click", () => toggleMprUvFlip(1));
els.mprFlipResetBtn.addEventListener("click", resetMprUvFlip);
for (const input of [els.mprPlaneToggle, els.mprPlaneOpacity, els.mprPlaneMode, els.mprCameraPlaneDistance, els.mprPlaneBoundaryToggle, els.foregroundFadeToggle, els.foregroundAlpha, els.foregroundDistance, els.foregroundFadeWidth, els.mprThresholdAlphaToggle, els.mprTransparentThreshold]) {
  input.addEventListener("change", applyMprPlaneOverlayControls);
}
for (const input of [els.mprPlaneOpacity, els.foregroundAlpha, els.foregroundDistance, els.foregroundFadeWidth]) {
  input.addEventListener("input", applyVisualRangeControls);
}
for (const input of [els.mprFovWidth, els.mprFovHeight]) {
  input.addEventListener("input", applyMprFovControls);
}
for (const input of [els.mprWidth, els.mprHeight, els.spacingU, els.spacingV]) {
  input.addEventListener("change", updateMprFovControlsFromInputs);
}
els.windowPreset.addEventListener("change", applyWindowPreset);
els.axialBtn.addEventListener("click", () => applyPlanePreset("axial"));
els.coronalBtn.addEventListener("click", () => applyPlanePreset("coronal"));
els.sagittalBtn.addEventListener("click", () => applyPlanePreset("sagittal"));

for (const input of document.querySelectorAll("input, select")) {
  input.addEventListener("change", () => {
    if (!state.volume || input.type === "file") return;
    if (els.dynamicToggle.checked && state.dynamicController) {
      requestLinkedMpr("control");
    } else {
      render();
    }
  });
}

initThree();
updateAxisFlipButtons();
updateMprUvFlipButtons();
updateVisualRangeOutputs();
updateMprFovControlsFromInputs();
writeDiagnostics();

function runFromFileMenu(action) {
  els.fileMenu.open = false;
  return action();
}

function toggleControlsPane() {
  const collapsed = !els.workspace.classList.contains("controls-collapsed");
  els.workspace.classList.toggle("controls-collapsed", collapsed);
  els.controlsToggleBtn.setAttribute("aria-expanded", String(!collapsed));
  els.controlsToggleBtn.textContent = collapsed ? "For Specialist" : "For Everyone";
  if (!collapsed && !els.mprPane.classList.contains("is-collapsed")) {
    els.mprPane.dataset.restoreAfterControls = "true";
    setMprPaneCollapsed(true);
  } else if (collapsed && els.mprPane.dataset.restoreAfterControls === "true") {
    delete els.mprPane.dataset.restoreAfterControls;
    setMprPaneCollapsed(false);
  }
}

function toggleMprPaneCollapsed() {
  const collapsed = !els.mprPane.classList.contains("is-collapsed");
  delete els.mprPane.dataset.restoreAfterControls;
  setMprPaneCollapsed(collapsed);
}

function setMprPaneCollapsed(collapsed) {
  els.mprPane.classList.toggle("is-collapsed", collapsed);
  els.mprPane.classList.remove("is-expanded");
  els.mprCollapseBtn.setAttribute("aria-pressed", String(collapsed));
  els.mprCollapseBtn.textContent = collapsed ? "+" : "−";
  els.mprCollapseBtn.setAttribute("aria-label", collapsed ? "MPRを表示" : "MPRを最小化");
  els.mprExpandBtn.setAttribute("aria-pressed", "false");
  els.mprExpandBtn.textContent = "拡大";
}

function toggleMprPaneExpanded() {
  const expanded = !els.mprPane.classList.contains("is-expanded");
  els.mprPane.classList.remove("is-collapsed");
  els.mprPane.classList.toggle("is-expanded", expanded);
  els.mprCollapseBtn.setAttribute("aria-pressed", "false");
  els.mprCollapseBtn.textContent = "−";
  els.mprCollapseBtn.setAttribute("aria-label", "MPRを最小化");
  els.mprExpandBtn.setAttribute("aria-pressed", String(expanded));
  els.mprExpandBtn.textContent = expanded ? "縮小" : "拡大";
}

async function handleFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  els.statusText.textContent = `${files.length}ファイルを読込中...`;
  try {
    state.volume = null;
    state.result = null;
    const { slices } = await loader.loadFiles(files);
    state.volume = builder.build(slices);
    installVolumeDefaults(state.volume);
    configureThreeForVolume(state.volume);
    updateSeriesInfo();
    if (els.dynamicToggle.checked && state.dynamicController) {
      requestLinkedMpr("initial");
    } else {
      render();
    }
    els.statusText.textContent = "読込完了";
  } catch (error) {
    logger.error("読込又はボリューム構築に失敗しました", { message: error.message });
    els.statusText.textContent = error.message;
  }
}

async function handleMhaFile(file) {
  if (!file) return;
  els.statusText.textContent = `${file.name} を読込中...`;
  try {
    state.volume = null;
    state.result = null;
    // MHAは単一ファイル内にASCIIヘッダとLOCALバイナリが続く形式として扱う。
    // ここでVolumeへ正規化しておくと、以後のMPR、Three.js Volume Box、動的連動はDICOMと同じ経路を通る。
    state.volume = await mhaLoader.loadFile(file);
    installVolumeDefaults(state.volume);
    configureThreeForVolume(state.volume);
    updateSeriesInfo();
    if (els.dynamicToggle.checked && state.dynamicController) {
      requestLinkedMpr("mha-initial");
    } else {
      render();
    }
    els.statusText.textContent = `MHA読込完了: ${file.name}`;
  } catch (error) {
    logger.error("MHA読込に失敗しました", { fileName: file.name, message: error.message });
    els.statusText.textContent = `MHA読込失敗: ${error.message}`;
  } finally {
    els.mhaInput.value = "";
  }
}

async function handleObjFile(file) {
  if (!file) return;
  if (!state.threeReady || !state.threeScene) {
    logger.error("OBJ読込失敗", { message: "Three.jsが初期化されていません" });
    return;
  }
  try {
    els.statusText.textContent = `${file.name} を読込中...`;
    await state.threeScene.loadObjFile(file, getObjOptions());
    clearDemoPhantomForModel();
    els.statusText.textContent = `OBJ読込完了: ${file.name}`;
    requestObjectLinkedMpr("obj-file");
    writeDiagnostics();
  } catch (error) {
    logger.error("OBJ読込失敗", { fileName: file.name, message: error.message });
    els.statusText.textContent = `OBJ読込失敗: ${error.message}`;
  } finally {
    els.objInput.value = "";
  }
}

async function handleGlbFile(file) {
  if (!file) return;
  if (!state.threeReady || !state.threeScene) {
    logger.error("GLB読込失敗", { message: "Three.jsが初期化されていません" });
    return;
  }
  try {
    els.statusText.textContent = `${file.name} を読込中...`;
    // GLBの色・texture・alpha設定を引き継ぎ、外観参照向けのUnlit materialへ変換する。
    await state.threeScene.loadGlbFile(file, getObjOptions());
    clearDemoPhantomForModel();
    els.statusText.textContent = `GLB読込完了: ${file.name}`;
    requestObjectLinkedMpr("glb-file");
    writeDiagnostics();
  } catch (error) {
    logger.error("GLB読込失敗", { fileName: file.name, message: error.message });
    els.statusText.textContent = `GLB読込失敗: ${error.message}`;
  } finally {
    els.glbInput.value = "";
  }
}

async function handleGlbFolder(fileList) {
  const files = [...(fileList ?? [])];
  if (!files.length) return;
  if (!state.threeReady || !state.threeScene) {
    logger.error("GLB＋テクスチャ読込失敗", { message: "Three.jsが初期化されていません" });
    return;
  }
  try {
    els.statusText.textContent = `GLB＋テクスチャを読込中... (${files.length}ファイル)`;
    const summary = await state.threeScene.loadGlbFileSet(files, getObjOptions());
    clearDemoPhantomForModel();
    els.statusText.textContent = `GLB＋テクスチャ読込完了: ${summary.fileName}`;
    requestObjectLinkedMpr("glb-folder");
    writeDiagnostics();
  } catch (error) {
    logger.error("GLB＋テクスチャ読込失敗", { message: error.message });
    els.statusText.textContent = `GLB＋テクスチャ読込失敗: ${error.message}`;
  } finally {
    els.glbFolderInput.value = "";
  }
}

async function handleTextureFile(file) {
  if (!file) return;
  if (!state.threeReady || !state.threeScene) {
    logger.error("テクスチャ適用失敗", { message: "Three.jsが初期化されていません" });
    return;
  }
  try {
    els.statusText.textContent = `${file.name} をテクスチャとして適用中...`;
    const summary = await state.threeScene.applyTextureFile(file);
    els.statusText.textContent = `テクスチャ適用完了: ${file.name}`;
    logger.info("モデルへテクスチャを適用しました", summary);
    writeDiagnostics();
  } catch (error) {
    logger.error("テクスチャ適用失敗", { fileName: file.name, message: error.message });
    els.statusText.textContent = `テクスチャ適用失敗: ${error.message}`;
  } finally {
    els.textureInput.value = "";
  }
}

function applyObjControls() {
  state.threeScene?.applyObjTransform(getObjOptions());
  requestObjectLinkedMpr("obj-transform");
  writeDiagnostics();
}

function loadDemoObj() {
  if (!state.threeReady || !state.threeScene) {
    logger.error("OBJデモ生成失敗", { message: "Three.jsが初期化されていません" });
    return;
  }
  const obj = [
    "o demo_obj_pyramid",
    "v -25 -25 0",
    "v 25 -25 0",
    "v 25 25 0",
    "v -25 25 0",
    "v 0 0 55",
    "f 1 2 3",
    "f 1 3 4",
    "f 1 2 5",
    "f 2 3 5",
    "f 3 4 5",
    "f 4 1 5",
  ].join("\n");
  state.threeScene.loadObjText(obj, "demo_obj_pyramid.obj", getObjOptions());
  clearDemoPhantomForModel();
  els.statusText.textContent = "モデルデモ読込完了";
  requestObjectLinkedMpr("obj-demo");
  writeDiagnostics();
}

function getObjOptions() {
  const controlPosition = [readFloat(els.objPosX, 0), readFloat(els.objPosY, 0), readFloat(els.objPosZ, 0)];
  const controlRotation = [readFloat(els.objRotX, 0), readFloat(els.objRotY, 0), readFloat(els.objRotZ, 0)];
  const position = applyAxisFlipToVector(controlPosition);
  const rotation = applyAxisFlipToVector(controlRotation);
  return {
    centerToTarget: els.objCenterToggle.checked,
    scale: Math.max(0.001, readFloat(els.objScale, 1)),
    opacity: Math.max(0.05, Math.min(1, readFloat(els.objOpacity, 1))),
    position,
    rotation,
    controlPosition,
    controlRotation,
    controlAxisFlip: [...state.volumeAxisFlip],
  };
}

function applyAxisFlipToVector(vector) {
  return vector.map((value, index) => value * state.volumeAxisFlip[index]);
}

function toggleVolumeAxisFlip(axis) {
  state.volumeAxisFlip[axis] *= -1;
  applyVolumeAxisFlip("axis-flip");
}

function resetVolumeAxisFlip() {
  state.volumeAxisFlip = [1, 1, 1];
  applyVolumeAxisFlip("axis-flip-reset");
}

function applyVolumeAxisFlip(reason) {
  updateAxisFlipButtons();
  if (!state.volume) {
    writeDiagnostics();
    return;
  }
  // CT/MHA側のworld座標をVolume中心まわりに反転する。
  // モデル操作入力にも同じ符号を掛け直し、UI上の移動/回転方向をCT座標の反転状態と揃える。
  configureThreeForVolume(state.volume, { preserveCamera: true });
  state.threeScene?.applyObjTransform(getObjOptions());
  if (els.dynamicToggle.checked && state.dynamicController) {
    if (shouldUseObjectMpr()) requestObjectLinkedMpr(reason);
    else requestLinkedMpr(reason);
  } else {
    render();
  }
  logger.info("CT座標軸反転を更新しました", { flip: formatAxisFlip() });
  writeDiagnostics();
}

function toggleMprUvFlip(axis) {
  state.mprUvFlip[axis] *= -1;
  applyMprUvFlip();
}

function resetMprUvFlip() {
  state.mprUvFlip = [...DEFAULT_MPR_UV_FLIP];
  applyMprUvFlip();
}

function applyMprUvFlip() {
  updateMprUvFlipButtons();
  if (state.result) updateMprPlaneOverlay(state.result, state.dynamicInfo?.request ?? getRequest());
  writeDiagnostics();
}

function updateMprUvFlipButtons() {
  const [u, v] = state.mprUvFlip;
  els.mprFlipUBtn.setAttribute("aria-pressed", String(u < 0));
  els.mprFlipVBtn.setAttribute("aria-pressed", String(v < 0));
  els.mprFlipUBtn.textContent = `重畳 U反転: ${u < 0 ? "ON" : "OFF"}`;
  els.mprFlipVBtn.textContent = `重畳 V反転: ${v < 0 ? "ON" : "OFF"}`;
}

function loadDemoVolume() {
  const start = performance.now();
  const width = 96;
  const height = 96;
  const depth = 96;
  const spacing = [1.5, 1.5, 1.5];
  const origin = [-72, -72, -72];
  const data = new Int16Array(width * height * depth);
  let minimumHU = Infinity;
  let maximumHU = -Infinity;
  const center = [(width - 1) / 2, (height - 1) / 2, (depth - 1) / 2];
  for (let z = 0; z < depth; z += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const dx = (x - center[0]) * spacing[0];
        const dy = (y - center[1]) * spacing[1];
        const dz = (z - center[2]) * spacing[2];
        const r = Math.hypot(dx, dy, dz);
        const shell = Math.abs(r - 45) < 2.5 ? 900 : -850;
        const core = r < 22 ? 80 : shell;
        const stripe = Math.abs(dx) < 4 || Math.abs(dy) < 4 ? 220 : core;
        const value = Math.round(stripe + 120 * Math.sin(z / 7));
        const index = z * width * height + y * width + x;
        data[index] = value;
        minimumHU = Math.min(minimumHU, value);
        maximumHU = Math.max(maximumHU, value);
      }
    }
  }
  state.volume = {
    data,
    metadata: {
      dimensions: [width, height, depth],
      origin,
      spacing,
      direction: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      inverseDirection: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      minimumHU,
      maximumHU,
      minimumValue: minimumHU,
      maximumValue: maximumHU,
      modality: "DEMO",
      scalarType: "MET_SHORT",
      sourceFormat: "DEMO",
      seriesInstanceUid: "demo.synthetic.phantom",
      studyInstanceUid: "demo",
      windowCenter: 40,
      windowWidth: 900,
      photometricInterpretation: "MONOCHROME2",
    },
  };
  installVolumeDefaults(state.volume);
  configureThreeForVolume(state.volume);
  updateSeriesInfo();
  els.windowLevel.value = 40;
  els.windowWidth.value = 900;
  if (els.dynamicToggle.checked && state.dynamicController) {
    requestLinkedMpr("demo");
  } else {
    render();
  }
  els.statusText.textContent = "デモファントム読込完了";
  logger.info("デモファントムを生成しました", { dimensions: state.volume.metadata.dimensions, ms: Math.round(performance.now() - start) });
}

function clearDemoPhantomForModel() {
  if (state.volume?.metadata?.sourceFormat !== "DEMO") return false;
  state.volume = null;
  state.result = null;
  state.dynamicInfo = null;
  state.dynamicController = null;
  state.threeScene?.clearVolume();
  drawPlaceholder("DICOM CTシリーズ又はMHAファイルを読み込んでください");
  els.seriesUid.textContent = "-";
  els.volumeDimensions.textContent = "-";
  els.volumeSpacing.textContent = "-";
  els.huRange.textContent = "-";
  logger.info("モデル読込に伴いデモファントムを消去しました");
  return true;
}

function render() {
  if (!state.volume) {
    drawPlaceholder("DICOM CTシリーズ又はMHAファイルを読み込んでください");
    return;
  }
  try {
    const request = getRequest();
    state.result = renderer.render(state.volume, request);
    renderer.draw(els.canvas, state.result);
    updateMprPlaneOverlay(state.result, request);
    updateDebugInfo();
  } catch (error) {
    logger.error("MPR生成失敗", { message: error.message, stack: error.stack });
  }
}

function getRequest() {
  return {
    width: readInt(els.mprWidth, 512),
    height: readInt(els.mprHeight, 512),
    spacingU: readFloat(els.spacingU, 1),
    spacingV: readFloat(els.spacingV, 1),
    eye: readVector("eye"),
    target: readVector("target"),
    up: readVector("up"),
    normalOffset: readFloat(els.normalOffset, 0),
    interpolation: els.interpolation.value,
    outsideValue: -1024,
    windowLevel: readFloat(els.windowLevel, 40),
    windowWidth: Math.max(1, readFloat(els.windowWidth, 400)),
  };
}

function getBaseMprRequest() {
  return {
    width: readInt(els.mprWidth, 512),
    height: readInt(els.mprHeight, 512),
    spacingU: readFloat(els.spacingU, 1),
    spacingV: readFloat(els.spacingV, 1),
    normalOffset: readFloat(els.normalOffset, 0),
    interpolation: els.interpolation.value,
    outsideValue: -1024,
    windowLevel: readFloat(els.windowLevel, 40),
    windowWidth: Math.max(1, readFloat(els.windowWidth, 400)),
  };
}

function installVolumeDefaults(volume) {
  const preset = planeCalculator.axial(volume.metadata);
  state.defaults.eye = preset.eye;
  state.defaults.target = preset.target;
  state.defaults.up = preset.up;
  state.defaults.wl = volume.metadata.windowCenter ?? 40;
  state.defaults.ww = volume.metadata.windowWidth ?? 400;
  setVector("eye", preset.eye);
  setVector("target", preset.target);
  setVector("up", preset.up);
  els.windowLevel.value = state.defaults.wl;
  els.windowWidth.value = state.defaults.ww;
  els.normalOffset.value = 0;
}

async function initThree() {
  try {
    state.threeScene = new ThreeSceneManager(els.threeViewport, viewerConfig.three, logger);
    await state.threeScene.init();
    state.threeReady = true;
    state.cameraController = new ThreeCameraController(
      state.threeScene.THREE,
      state.threeScene.camera,
      state.threeScene.controls,
      () => readFloat(els.objMprDistance, DEFAULT_MPR_DISTANCE)
    );
    installScheduler();
    state.threeScene.onControlsChange(() => requestLinkedMpr("camera"));
    if (state.volume) configureThreeForVolume(state.volume);
    writeDiagnostics();
  } catch (error) {
    state.threeReady = false;
    els.threeViewport.innerHTML = `<div class="three-placeholder">Three.jsを読み込めませんでした。静的MPRは利用できます。<br>${escapeHtml(error.message)}</div>`;
    logger.error("Three.js初期化失敗", { message: error.message });
    writeDiagnostics();
  }
}

function installScheduler() {
  state.scheduler = new MprUpdateScheduler((reason) => {
    if (!state.volume || !state.dynamicController) return;
    if (shouldUseObjectMpr()) {
      const objInput = state.threeScene.getObjCameraInput(readFloat(els.objMprDistance, DEFAULT_MPR_DISTANCE));
      state.dynamicController.updateFromThreeInput(state.volume, objInput, reason);
    } else {
      state.dynamicController.update(state.volume, reason);
    }
  }, { minUpdateIntervalMs: readFloat(els.updateInterval, viewerConfig.dynamicMpr.minUpdateIntervalMs) });
}

function configureThreeForVolume(volume, { preserveCamera = false } = {}) {
  if (!state.threeReady || !state.threeScene) return;
  updateMapperForVolume(volume);
  state.threeScene.configureVolume(volume.metadata, state.mapper, { preserveCamera });
  state.dynamicController = new DynamicMprController({
    renderer,
    mprCanvas: els.canvas,
    cameraController: state.cameraController,
    mapper: state.mapper,
    getBaseRequest: getBaseMprRequest,
    logger,
    onResult: (info) => {
      state.dynamicInfo = info;
      state.result = info.result;
      state.threeScene?.setDynamicFocusPoint(info.threeInput.target);
      setVector("eye", info.dicomInput.eye);
      setVector("target", info.dicomInput.target);
      setVector("up", info.dicomInput.up);
      updateMprPlaneOverlay(info.result, info.request);
      updateDynamicDebugInfo(info);
      if (info.timings.totalMs > viewerConfig.dynamicMpr.performanceWarnMs) {
        logger.warn("動的MPR更新が性能目標を超えました", { totalMs: Math.round(info.timings.totalMs) });
      }
    },
  });
  state.dynamicController.setEnabled(els.dynamicToggle.checked);
  updateAxisFlipButtons();
}

function updateMapperForVolume(volume) {
  const center = volumeCenter(volume.metadata);
  const flipMatrix = buildCenteredAxisFlipMatrix(center, state.volumeAxisFlip);
  state.mapper.setThreeToDicomMatrix(multiplyMat4(flipMatrix, viewerConfig.mapper.threeToDicomMatrix));
}

function requestDynamicMpr(reason) {
  if (!state.volume || !state.dynamicController) return;
  state.scheduler?.request(reason);
}

function requestLinkedMpr(reason) {
  requestDynamicMpr(reason);
}

function requestObjectLinkedMpr(reason) {
  if (shouldUseObjectMpr()) requestDynamicMpr(`model-${reason}`);
}

function shouldUseObjectMpr() {
  return Boolean(els.objMprToggle.checked && state.threeScene?.getObjSummary?.());
}

function applyPlanePreset(name) {
  if (!state.volume) return;
  const preset = planeCalculator[name](state.volume.metadata);
  setVector("eye", preset.eye);
  setVector("target", preset.target);
  setVector("up", preset.up);
  els.normalOffset.value = 0;
  render();
}

function applyMprPlaneOverlayControls() {
  state.threeScene?.updateOverlayOptions(getMprPlaneOverlayOptions());
  if (state.result) updateMprPlaneOverlay(state.result, state.dynamicInfo?.request ?? getRequest());
  writeDiagnostics();
}

function applyInteractionControls() {
  const panSpeed = readFloat(els.panSensitivity, viewerConfig.three.panSpeed);
  const inertia = readFloat(els.controlInertia, viewerConfig.three.inertia);
  els.panSensitivityValue.value = panSpeed.toFixed(2);
  els.controlInertiaValue.value = inertia.toFixed(2);
  state.threeScene?.setInteractionOptions({ panSpeed, inertia });
  writeDiagnostics();
}

function applyVisualRangeControls(event) {
  updateVisualRangeOutputs();
  if (event?.currentTarget === els.objOpacity) applyObjControls();
  else applyMprPlaneOverlayControls();
}

function updateVisualRangeOutputs() {
  els.objOpacityValue.value = readFloat(els.objOpacity, 1).toFixed(2);
  els.mprPlaneOpacityValue.value = readFloat(els.mprPlaneOpacity, 1).toFixed(2);
  els.foregroundAlphaValue.value = readFloat(els.foregroundAlpha, 0).toFixed(2);
  els.foregroundDistanceValue.value = `${Math.round(readFloat(els.foregroundDistance, 0))} mm`;
  els.foregroundFadeWidthValue.value = `${Math.round(readFloat(els.foregroundFadeWidth, 10))} mm`;
}

function applyMprFovControls() {
  const widthPx = Math.max(1, readInt(els.mprWidth, 512));
  const heightPx = Math.max(1, readInt(els.mprHeight, 512));
  const widthMm = readFloat(els.mprFovWidth, 512);
  const heightMm = readFloat(els.mprFovHeight, 512);
  els.spacingU.value = (widthMm / widthPx).toFixed(4);
  els.spacingV.value = (heightMm / heightPx).toFixed(4);
  updateMprFovOutputs(widthMm, heightMm);
  if (state.volume) requestLinkedMpr("mpr-fov");
  writeDiagnostics();
}

function updateMprFovControlsFromInputs() {
  const widthMm = Math.max(50, Math.min(1200, readInt(els.mprWidth, 512) * readFloat(els.spacingU, 1)));
  const heightMm = Math.max(50, Math.min(1200, readInt(els.mprHeight, 512) * readFloat(els.spacingV, 1)));
  els.mprFovWidth.value = String(Math.round(widthMm));
  els.mprFovHeight.value = String(Math.round(heightMm));
  updateMprFovOutputs(widthMm, heightMm);
}

function updateMprFovOutputs(widthMm, heightMm) {
  els.mprFovWidthValue.value = `${Math.round(widthMm)} mm`;
  els.mprFovHeightValue.value = `${Math.round(heightMm)} mm`;
}

function updateMprPlaneOverlay(result, request) {
  if (!state.threeReady || !state.threeScene || !result || !request) return;
  state.threeScene.updateMprPlaneOverlay(els.canvas, result, request, getMprPlaneOverlayOptions());
}

function getMprPlaneOverlayOptions() {
  return {
    visible: els.mprPlaneToggle.checked,
    opacity: Math.max(0, Math.min(1, readFloat(els.mprPlaneOpacity, 1))),
    displayMode: els.mprPlaneMode.value,
    cameraPlaneDistance: Math.max(0, readFloat(els.mprCameraPlaneDistance, 0)),
    flipU: state.mprUvFlip[0] < 0,
    flipV: state.mprUvFlip[1] < 0,
    showBoundary: els.mprPlaneBoundaryToggle.checked,
    foregroundEnabled: els.foregroundFadeToggle.checked,
    frontAlpha: Math.max(0, Math.min(1, readFloat(els.foregroundAlpha, 0))),
    foregroundDistance: Math.max(0, readFloat(els.foregroundDistance, 0)),
    fadeWidth: Math.max(0, readFloat(els.foregroundFadeWidth, 10)),
    thresholdAlphaEnabled: els.mprThresholdAlphaToggle.checked,
    transparentThreshold: readFloat(els.mprTransparentThreshold, -500),
  };
}

function applyWindowPreset() {
  const presets = {
    soft: [40, 400],
    lung: [-600, 1500],
    bone: [300, 1800],
    dicom: [state.volume?.metadata.windowCenter ?? state.defaults.wl, state.volume?.metadata.windowWidth ?? state.defaults.ww],
    minmax: [
      ((state.volume?.metadata.minimumValue ?? state.volume?.metadata.minimumHU ?? -200) + (state.volume?.metadata.maximumValue ?? state.volume?.metadata.maximumHU ?? 200)) / 2,
      Math.max(1, (state.volume?.metadata.maximumValue ?? state.volume?.metadata.maximumHU ?? 200) - (state.volume?.metadata.minimumValue ?? state.volume?.metadata.minimumHU ?? -200)),
    ],
  };
  const preset = presets[els.windowPreset.value];
  if (!preset) return;
  els.windowLevel.value = preset[0];
  els.windowWidth.value = preset[1];
  render();
}

function resetView() {
  setVector("eye", state.defaults.eye);
  setVector("target", state.defaults.target);
  setVector("up", state.defaults.up);
  els.windowLevel.value = state.defaults.wl;
  els.windowWidth.value = state.defaults.ww;
  els.mprWidth.value = 512;
  els.mprHeight.value = 512;
  els.spacingU.value = 1;
  els.spacingV.value = 1;
  els.normalOffset.value = 0;
  els.interpolation.value = "linear";
  render();
}

function updateSeriesInfo() {
  const m = state.volume.metadata;
  els.seriesUid.textContent = compactUid(m.seriesInstanceUid || m.sourceFileName || m.sourceFormat);
  els.volumeDimensions.textContent = m.dimensions.join(" x ");
  els.volumeSpacing.textContent = m.spacing.map((x) => x.toFixed(3)).join(", ");
  els.huRange.textContent = `${formatNumber(m.minimumValue ?? m.minimumHU)} to ${formatNumber(m.maximumValue ?? m.maximumHU)}`;
}

function updateDebugInfo() {
  if (!state.volume || !state.result) return;
  const request = getRequest();
  const { plane } = state.result;
  const memory = performance.memory ? `${Math.round(performance.memory.usedJSHeapSize / 1024 / 1024)} MB` : "n/a";
  els.debugInfo.textContent = [
    `eye: ${formatVector(request.eye)}`,
    `target: ${formatVector(request.target)}`,
    `up: ${formatVector(request.up)}`,
    `n: ${formatVector(plane.n)}`,
    `u: ${formatVector(plane.u)}`,
    `v: ${formatVector(plane.v)}`,
    `center: ${formatVector(plane.center)}`,
    `MPR Pixel Spacing: ${request.spacingU} x ${request.spacingV} mm`,
    `MPR Size: ${request.width} x ${request.height}`,
    `Physical Size: ${(request.width * request.spacingU).toFixed(1)} x ${(request.height * request.spacingV).toFixed(1)} mm`,
    `CT Axis Flip: X=${state.volumeAxisFlip[0]}, Y=${state.volumeAxisFlip[1]}, Z=${state.volumeAxisFlip[2]}`,
    `Render Time: ${state.result.elapsedMs.toFixed(1)} ms`,
    `Outside / Invalid: ${state.result.outsideCount} / ${state.result.invalidCount}`,
    `Volume Dimensions: ${state.volume.metadata.dimensions.join(" x ")}`,
    `Memory: ${memory}`,
  ].join("\n");
}

function updateDynamicDebugInfo(info) {
  const memory = performance.memory ? `${Math.round(performance.memory.usedJSHeapSize / 1024 / 1024)} MB` : "n/a";
  els.debugInfo.textContent = [
    info.text,
    "",
    `MPR Pixel Spacing: ${readFloat(els.spacingU, 1)} x ${readFloat(els.spacingV, 1)} mm`,
    `MPR Size: ${readInt(els.mprWidth, 512)} x ${readInt(els.mprHeight, 512)}`,
    `CT Axis Flip: X=${state.volumeAxisFlip[0]}, Y=${state.volumeAxisFlip[1]}, Z=${state.volumeAxisFlip[2]}`,
    `Volume Dimensions: ${state.volume.metadata.dimensions.join(" x ")}`,
    `Memory: ${memory}`,
  ].join("\n");
  writeDiagnostics();
}

function getMprPixelStats() {
  if (!state.result) return null;
  let nonOutside = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const value of state.result.pixels) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    if (value !== -1024) nonOutside += 1;
  }
  return {
    width: state.result.width,
    height: state.result.height,
    nonOutside,
    min,
    max,
  };
}

function writeDiagnostics() {
  if (!els.diagnosticsOutput) return;
  els.diagnosticsOutput.textContent = JSON.stringify({
    threeReady: state.threeReady,
    hasVolume: Boolean(state.volume),
    volumeAxisFlip: [...state.volumeAxisFlip],
    mprUvFlip: [...state.mprUvFlip],
    threeToVolumeMatrix: state.mapper.threeToDicomMatrix,
    volumeDimensions: state.volume?.metadata.dimensions ?? null,
    volumeSourceFormat: state.volume?.metadata.sourceFormat ?? null,
    volumeScalarType: state.volume?.metadata.scalarType ?? null,
    seriesInstanceUid: state.volume?.metadata.seriesInstanceUid ?? null,
    dynamicEnabled: els.dynamicToggle.checked,
    modelLinkedMpr: shouldUseObjectMpr(),
    threePixelStats: state.threeScene?.getPixelStats?.() ?? null,
    interaction: state.threeScene?.getInteractionSummary?.() ?? null,
    modelSummary: state.threeScene?.getModelSummary?.() ?? state.threeScene?.getObjSummary?.() ?? null,
    mprPlaneOverlay: state.threeScene?.getMprPlaneOverlaySummary?.() ?? null,
    foregroundTransparency: state.threeScene?.getForegroundTransparencySummary?.() ?? null,
    mprPixelStats: getMprPixelStats(),
    lastDynamicTimings: state.dynamicInfo?.timings ?? null,
  });
}

function createVectorInputs(name, initial) {
  const container = document.querySelector(`[data-vector="${name}"]`);
  ["x", "y", "z"].forEach((axis, index) => {
    const label = document.createElement("label");
    label.textContent = axis;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.1";
    input.value = initial[index];
    input.dataset.vectorInput = name;
    input.dataset.axis = String(index);
    label.append(input);
    container.append(label);
  });
}

function readVector(name) {
  return [...document.querySelectorAll(`[data-vector-input="${name}"]`)].map((input) => readFloat(input, 0));
}

function setVector(name, vector) {
  for (const input of document.querySelectorAll(`[data-vector-input="${name}"]`)) {
    input.value = Number(vector[Number(input.dataset.axis)]).toFixed(3);
  }
}

function readFloat(input, fallback) {
  const value = Number.parseFloat(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function readInt(input, fallback) {
  const value = Number.parseInt(input.value, 10);
  return Number.isFinite(value) ? value : fallback;
}

function compactUid(uid) {
  if (!uid) return "-";
  return uid.length > 28 ? `${uid.slice(0, 12)}...${uid.slice(-10)}` : uid;
}

function formatNumber(value) {
  return Number.isFinite(value) ? Number(value.toFixed(3)).toString() : "-";
}

function updateAxisFlipButtons() {
  const entries = [
    [els.flipVolumeXBtn, state.volumeAxisFlip[0]],
    [els.flipVolumeYBtn, state.volumeAxisFlip[1]],
    [els.flipVolumeZBtn, state.volumeAxisFlip[2]],
  ];
  for (const [button, sign] of entries) {
    if (!button) continue;
    const flipped = sign < 0;
    button.setAttribute("aria-pressed", String(flipped));
    button.classList.toggle("is-active", flipped);
  }
}

function formatAxisFlip() {
  return {
    x: state.volumeAxisFlip[0] < 0 ? "反転" : "通常",
    y: state.volumeAxisFlip[1] < 0 ? "反転" : "通常",
    z: state.volumeAxisFlip[2] < 0 ? "反転" : "通常",
  };
}

function buildCenteredAxisFlipMatrix(center, signs) {
  const [sx, sy, sz] = signs;
  const [cx, cy, cz] = center;
  return [
    sx, 0, 0, (1 - sx) * cx,
    0, sy, 0, (1 - sy) * cy,
    0, 0, sz, (1 - sz) * cz,
    0, 0, 0, 1,
  ];
}

function multiplyMat4(a, b) {
  const out = new Array(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      for (let k = 0; k < 4; k += 1) {
        out[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
      }
    }
  }
  return out;
}

function drawPlaceholder(message) {
  const ctx = els.canvas.getContext("2d");
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.fillStyle = "#d1d5db";
  ctx.font = "16px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(message, els.canvas.width / 2, els.canvas.height / 2);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

drawPlaceholder("DICOM CTシリーズ又はMHAファイルを読み込んでください");
