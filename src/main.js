import { DicomLoader } from "./DicomLoader.js";
import { MhaLoader } from "./MhaLoader.js";
import { VolumeBuilder } from "./VolumeBuilder.js";
import { MprPlaneCalculator } from "./MprPlaneCalculator.js";
import { MprRenderer } from "./MprRenderer.js?v=performance-pass-1";
import { ValidationLogger } from "./ValidationLogger.js";
import { formatVector } from "./math3d.js";
import { volumeCenter } from "./VolumeGeometry.js";
import { viewerConfig } from "./config/viewerConfig.js?v=interaction-defaults-1";
import { ThreeSceneManager } from "./three/ThreeSceneManager.js?v=validation-transparent-target-1";
import { ThreeCameraController } from "./three/ThreeCameraController.js";
import { ThreeDicomCoordinateMapper } from "./three/ThreeDicomCoordinateMapper.js";
import { DynamicMprController, constrainDicomInputToPlane } from "./three/DynamicMprController.js?v=fixed-mpr-plane-2";
import { MprUpdateScheduler } from "./three/MprUpdateScheduler.js";
import { ValidationController } from "./validation/ValidationController.js?v=mpr-zoom-1";

const logger = new ValidationLogger();
const loader = new DicomLoader(logger);
const mhaLoader = new MhaLoader(logger);
const builder = new VolumeBuilder(logger);
const renderer = new MprRenderer(logger);
const planeCalculator = new MprPlaneCalculator();
const DEFAULT_MPR_DISTANCE = 600;
const MPR_DISTANCE_MIN = 50;
const MPR_DISTANCE_MAX = 1200;
const MPR_THRESHOLD_MIN = -1200;
const MPR_THRESHOLD_MAX = 500;
const WINDOW_LEVEL_MIN = -2000;
const WINDOW_LEVEL_MAX = 3000;
const WINDOW_WIDTH_MAX = 5000;
const DEMO_DATASETS = Object.freeze({
  mikan: {
    id: "mikan",
    name: "みかん",
    mprDistance: 300,
    modelFileName: "mikan.glb",
    volumeFileName: "mikan-volume.mha",
    glbUrl: "./data/mikan/mikan.glb?v=1",
    mhaUrl: "./data/mikan/volume_preprocessed.mha.gz?v=model-bounds-2",
    mhaFallbackUrl: "./data/mikan/volume_preprocessed.model-bounds.mha?v=model-bounds-1",
  },
  banana: {
    id: "banana",
    name: "バナナ",
    mprDistance: 300,
    modelFileName: "banana.glb",
    volumeFileName: "banana-volume.mha",
    glbUrl: "./data/banana/banana.glb?v=1",
    mhaUrl: "./data/banana/volume_preprocessed.mha.gz?v=model-bounds-1",
    mhaFallbackUrl: "./data/banana/volume_preprocessed.model-bounds.mha?v=model-bounds-1",
  },
});
const STARTUP_DEMO = DEMO_DATASETS.mikan;
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
  observationLocked: false,
  startupDemoToken: 0,
  startupDemoLoading: false,
  activeDemoId: null,
  defaults: {
    eye: [0, 0, -500],
    target: [0, 0, 0],
    up: [0, 1, 0],
    wl: 40,
    ww: 400,
  },
};

const els = {
  startupLoading: document.querySelector("#startupLoading"),
  startupLoadingTitle: document.querySelector("#startupLoadingTitle"),
  startupLoadingDetail: document.querySelector("#startupLoadingDetail"),
  fileInput: document.querySelector("#fileInput"),
  folderInput: document.querySelector("#folderInput"),
  mhaInput: document.querySelector("#mhaInput"),
  objInput: document.querySelector("#objInput"),
  glbInput: document.querySelector("#glbInput"),
  glbFolderInput: document.querySelector("#glbFolderInput"),
  textureInput: document.querySelector("#textureInput"),
  mikanDemoBtn: document.querySelector("#mikanDemoBtn"),
  bananaDemoBtn: document.querySelector("#bananaDemoBtn"),
  demoVolumeBtn: document.querySelector("#demoVolumeBtn"),
  statusText: document.querySelector("#statusText"),
  fileMenu: document.querySelector(".file-menu"),
  seriesUid: document.querySelector("#seriesUid"),
  volumeDimensions: document.querySelector("#volumeDimensions"),
  volumeSpacing: document.querySelector("#volumeSpacing"),
  huRange: document.querySelector("#huRange"),
  threeViewport: document.querySelector("#threeViewport"),
  workspace: document.querySelector(".workspace"),
  controlsPane: document.querySelector(".controls-pane"),
  controlsToggleBtn: document.querySelector("#controlsToggleBtn"),
  observationLockBtn: document.querySelector("#observationLockBtn"),
  canvas: document.querySelector("#mprCanvas"),
  mprPane: document.querySelector(".mpr-pane"),
  mprPaneTitle: document.querySelector("#mprPaneTitle"),
  mprCollapseBtn: document.querySelector("#mprCollapseBtn"),
  mprExpandBtn: document.querySelector("#mprExpandBtn"),
  panSensitivity: document.querySelector("#panSensitivity"),
  panSensitivityValue: document.querySelector("#panSensitivityValue"),
  controlInertia: document.querySelector("#controlInertia"),
  controlInertiaValue: document.querySelector("#controlInertiaValue"),
  objScale: document.querySelector("#objScale"),
  objOpacity: document.querySelector("#objOpacity"),
  objOpacityValue: document.querySelector("#objOpacityValue"),
  objMprDistance: document.querySelector("#objMprDistance"),
  objMprDistanceValue: document.querySelector("#objMprDistanceValue"),
  objPosX: document.querySelector("#objPosX"),
  objPosY: document.querySelector("#objPosY"),
  objPosZ: document.querySelector("#objPosZ"),
  objRotX: document.querySelector("#objRotX"),
  objRotY: document.querySelector("#objRotY"),
  objRotZ: document.querySelector("#objRotZ"),
  flipVolumeXBtn: document.querySelector("#flipVolumeXBtn"),
  flipVolumeYBtn: document.querySelector("#flipVolumeYBtn"),
  flipVolumeZBtn: document.querySelector("#flipVolumeZBtn"),
  resetVolumeFlipBtn: document.querySelector("#resetVolumeFlipBtn"),
  windowLevel: document.querySelector("#windowLevel"),
  windowLevelValue: document.querySelector("#windowLevelValue"),
  windowWidth: document.querySelector("#windowWidth"),
  windowWidthValue: document.querySelector("#windowWidthValue"),
  windowPreset: document.querySelector("#windowPreset"),
  mprWidth: document.querySelector("#mprWidth"),
  mprHeight: document.querySelector("#mprHeight"),
  spacingU: document.querySelector("#spacingU"),
  spacingV: document.querySelector("#spacingV"),
  normalOffset: document.querySelector("#normalOffset"),
  interpolation: document.querySelector("#interpolation"),
  mprPlaneToggle: document.querySelector("#mprPlaneToggle"),
  mprOrientationMode: document.querySelector("#mprOrientationMode"),
  mprOrientationHint: document.querySelector("#mprOrientationHint"),
  mprFovWidth: document.querySelector("#mprFovWidth"),
  mprFovWidthValue: document.querySelector("#mprFovWidthValue"),
  mprFovHeight: document.querySelector("#mprFovHeight"),
  mprFovHeightValue: document.querySelector("#mprFovHeightValue"),
  mprPlaneOpacity: document.querySelector("#mprPlaneOpacity"),
  mprPlaneOpacityValue: document.querySelector("#mprPlaneOpacityValue"),
  mprFlipUBtn: document.querySelector("#mprFlipUBtn"),
  mprFlipVBtn: document.querySelector("#mprFlipVBtn"),
  mprFlipResetBtn: document.querySelector("#mprFlipResetBtn"),
  mprPlaneBoundaryToggle: document.querySelector("#mprPlaneBoundaryToggle"),
  foregroundAlpha: document.querySelector("#foregroundAlpha"),
  foregroundAlphaValue: document.querySelector("#foregroundAlphaValue"),
  foregroundDistance: document.querySelector("#foregroundDistance"),
  foregroundDistanceValue: document.querySelector("#foregroundDistanceValue"),
  foregroundFadeWidth: document.querySelector("#foregroundFadeWidth"),
  foregroundFadeWidthValue: document.querySelector("#foregroundFadeWidthValue"),
  mprThresholdAlphaToggle: document.querySelector("#mprThresholdAlphaToggle"),
  mprTransparentThreshold: document.querySelector("#mprTransparentThreshold"),
  mprTransparentThresholdValue: document.querySelector("#mprTransparentThresholdValue"),
  mprTransparentThresholdControl: document.querySelector("#mprTransparentThresholdControl"),
  debugInfo: document.querySelector("#debugInfo"),
  logOutput: document.querySelector("#logOutput"),
  diagnosticsOutput: document.querySelector("#diagnosticsOutput"),
  validationModeBtn: document.querySelector("#validationModeBtn"),
};

createVectorInputs("eye", state.defaults.eye);
createVectorInputs("target", state.defaults.target);
createVectorInputs("up", state.defaults.up);

logger.subscribe(() => {
  els.logOutput.textContent = logger.toText();
  els.logOutput.scrollTop = els.logOutput.scrollHeight;
});

els.fileInput.addEventListener("change", () => runManualFromFileMenu(() => handleFiles(els.fileInput.files)));
els.folderInput.addEventListener("change", () => runManualFromFileMenu(() => handleFiles(els.folderInput.files)));
els.mhaInput.addEventListener("change", () => runManualFromFileMenu(() => handleMhaFile(els.mhaInput.files?.[0])));
els.objInput.addEventListener("change", () => runManualFromFileMenu(() => handleObjFile(els.objInput.files?.[0])));
els.glbInput.addEventListener("change", () => runManualFromFileMenu(() => handleGlbFile(els.glbInput.files?.[0])));
els.glbFolderInput.addEventListener("change", () => runManualFromFileMenu(() => handleGlbFolder(els.glbFolderInput.files)));
els.textureInput.addEventListener("change", () => runManualFromFileMenu(() => handleTextureFile(els.textureInput.files?.[0])));
els.mikanDemoBtn.addEventListener("click", () => loadDemoFromToolbar(DEMO_DATASETS.mikan));
els.bananaDemoBtn.addEventListener("click", () => loadDemoFromToolbar(DEMO_DATASETS.banana));
els.demoVolumeBtn.addEventListener("click", () => runManualFromFileMenu(loadDemoVolume));
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
els.controlsToggleBtn.addEventListener("click", toggleControlsPane);
els.observationLockBtn.addEventListener("click", toggleObservationLock);
els.mprCollapseBtn.addEventListener("click", toggleMprPaneCollapsed);
els.mprExpandBtn.addEventListener("click", toggleMprPaneExpanded);
els.mprOrientationMode.addEventListener("change", updateMprOrientationUi);
els.controlsPane.addEventListener("wheel", (event) => {
  const input = event.target instanceof HTMLInputElement ? event.target : null;
  if (input?.type !== "number") return;
  event.preventDefault();
  els.controlsPane.scrollTop += event.deltaY;
}, { passive: false });
for (const input of [els.panSensitivity, els.controlInertia]) {
  input.addEventListener("input", applyInteractionControls);
}
els.objScale.addEventListener("change", applyObjControls);
els.objOpacity.addEventListener("change", applyObjControls);
els.objOpacity.addEventListener("input", applyVisualRangeControls);
els.objMprDistance.addEventListener("input", () => applyMprDistanceControl("range"));
els.objMprDistanceValue.addEventListener("change", () => applyMprDistanceControl("number"));
els.mprTransparentThreshold.addEventListener("input", () => applyMprThresholdControl("range"));
els.mprTransparentThresholdValue.addEventListener("change", () => applyMprThresholdControl("number"));
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
for (const input of [els.mprPlaneToggle, els.mprPlaneOpacity, els.mprPlaneBoundaryToggle, els.foregroundAlpha, els.foregroundDistance, els.foregroundFadeWidth, els.mprThresholdAlphaToggle, els.mprTransparentThreshold]) {
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
els.windowLevel.addEventListener("input", () => applyWindowControl("level", "range"));
els.windowLevelValue.addEventListener("change", () => applyWindowControl("level", "number"));
els.windowWidth.addEventListener("input", () => applyWindowControl("width", "range"));
els.windowWidthValue.addEventListener("change", () => applyWindowControl("width", "number"));
for (const input of document.querySelectorAll("input, select")) {
  input.addEventListener("change", () => {
    if (!state.volume || input.type === "file") return;
    if (state.dynamicController) {
      requestLinkedMpr("control");
    } else {
      render();
    }
  });
}

const validationController = new ValidationController({
  elements: {
    entry: els.validationModeBtn,
    practiceEntry: document.querySelector("#validationPracticeBtn"),
    targetEntry: document.querySelector("#validationTargetSetupBtn"),
    shell: document.querySelector("#validationShell"),
    title: document.querySelector("#validationTitle"),
    participantBadge: document.querySelector("#validationParticipantBadge"),
    stepBadge: document.querySelector("#validationStepBadge"),
    setup: document.querySelector("#validationSetup"),
    setupDescription: document.querySelector("#validationSetupDescription"),
    task: document.querySelector("#validationTask"),
    targetEditor: document.querySelector("#validationTargetEditor"),
    complete: document.querySelector("#validationComplete"),
    form: document.querySelector("#validationSetupForm"),
    start: document.querySelector("#validationStartBtn"),
    participantField: document.querySelector("#validationParticipantField"),
    participant: document.querySelector("#validationParticipantId"),
    dataset: document.querySelector("#validationDataset"),
    allocation: document.querySelector("#validationAllocation"),
    status: document.querySelector("#validationSetupStatus"),
    cancel: document.querySelector("#validationCancelBtn"),
    condition: document.querySelector("#validationConditionLabel"),
    instructionText: document.querySelector("#validationInstructionText"),
    mprGrid: document.querySelector("#validationMprGrid"),
    sagittal: document.querySelector("#validationSagittal"),
    sagittalOffset: document.querySelector("#validationSagittalOffset"),
    sagittalOffsetValue: document.querySelector("#validationSagittalOffsetValue"),
    sagittalZoom: document.querySelector("#validationSagittalZoom"),
    sagittalZoomValue: document.querySelector("#validationSagittalZoomValue"),
    coronal: document.querySelector("#validationCoronal"),
    coronalOffset: document.querySelector("#validationCoronalOffset"),
    coronalOffsetValue: document.querySelector("#validationCoronalOffsetValue"),
    coronalZoom: document.querySelector("#validationCoronalZoom"),
    coronalZoomValue: document.querySelector("#validationCoronalZoomValue"),
    axial: document.querySelector("#validationAxial"),
    axialOffset: document.querySelector("#validationAxialOffset"),
    axialOffsetValue: document.querySelector("#validationAxialOffsetValue"),
    axialZoom: document.querySelector("#validationAxialZoom"),
    axialZoomValue: document.querySelector("#validationAxialZoomValue"),
    responsePanel: document.querySelector("#validationResponsePanel"),
    resetAnswer: document.querySelector("#validationResetAnswerBtn"),
    confirmAnswer: document.querySelector("#validationConfirmAnswerBtn"),
    endObservation: document.querySelector("#validationEndObservationBtn"),
    csv: document.querySelector("#validationCsvBtn"),
    json: document.querySelector("#validationJsonBtn"),
    finish: document.querySelector("#validationFinishBtn"),
    completeTitle: document.querySelector("#validationCompleteTitle"),
    completeText: document.querySelector("#validationCompleteText"),
    targetDataset: document.querySelector("#validationTargetDataset"),
    targetSlot: document.querySelector("#validationTargetSlot"),
    targetX: document.querySelector("#validationTargetX"),
    targetY: document.querySelector("#validationTargetY"),
    targetZ: document.querySelector("#validationTargetZ"),
    targetStatus: document.querySelector("#validationTargetStatus"),
    targetReset: document.querySelector("#validationTargetResetBtn"),
    targetSave: document.querySelector("#validationTargetSaveBtn"),
    targetExport: document.querySelector("#validationTargetExportBtn"),
    targetClose: document.querySelector("#validationTargetCloseBtn"),
  },
  getVolume: () => state.volume,
  getScene: () => state.threeScene,
  getMapper: () => state.mapper,
  getBaseMprRequest,
  renderMpr: (volume, request) => renderer.render(volume, request),
  loadDataset: async (datasetId) => {
    if (state.activeDemoId === datasetId && state.volume && state.threeScene?.getObjSummary?.()) return;
    const demo = DEMO_DATASETS[datasetId];
    if (!demo) throw new Error(`不明なデータセットです: ${datasetId}`);
    await loadBundledDemo(demo);
  },
  restoreViewer: () => {
    state.threeScene?.setValidationResponseMode(false);
    applyMprPlaneOverlayControls();
  },
});

initThree();
updateAxisFlipButtons();
updateMprUvFlipButtons();
updateMprOrientationUi();
updateVisualRangeOutputs();
updateMprFovControlsFromInputs();
writeDiagnostics();

function runFromFileMenu(action) {
  els.fileMenu.open = false;
  return action();
}

function runManualFromFileMenu(action) {
  cancelStartupDemo();
  setActiveDemoButton(null);
  return runFromFileMenu(action);
}

function loadDemoFromToolbar(demo) {
  if (state.observationLocked) return;
  cancelStartupDemo();
  els.fileMenu.open = false;
  void loadBundledDemo(demo);
}

function setActiveDemoButton(demoId) {
  state.activeDemoId = demoId;
  for (const [id, button] of [["mikan", els.mikanDemoBtn], ["banana", els.bananaDemoBtn]]) {
    button.classList.toggle("is-active", id === demoId);
    button.setAttribute("aria-pressed", String(id === demoId));
  }
}

function cancelStartupDemo() {
  if (!state.startupDemoLoading) return;
  state.startupDemoToken += 1;
  state.startupDemoLoading = false;
  hideStartupLoading();
  logger.info("手動読込を優先し、初期デモの読込を中止しました");
}

function updateStartupLoading(demo, step, detail) {
  if (!els.startupLoading) return;
  els.startupLoading.classList.remove("is-hidden");
  els.startupLoadingTitle.textContent = `${demo.name}デモを読み込んでいます`;
  els.startupLoadingDetail.textContent = detail;
  const order = ["model", "ct", "display"];
  const activeIndex = order.indexOf(step);
  for (const item of els.startupLoading.querySelectorAll("[data-loading-step]")) {
    const index = order.indexOf(item.dataset.loadingStep);
    item.classList.toggle("is-active", index === activeIndex);
    item.classList.toggle("is-complete", activeIndex >= 0 && index < activeIndex);
  }
}

function hideStartupLoading() {
  els.startupLoading?.classList.add("is-hidden");
}

function toggleControlsPane() {
  if (state.observationLocked) return;
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

function toggleObservationLock() {
  const locking = !state.observationLocked;
  if (locking && !els.workspace.classList.contains("controls-collapsed")) toggleControlsPane();
  state.observationLocked = locking;
  state.threeScene?.setForegroundSideLocked(locking);
  els.workspace.classList.toggle("observation-locked", locking);
  els.fileMenu.classList.toggle("is-locked", locking);
  els.fileMenu.open = false;
  els.fileMenu.inert = locking;
  els.fileMenu.querySelector("summary")?.setAttribute("aria-disabled", String(locking));
  els.controlsToggleBtn.disabled = locking;
  els.mikanDemoBtn.disabled = locking;
  els.bananaDemoBtn.disabled = locking;
  setContainerControlsLocked(els.controlsPane, locking);
  els.observationLockBtn.setAttribute("aria-pressed", String(locking));
  els.observationLockBtn.setAttribute("aria-label", locking ? "観察モードを解除" : "現在の状態をロック");
  els.observationLockBtn.title = locking ? "観察モードを解除" : "現在の状態をロック";
  if (!locking && state.volume) {
    if (shouldUseObjectMpr()) requestObjectLinkedMpr("observation-unlock");
    else requestLinkedMpr("observation-unlock");
  }
  writeDiagnostics();
}

function setContainerControlsLocked(container, locked) {
  for (const control of container.querySelectorAll("input, select, button")) {
    if (locked) {
      control.dataset.beforeObservationLockDisabled = String(control.disabled);
      control.disabled = true;
    } else {
      control.disabled = control.dataset.beforeObservationLockDisabled === "true";
      delete control.dataset.beforeObservationLockDisabled;
    }
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
    if (state.dynamicController) {
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
    if (state.dynamicController) {
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

function getObjOptions() {
  const controlPosition = [readFloat(els.objPosX, 0), readFloat(els.objPosY, 0), readFloat(els.objPosZ, 0)];
  const controlRotation = [readFloat(els.objRotX, 0), readFloat(els.objRotY, 0), readFloat(els.objRotZ, 0)];
  const position = applyAxisFlipToVector(controlPosition);
  const rotation = applyAxisFlipToVector(controlRotation);
  return {
    centerToTarget: false,
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
  if (state.dynamicController) {
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

function updateMprOrientationUi() {
  const labels = {
    view: "VIEW",
    axial: "AXIAL",
    coronal: "CORONAL",
    sagittal: "SAGITTAL",
  };
  const mode = els.mprOrientationMode.value;
  const fixed = mode !== "view";
  els.mprPaneTitle.textContent = `MPR · ${labels[mode] ?? "VIEW"}`;
  els.mprOrientationHint.textContent = fixed
    ? `${labels[mode]}方向を固定し、スライス位置だけがカメラ前方点に追従します。`
    : "断面方向と位置が視線に追従します。";
  updateConditionalSettingsUi();
  writeDiagnostics();
}

function updateConditionalSettingsUi() {
  els.mprTransparentThresholdControl.hidden = !els.mprThresholdAlphaToggle.checked;
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
  setWindowValues(40, 900);
  if (state.dynamicController) {
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
    const baseRequest = getRequest();
    const planeInput = constrainDicomInputToPlane(state.volume, baseRequest, els.mprOrientationMode.value);
    const request = {
      ...baseRequest,
      eye: planeInput.eye,
      target: planeInput.target,
      up: planeInput.up,
    };
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
  configureWindowControlBounds(volume);
  setWindowValues(state.defaults.wl, state.defaults.ww);
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
    void loadBundledDemo(STARTUP_DEMO, { startup: true });
  } catch (error) {
    state.threeReady = false;
    els.threeViewport.innerHTML = `<div class="three-placeholder">Three.jsを読み込めませんでした。静的MPRは利用できます。<br>${escapeHtml(error.message)}</div>`;
    logger.error("Three.js初期化失敗", { message: error.message });
    hideStartupLoading();
    writeDiagnostics();
  }
}

async function loadBundledDemo(demo, { startup = false } = {}) {
  const token = state.startupDemoToken + 1;
  state.startupDemoToken = token;
  state.startupDemoLoading = true;
  const startedAt = performance.now();
  try {
    setMprDistanceValue(demo.mprDistance);
    updateStartupLoading(demo, "model", "3Dモデルを読み込んでいます...");
    els.statusText.textContent = `${demo.name}デモのモデルを読込中...`;
    const glbBuffer = await fetchArrayBuffer(demo.glbUrl);
    if (token !== state.startupDemoToken) return;
    await state.threeScene.loadGlbArrayBuffer(glbBuffer, demo.modelFileName, getObjOptions());
    if (token !== state.startupDemoToken) return;

    // 先にモデルの描画機会を作り、その後に大容量CTを読み込む。
    await new Promise((resolve) => requestAnimationFrame(resolve));
    let mhaBuffer;
    try {
      updateStartupLoading(demo, "ct", "圧縮CTデータを読み込んでいます...");
      els.statusText.textContent = `${demo.name}デモのCTを読込中...`;
      const compressedMhaBuffer = await fetchArrayBuffer(demo.mhaUrl);
      if (token !== state.startupDemoToken) return;
      updateStartupLoading(demo, "ct", "CTデータをバックグラウンドで展開しています...");
      els.statusText.textContent = `${demo.name}デモのCTを展開中...`;
      mhaBuffer = await decompressGzipArrayBuffer(compressedMhaBuffer);
    } catch (compressedError) {
      if (token !== state.startupDemoToken) return;
      logger.warn("圧縮CTを利用できないため非圧縮データへ切り替えます", {
        demo: demo.id,
        message: compressedError.message,
      });
      updateStartupLoading(demo, "ct", "互換モードでCTデータを読み込んでいます...");
      els.statusText.textContent = `${demo.name}デモのCTを互換モードで読込中...`;
      mhaBuffer = await fetchArrayBuffer(demo.mhaFallbackUrl);
    }
    if (token !== state.startupDemoToken) return;
    state.volume = mhaLoader.parse(mhaBuffer, demo.volumeFileName);
    updateStartupLoading(demo, "display", "断面画像と3D表示を準備しています...");
    state.result = null;
    installVolumeDefaults(state.volume);
    setWindowValues(40, 400);
    els.windowPreset.value = "soft";
    configureThreeForVolume(state.volume);
    state.threeScene.focusObjModelAtDistance(readFloat(els.objMprDistance, DEFAULT_MPR_DISTANCE));
    updateSeriesInfo();
    requestLinkedMpr(`demo-${demo.id}`);
    setActiveDemoButton(demo.id);
    els.statusText.textContent = `${demo.name}デモ読込完了`;
    logger.info("デモデータを読み込みました", {
      demo: demo.id,
      glb: demo.glbUrl,
      mha: demo.mhaUrl,
      ms: Math.round(performance.now() - startedAt),
    });
    writeDiagnostics();
  } catch (error) {
    if (token !== state.startupDemoToken) return;
    const missing = error.status === 404;
    if (missing && startup) {
      logger.warn("初期デモが見つからないため通常起動します", { demo: demo.id, message: error.message });
      els.statusText.textContent = "待機中";
    } else {
      logger.error("デモデータの読込に失敗しました", { demo: demo.id, message: error.message });
      els.statusText.textContent = `${demo.name}デモ読込失敗: ${error.message}`;
    }
  } finally {
    if (token === state.startupDemoToken) {
      state.startupDemoLoading = false;
      hideStartupLoading();
    }
  }
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const error = new Error(`${url} (${response.status} ${response.statusText})`);
    error.status = response.status;
    throw error;
  }
  return response.arrayBuffer();
}

async function decompressGzipArrayBuffer(compressedBuffer) {
  if (typeof Worker !== "undefined") {
    return new Promise((resolve, reject) => {
      const workerUrl = new URL("./workers/GzipDecompressionWorker.js?v=demo-fallback-1", import.meta.url);
      const worker = new Worker(workerUrl, { type: "module" });
      let settled = false;
      const finish = (callback, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutId);
        worker.terminate();
        callback(value);
      };
      const timeoutId = setTimeout(() => {
        finish(reject, new Error("CTデータの展開がタイムアウトしました"));
      }, 12000);
      worker.addEventListener("message", (event) => {
        if (event.data?.error) {
          finish(reject, new Error(event.data.error));
          return;
        }
        if (!(event.data?.buffer instanceof ArrayBuffer)) {
          finish(reject, new Error("展開後のCTデータが不正です"));
          return;
        }
        finish(resolve, event.data.buffer);
      }, { once: true });
      worker.addEventListener("error", (event) => {
        finish(reject, new Error(`CTデータの展開に失敗しました: ${event.message || "Worker error"}`));
      }, { once: true });
      worker.postMessage({ buffer: compressedBuffer }, [compressedBuffer]);
    });
  }

  if (typeof DecompressionStream === "undefined") {
    throw new Error("このブラウザはgzip展開に対応していません");
  }
  const stream = new Blob([compressedBuffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

function installScheduler() {
  state.scheduler = new MprUpdateScheduler((reason) => {
    if (state.observationLocked || !state.volume || !state.dynamicController) return;
    if (shouldUseObjectMpr()) {
      const objInput = state.threeScene.getObjCameraInput(readFloat(els.objMprDistance, DEFAULT_MPR_DISTANCE));
      state.dynamicController.updateFromThreeInput(state.volume, objInput, reason);
    } else {
      state.dynamicController.update(state.volume, reason);
    }
  }, { minUpdateIntervalMs: viewerConfig.dynamicMpr.minUpdateIntervalMs });
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
    getPlaneMode: () => els.mprOrientationMode.value,
    logger,
    onResult: (info) => {
      state.dynamicInfo = info;
      state.result = info.result;
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
  state.dynamicController.setEnabled(true);
  updateAxisFlipButtons();
}

function updateMapperForVolume(volume) {
  const center = volumeCenter(volume.metadata);
  const flipMatrix = buildCenteredAxisFlipMatrix(center, state.volumeAxisFlip);
  state.mapper.setThreeToDicomMatrix(multiplyMat4(flipMatrix, viewerConfig.mapper.threeToDicomMatrix));
}

function requestDynamicMpr(reason) {
  if (state.observationLocked || !state.volume || !state.dynamicController) return;
  state.scheduler?.request(reason);
}

function requestLinkedMpr(reason) {
  requestDynamicMpr(reason);
}

function requestObjectLinkedMpr(reason) {
  if (shouldUseObjectMpr()) requestDynamicMpr(`model-${reason}`);
}

function shouldUseObjectMpr() {
  return Boolean(state.threeScene?.getObjSummary?.());
}

function applyMprPlaneOverlayControls() {
  updateConditionalSettingsUi();
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

function applyMprDistanceControl(source) {
  const sourceInput = source === "number" ? els.objMprDistanceValue : els.objMprDistance;
  setMprDistanceValue(readFloat(sourceInput, DEFAULT_MPR_DISTANCE));
  requestLinkedMpr("mpr-distance");
  writeDiagnostics();
}

function setMprDistanceValue(value) {
  const distance = Math.max(MPR_DISTANCE_MIN, Math.min(MPR_DISTANCE_MAX, Math.round(value / 10) * 10));
  els.objMprDistance.value = String(distance);
  els.objMprDistanceValue.value = String(distance);
  return distance;
}

function applyMprThresholdControl(source) {
  const sourceInput = source === "number" ? els.mprTransparentThresholdValue : els.mprTransparentThreshold;
  const threshold = Math.max(MPR_THRESHOLD_MIN, Math.min(MPR_THRESHOLD_MAX, Math.round(readFloat(sourceInput, -500) / 10) * 10));
  els.mprTransparentThreshold.value = String(threshold);
  els.mprTransparentThresholdValue.value = String(threshold);
  applyMprPlaneOverlayControls();
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
  const frontAlpha = Math.max(0, Math.min(1, readFloat(els.foregroundAlpha, 0)));
  return {
    visible: els.mprPlaneToggle.checked,
    opacity: Math.max(0, Math.min(1, readFloat(els.mprPlaneOpacity, 1))),
    displayMode: "world",
    cameraPlaneDistance: 0,
    flipU: state.mprUvFlip[0] < 0,
    flipV: state.mprUvFlip[1] < 0,
    showBoundary: els.mprPlaneBoundaryToggle.checked,
    foregroundEnabled: frontAlpha < 0.999,
    frontAlpha,
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
  setWindowValues(preset[0], preset[1]);
  render();
}

function applyWindowControl(kind, source) {
  const isLevel = kind === "level";
  const sourceInput = isLevel
    ? (source === "number" ? els.windowLevelValue : els.windowLevel)
    : (source === "number" ? els.windowWidthValue : els.windowWidth);
  const level = isLevel ? readFloat(sourceInput, 40) : readFloat(els.windowLevel, 40);
  const width = isLevel ? readFloat(els.windowWidth, 400) : readFloat(sourceInput, 400);
  setWindowValues(level, width);
  els.windowPreset.value = "";
  if (!state.volume) return;
  if (state.dynamicController) requestLinkedMpr(`window-${kind}`);
  else render();
}

function setWindowValues(level, width) {
  const levelMin = readFloat({ value: els.windowLevel.min }, WINDOW_LEVEL_MIN);
  const levelMax = readFloat({ value: els.windowLevel.max }, WINDOW_LEVEL_MAX);
  const widthMax = readFloat({ value: els.windowWidth.max }, WINDOW_WIDTH_MAX);
  const normalizedLevel = Math.max(levelMin, Math.min(levelMax, Math.round(level)));
  const normalizedWidth = Math.max(1, Math.min(widthMax, Math.round(width)));
  els.windowLevel.value = String(normalizedLevel);
  els.windowLevelValue.value = String(normalizedLevel);
  els.windowWidth.value = String(normalizedWidth);
  els.windowWidthValue.value = String(normalizedWidth);
}

function configureWindowControlBounds(volume) {
  const minimum = volume.metadata.minimumValue ?? volume.metadata.minimumHU ?? WINDOW_LEVEL_MIN;
  const maximum = volume.metadata.maximumValue ?? volume.metadata.maximumHU ?? WINDOW_LEVEL_MAX;
  const levelMin = Math.floor(Math.min(WINDOW_LEVEL_MIN, minimum));
  const levelMax = Math.ceil(Math.max(WINDOW_LEVEL_MAX, maximum));
  const widthMax = Math.ceil(Math.max(WINDOW_WIDTH_MAX, maximum - minimum));
  for (const input of [els.windowLevel, els.windowLevelValue]) {
    input.min = String(levelMin);
    input.max = String(levelMax);
  }
  for (const input of [els.windowWidth, els.windowWidthValue]) {
    input.min = "1";
    input.max = String(widthMax);
  }
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
    dynamicEnabled: true,
    observationLocked: state.observationLocked,
    mprOrientationMode: els.mprOrientationMode.value,
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
    input.value = Number(initial[index]).toFixed(3);
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
