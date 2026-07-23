const STORAGE_KEY = "viewsync.validation.phase1.sessions.v1";
const APP_VERSION = "validation-phase1-1";

export class ValidationController {
  constructor({
    elements,
    getVolume,
    getScene,
    getMapper,
    getBaseMprRequest,
    renderMpr,
    loadDataset,
    restoreViewer,
  }) {
    this.els = elements;
    this.getVolume = getVolume;
    this.getScene = getScene;
    this.getMapper = getMapper;
    this.getBaseMprRequest = getBaseMprRequest;
    this.renderMpr = renderMpr;
    this.loadDataset = loadDataset;
    this.restoreViewer = restoreViewer;
    this.session = null;
    this.trialIndex = -1;
    this.timerId = null;
    this.responseStartedAt = 0;
    this.answerDicom = null;
    this.pointerStart = null;
    this.installEvents();
  }

  installEvents() {
    this.els.entry.addEventListener("click", () => this.open());
    this.els.cancel.addEventListener("click", () => this.close());
    this.els.finish.addEventListener("click", () => this.close());
    this.els.form.addEventListener("submit", (event) => {
      event.preventDefault();
      void this.start();
    });
    this.els.endObservation.addEventListener("click", () => this.beginResponse());
    this.els.resetAnswer.addEventListener("click", () => this.resetAnswer());
    this.els.confirmAnswer.addEventListener("click", () => this.confirmAnswer());
    this.els.csv.addEventListener("click", () => this.downloadCsv());
    this.els.json.addEventListener("click", () => this.downloadJson());
  }

  open() {
    document.body.classList.add("validation-active");
    this.els.shell.hidden = false;
    this.els.shell.classList.remove("is-viewer-stage");
    this.showOnly("setup");
    this.els.participant.focus();
  }

  close() {
    this.stopTimer();
    this.detachPicking();
    this.getScene()?.clearValidationMarkers();
    this.restoreViewer();
    document.body.classList.remove("validation-active");
    this.els.shell.hidden = true;
    this.els.shell.classList.remove("is-viewer-stage");
  }

  async start() {
    const participantId = sanitizeId(this.els.participant.value);
    if (!participantId) {
      this.els.status.textContent = "研究用IDを入力してください。";
      return;
    }
    if (readSessions().some((session) => session.participant_id === participantId)) {
      this.els.status.textContent = "この研究用IDは既に使用されています。上書きはできません。";
      return;
    }
    this.els.status.textContent = "データと座標系を確認しています...";
    this.els.form.querySelector("button[type=submit]").disabled = true;
    try {
      const datasetId = this.els.dataset.value;
      const configResponse = await fetch("./data/validation/study-config.phase1.json");
      if (!configResponse.ok) throw new Error("研究設定を読み込めませんでした。");
      const config = await configResponse.json();
      await this.loadDataset(datasetId);
      const volume = this.getVolume();
      const scene = this.getScene();
      const mapper = this.getMapper();
      if (!volume || !scene?.getObjSummary?.() || !mapper) throw new Error("CT、モデル、座標変換を準備できませんでした。");
      const targetDefinitions = config.datasets?.[datasetId]?.targets;
      if (!Array.isArray(targetDefinitions) || targetDefinitions.length < 2) throw new Error("ターゲット定義が不足しています。");
      const targets = buildTargets(volume.metadata, targetDefinitions);
      const order = this.els.allocation.value === "B" ? ["proposal", "mpr"] : ["mpr", "proposal"];
      this.session = {
        study_id: config.study_id,
        config_version: config.config_version,
        app_version: APP_VERSION,
        dataset_id: datasetId,
        participant_id: participantId,
        session_id: `${participantId}-${Date.now()}`,
        allocation_group: this.els.allocation.value,
        observation_time_s: Number(config.observation_time_s) || 60,
        started_at: new Date().toISOString(),
        status: "in_progress",
        trials: order.map((condition, index) => ({
          trial_id: `${participantId}-${index + 1}`,
          sequence_no: index + 1,
          condition,
          target_id: targets[index].target_id,
          target_xyz_mm: targets[index].xyz,
          practice_flag: false,
          status: "ready",
        })),
        events: [],
      };
      this.log("session_started", { allocation_group: this.session.allocation_group });
      this.persist();
      this.els.participantBadge.textContent = participantId;
      this.trialIndex = -1;
      this.nextTrial();
    } catch (error) {
      this.els.status.textContent = `開始できません：${error.message}`;
    } finally {
      this.els.form.querySelector("button[type=submit]").disabled = false;
    }
  }

  nextTrial() {
    this.trialIndex += 1;
    if (this.trialIndex >= this.session.trials.length) {
      this.completeSession();
      return;
    }
    const trial = this.currentTrial();
    this.els.title.textContent = "位置対応理解課題";
    trial.status = "observing";
    trial.observation_started_at = new Date().toISOString();
    trial.operation_count = 0;
    this.answerDicom = null;
    this.getScene().clearValidationMarkers();
    this.getScene().showValidationTarget(trial.target_xyz_mm);
    this.showOnly("task");
    this.els.stepBadge.textContent = `TRIAL ${trial.sequence_no} / 2`;
    this.els.condition.textContent = trial.condition === "mpr" ? "MPR条件" : "提案ビューア条件";
    this.els.instructionText.textContent = "表示されたターゲット位置を観察してください";
    this.els.responsePanel.hidden = true;
    this.els.endObservation.hidden = false;
    this.els.shell.classList.toggle("is-viewer-stage", trial.condition === "proposal");
    this.els.mprGrid.hidden = trial.condition !== "mpr";
    if (trial.condition === "mpr") this.renderThreePlaneMpr(trial.target_xyz_mm);
    else this.restoreViewer({ validation: true });
    this.log("observation_started", { condition: trial.condition, target_id: trial.target_id });
    this.startTimer(this.session.observation_time_s);
  }

  renderThreePlaneMpr(target) {
    const volume = this.getVolume();
    const base = this.getBaseMprRequest();
    const definitions = [
      ["sagittal", [1, 0, 0], [0, 0, -1], this.els.sagittal],
      ["coronal", [0, 1, 0], [0, 0, -1], this.els.coronal],
      ["axial", [0, 0, 1], [0, 1, 0], this.els.axial],
    ];
    for (const [, normal, up, canvas] of definitions) {
      const request = {
        ...base,
        width: 384,
        height: 384,
        target: [...target],
        eye: target.map((value, index) => value - normal[index] * 500),
        up,
        normalOffset: 0,
      };
      const result = this.renderMpr(volume, request);
      drawMprWithTarget(canvas, result);
    }
  }

  startTimer(seconds) {
    this.stopTimer();
    let remaining = seconds;
    this.els.timer.textContent = formatTime(remaining);
    this.timerId = window.setInterval(() => {
      remaining -= 1;
      this.els.timer.textContent = formatTime(Math.max(0, remaining));
      if (remaining <= 0) this.beginResponse();
    }, 1000);
  }

  stopTimer() {
    if (this.timerId) window.clearInterval(this.timerId);
    this.timerId = null;
  }

  beginResponse() {
    const trial = this.currentTrial();
    if (!trial || trial.status !== "observing") return;
    this.stopTimer();
    trial.observation_ended_at = new Date().toISOString();
    trial.final_camera_pose = this.getScene().getCameraPose();
    trial.status = "responding";
    this.getScene().clearValidationMarkers();
    this.getScene().setValidationResponseMode(true);
    this.els.shell.classList.add("is-viewer-stage");
    this.els.mprGrid.hidden = true;
    this.els.condition.textContent = "回答";
    this.els.instructionText.textContent = "可視光モデル上で位置を指定してください";
    this.els.timer.textContent = "";
    this.els.endObservation.hidden = true;
    this.els.responsePanel.hidden = false;
    this.els.confirmAnswer.disabled = true;
    this.responseStartedAt = performance.now();
    this.attachPicking();
    this.log("response_started", { trial_id: trial.trial_id });
    this.persist();
  }

  attachPicking() {
    const canvas = this.getScene()?.renderer?.domElement;
    if (!canvas || this.pickHandlers) return;
    let draggingGizmo = false;
    const down = (event) => {
      if (event.button !== 0) return;
      draggingGizmo = this.getScene().beginValidationMarkerDrag(event.clientX, event.clientY);
      if (draggingGizmo) {
        canvas.setPointerCapture?.(event.pointerId);
        event.preventDefault();
      }
      this.pointerStart = draggingGizmo ? null : [event.clientX, event.clientY];
    };
    const move = (event) => {
      if (!draggingGizmo) return;
      this.getScene().updateValidationMarkerDrag(event.clientX, event.clientY);
      event.preventDefault();
    };
    const up = (event) => {
      if (draggingGizmo) {
        draggingGizmo = false;
        if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
        const point = this.getScene().endValidationMarkerDrag();
        if (point) {
          const trial = this.currentTrial();
          this.answerDicom = this.getMapper().threePointToDicom(point);
          trial.reposition_count = (trial.reposition_count ?? 0) + 1;
          this.els.instructionText.textContent = "X・Y・Z軸で調整し、回答を確定してください";
          this.log("answer_gizmo_moved", { reposition_count: trial.reposition_count });
        }
        return;
      }
      if (!this.pointerStart || Math.hypot(event.clientX - this.pointerStart[0], event.clientY - this.pointerStart[1]) > 5) return;
      const point = this.getScene().pickModelPoint(event.clientX, event.clientY);
      if (!point) {
        this.els.instructionText.textContent = "モデル表面を選択してください";
        this.log("invalid_pick", {});
        return;
      }
      const trial = this.currentTrial();
      if (this.answerDicom) trial.reposition_count = (trial.reposition_count ?? 0) + 1;
      this.answerDicom = this.getMapper().threePointToDicom(point);
      this.getScene().showValidationAnswer(this.answerDicom);
      this.els.confirmAnswer.disabled = false;
      this.els.instructionText.textContent = "X・Y・Z軸で位置を調整できます";
      this.log("answer_placed", { reposition_count: trial.reposition_count ?? 0 });
    };
    const cancel = (event) => {
      if (!draggingGizmo) return;
      draggingGizmo = false;
      if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      this.getScene().endValidationMarkerDrag();
    };
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", cancel);
    this.pickHandlers = { canvas, down, move, up, cancel };
  }

  detachPicking() {
    if (!this.pickHandlers) return;
    const { canvas, down, move, up, cancel } = this.pickHandlers;
    canvas.removeEventListener("pointerdown", down);
    canvas.removeEventListener("pointermove", move);
    canvas.removeEventListener("pointerup", up);
    canvas.removeEventListener("pointercancel", cancel);
    this.getScene()?.endValidationMarkerDrag();
    this.pickHandlers = null;
  }

  resetAnswer() {
    this.answerDicom = null;
    this.getScene().clearValidationAnswer();
    this.els.confirmAnswer.disabled = true;
    this.els.instructionText.textContent = "可視光モデル上で位置を指定してください";
    this.log("answer_reset", {});
  }

  confirmAnswer() {
    if (!this.answerDicom) return;
    this.detachPicking();
    const trial = this.currentTrial();
    const target = trial.target_xyz_mm;
    const calculation = calculateValidationError(this.answerDicom, target);
    trial.answer_xyz_mm = [...this.answerDicom];
    trial.delta_xyz_mm = calculation.delta;
    trial.error_3d_mm = calculation.error3d;
    trial.response_time_ms = Math.round(performance.now() - this.responseStartedAt);
    trial.reposition_count ??= 0;
    trial.answered_at = new Date().toISOString();
    trial.status = "completed";
    this.log("answer_confirmed", { trial_id: trial.trial_id });
    this.persist();
    this.nextTrial();
  }

  completeSession() {
    this.stopTimer();
    this.detachPicking();
    this.getScene().clearValidationMarkers();
    this.getScene().setValidationResponseMode(false);
    this.session.status = "completed";
    this.session.completed_at = new Date().toISOString();
    this.log("session_completed", {});
    this.persist();
    this.els.shell.classList.remove("is-viewer-stage");
    this.els.title.textContent = "検証完了";
    this.showOnly("complete");
    this.els.stepBadge.textContent = "COMPLETED";
  }

  showOnly(name) {
    this.els.setup.hidden = name !== "setup";
    this.els.task.hidden = name !== "task";
    this.els.complete.hidden = name !== "complete";
  }

  currentTrial() {
    return this.session?.trials?.[this.trialIndex] ?? null;
  }

  log(event_type, detail) {
    this.session?.events.push({
      timestamp: new Date().toISOString(),
      session_id: this.session.session_id,
      trial_id: this.currentTrial()?.trial_id ?? null,
      event_type,
      detail,
    });
  }

  persist() {
    const sessions = readSessions();
    const index = sessions.findIndex((item) => item.session_id === this.session.session_id);
    if (index >= 0) sessions[index] = this.session;
    else sessions.push(this.session);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  }

  downloadCsv() {
    const rows = this.session.trials.map((trial) => trialToFlatRecord(this.session, trial));
    const headers = Object.keys(rows[0]);
    const csv = [headers.join(","), ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(","))].join("\r\n");
    download(`viewsync-${this.session.session_id}.csv`, `\uFEFF${csv}`, "text/csv;charset=utf-8");
  }

  downloadJson() {
    download(`viewsync-${this.session.session_id}.json`, JSON.stringify(this.session, null, 2), "application/json");
  }
}

function buildTargets(metadata, definitions) {
  const [width, height, depth] = metadata.dimensions;
  return definitions.map((definition) => ({
    target_id: definition.target_id,
    xyz: voxelToDicom(metadata, [
      (width - 1) * definition.voxel_fraction[0],
      (height - 1) * definition.voxel_fraction[1],
      (depth - 1) * definition.voxel_fraction[2],
    ]),
  }));
}

function voxelToDicom(metadata, voxel) {
  const scaled = voxel.map((value, index) => value * metadata.spacing[index]);
  const d = metadata.direction;
  return [
    metadata.origin[0] + d[0] * scaled[0] + d[1] * scaled[1] + d[2] * scaled[2],
    metadata.origin[1] + d[3] * scaled[0] + d[4] * scaled[1] + d[5] * scaled[2],
    metadata.origin[2] + d[6] * scaled[0] + d[7] * scaled[1] + d[8] * scaled[2],
  ];
}

function drawMprWithTarget(canvas, result) {
  canvas.width = result.width;
  canvas.height = result.height;
  const context = canvas.getContext("2d");
  context.putImageData(new ImageData(result.rgba, result.width, result.height), 0, 0);
  const x = result.width / 2;
  const y = result.height / 2;
  context.strokeStyle = "#62f0a7";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x - 14, y);
  context.lineTo(x + 14, y);
  context.moveTo(x, y - 14);
  context.lineTo(x, y + 14);
  context.stroke();
  context.beginPath();
  context.arc(x, y, 7, 0, Math.PI * 2);
  context.stroke();
}

function trialToFlatRecord(session, trial) {
  return {
    study_id: session.study_id,
    config_version: session.config_version,
    app_version: session.app_version,
    dataset_id: session.dataset_id,
    participant_id: session.participant_id,
    session_id: session.session_id,
    trial_id: trial.trial_id,
    condition: trial.condition,
    sequence_no: trial.sequence_no,
    target_id: trial.target_id,
    practice_flag: trial.practice_flag,
    target_x_mm: fixed(trial.target_xyz_mm[0]),
    target_y_mm: fixed(trial.target_xyz_mm[1]),
    target_z_mm: fixed(trial.target_xyz_mm[2]),
    answer_x_mm: fixed(trial.answer_xyz_mm[0]),
    answer_y_mm: fixed(trial.answer_xyz_mm[1]),
    answer_z_mm: fixed(trial.answer_xyz_mm[2]),
    dx_mm: fixed(trial.delta_xyz_mm[0]),
    dy_mm: fixed(trial.delta_xyz_mm[1]),
    dz_mm: fixed(trial.delta_xyz_mm[2]),
    error_3d_mm: fixed(trial.error_3d_mm),
    response_time_ms: trial.response_time_ms,
    reposition_count: trial.reposition_count,
    status: trial.status,
  };
}

function readSessions() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function csvCell(value) {
  let text = String(value ?? "");
  if (/^[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

function sanitizeId(value) {
  return value.trim().replace(/[^\w.-]/g, "-").slice(0, 64);
}

function fixed(value) {
  return Number(value).toFixed(3);
}

function formatTime(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function calculateValidationError(answer, target) {
  if (!Array.isArray(answer) || !Array.isArray(target) || answer.length !== 3 || target.length !== 3) {
    throw new Error("answer and target must be 3D coordinates");
  }
  const delta = answer.map((value, index) => value - target[index]);
  return { delta, error3d: Math.hypot(...delta) };
}
