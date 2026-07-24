const STORAGE_KEY = "viewsync.validation.phase1.sessions.v1";
const TARGETS_KEY = "viewsync.validation.phase1.targets.v1";
const APP_VERSION = "validation-phase1-2";

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
    this.responseStartedAt = 0;
    this.answerDicom = null;
    this.pointerStart = null;
    this.mprTarget = null;
    this.isPractice = false;
    this.config = null;
    this.targetDraftDicom = null;
    this.targetBaseDicom = null;
    this.pickingPurpose = null;
    this.mprDefinitions = [
      {
        id: "sagittal", color: "#ff6b7a",
        normal: [1, 0, 0], up: [0, 0, -1], canvas: elements.sagittal,
        slider: elements.sagittalOffset, output: elements.sagittalOffsetValue,
        zoom: elements.sagittalZoom, zoomOutput: elements.sagittalZoomValue,
      },
      {
        id: "coronal", color: "#62f0a7",
        normal: [0, 1, 0], up: [0, 0, -1], canvas: elements.coronal,
        slider: elements.coronalOffset, output: elements.coronalOffsetValue,
        zoom: elements.coronalZoom, zoomOutput: elements.coronalZoomValue,
      },
      {
        id: "axial", color: "#59a8ff",
        normal: [0, 0, 1], up: [0, 1, 0], canvas: elements.axial,
        slider: elements.axialOffset, output: elements.axialOffsetValue,
        zoom: elements.axialZoom, zoomOutput: elements.axialZoomValue,
      },
    ];
    this.installEvents();
  }

  installEvents() {
    this.els.entry.addEventListener("click", () => this.open(false));
    this.els.practiceEntry.addEventListener("click", () => this.open(true));
    this.els.targetEntry.addEventListener("click", () => void this.openTargetEditor());
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
    this.els.targetDataset.addEventListener("change", () => {
      void this.prepareTargetEditorDataset().catch((error) => {
        this.els.targetStatus.textContent = `データを変更できません：${error.message}`;
      });
    });
    this.els.targetSlot.addEventListener("change", () => this.loadTargetDraft());
    this.els.targetReset.addEventListener("click", () => this.resetTargetDraft());
    this.els.targetSave.addEventListener("click", () => this.saveTargetDraft());
    this.els.targetExport.addEventListener("click", () => this.exportTargetOverrides());
    this.els.targetClose.addEventListener("click", () => this.close());
    for (const definition of this.mprDefinitions) {
      definition.slider.addEventListener("input", () => this.renderAllMprPlanes());
      definition.zoom.addEventListener("input", () => this.renderMprPlane(definition));
      definition.canvas.addEventListener("wheel", (event) => {
        if (this.els.mprGrid.hidden) return;
        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 : -1;
        if (event.ctrlKey || event.metaKey) {
          const zoom = Math.max(
            Number(definition.zoom.min),
            Math.min(Number(definition.zoom.max), Number(definition.zoom.value) - direction * 0.1)
          );
          definition.zoom.value = zoom.toFixed(1);
          this.renderMprPlane(definition);
          return;
        }
        const next = Math.max(
          Number(definition.slider.min),
          Math.min(Number(definition.slider.max), Number(definition.slider.value) + direction)
        );
        definition.slider.value = String(next);
        this.renderAllMprPlanes();
      }, { passive: false });
    }
  }

  open(practice = false) {
    this.isPractice = practice;
    document.body.classList.add("validation-active");
    this.els.shell.hidden = false;
    this.els.shell.classList.remove("is-viewer-stage");
    this.els.shell.classList.remove("is-response-stage");
    this.els.shell.classList.remove("is-result-stage");
    this.showOnly("setup");
    this.els.title.textContent = practice ? "練習モード設定" : "検証モード設定";
    this.els.participantBadge.textContent = practice ? "練習" : "未開始";
    this.els.stepBadge.textContent = "SETUP";
    this.els.participantField.hidden = practice;
    this.els.participant.required = !practice;
    this.els.start.textContent = practice ? "練習を開始" : "検証を開始";
    this.els.setupDescription.textContent = practice
      ? "研究用IDを使わず、MPR条件と提案ビューア条件の一連の操作を練習します。"
      : "氏名は入力せず、研究用IDのみを使用します。2条件の観察後、可視光モデル上で記憶した位置を回答します。";
    this.els.csv.hidden = false;
    this.els.json.hidden = false;
    this.els.status.textContent = practice
      ? "練習結果は保存されません。MPR・提案ビューアの両方を操作できます。"
      : "開始時にデータと座標系を確認します。";
    if (!practice) this.els.participant.focus();
  }

  async openTargetEditor() {
    this.isPractice = false;
    document.body.classList.add("validation-active");
    this.els.shell.hidden = false;
    this.els.shell.classList.add("is-viewer-stage");
    this.els.shell.classList.remove("is-response-stage");
    this.els.shell.classList.remove("is-result-stage");
    this.showOnly("targetEditor");
    this.els.title.textContent = "正解座標設定";
    this.els.participantBadge.textContent = "研究担当者";
    this.els.stepBadge.textContent = "TARGET SETUP";
    this.els.targetDataset.value = this.els.dataset.value;
    this.els.targetStatus.textContent = "データと座標系を確認しています...";
    try {
      await this.loadConfig();
      this.populateTargetSlots();
      await this.prepareTargetEditorDataset();
    } catch (error) {
      this.els.targetStatus.textContent = `正解座標設定を開始できません：${error.message}`;
    }
  }

  populateTargetSlots() {
    const datasetConfig = this.config?.datasets?.[this.els.targetDataset.value];
    const definitions = [
      ...(datasetConfig?.practice_targets ?? []).map((definition, index) => ({
        ...definition,
        label: `練習ターゲット ${index + 1}`,
      })),
      ...(datasetConfig?.targets ?? []).map((definition, index) => ({
        ...definition,
        label: `本番ターゲット ${index + 1}`,
      })),
    ];
    const previous = this.els.targetSlot.value;
    this.els.targetSlot.replaceChildren(...definitions.map((definition) => {
      const option = document.createElement("option");
      option.value = definition.target_id;
      option.textContent = definition.label;
      option.dataset.definition = JSON.stringify(definition);
      return option;
    }));
    if (definitions.some((definition) => definition.target_id === previous)) {
      this.els.targetSlot.value = previous;
    }
  }

  async prepareTargetEditorDataset() {
    this.detachPicking();
    this.getScene()?.clearValidationMarkers();
    this.els.targetSave.disabled = true;
    this.els.targetStatus.textContent = "データを読み込んでいます...";
    this.populateTargetSlots();
    const datasetId = this.els.targetDataset.value;
    await this.loadDataset(datasetId);
    if (!this.getVolume() || !this.getScene()?.getObjSummary?.() || !this.getMapper()) {
      throw new Error("CT、モデル、座標変換を準備できませんでした。");
    }
    this.getScene().setValidationResponseMode(false);
    this.restoreViewer({ validation: true });
    this.loadTargetDraft();
  }

  loadTargetDraft() {
    const option = this.els.targetSlot.selectedOptions[0];
    if (!option || !this.getVolume()) return;
    const definition = JSON.parse(option.dataset.definition);
    const datasetId = this.els.targetDataset.value;
    const override = readTargetOverrides()[datasetId]?.[definition.target_id];
    const target = buildTargets(
      this.getVolume().metadata,
      [definition],
      override ? { [definition.target_id]: override } : {}
    )[0];
    this.targetBaseDicom = [...target.xyz];
    this.targetDraftDicom = [...target.xyz];
    this.getScene().showValidationAnswer(this.targetDraftDicom);
    this.updateTargetReadout();
    this.els.targetSave.disabled = false;
    this.els.targetStatus.textContent = override
      ? "このブラウザに保存された正解座標を表示しています。"
      : "研究設定の初期座標を表示しています。ギズモで調整してください。";
    this.attachPicking("target");
  }

  resetTargetDraft() {
    if (!this.targetBaseDicom) return;
    this.targetDraftDicom = [...this.targetBaseDicom];
    this.getScene().showValidationAnswer(this.targetDraftDicom);
    this.updateTargetReadout();
    this.els.targetStatus.textContent = "編集開始時の座標に戻しました。";
  }

  saveTargetDraft() {
    if (!this.targetDraftDicom || !this.els.targetSlot.value) return;
    const overrides = readTargetOverrides();
    const datasetId = this.els.targetDataset.value;
    overrides[datasetId] ??= {};
    overrides[datasetId][this.els.targetSlot.value] = this.targetDraftDicom.map(Number);
    localStorage.setItem(TARGETS_KEY, JSON.stringify(overrides));
    this.targetBaseDicom = [...this.targetDraftDicom];
    this.els.targetStatus.textContent = "正解座標をこのブラウザに保存しました。本番・練習で優先使用されます。";
  }

  exportTargetOverrides() {
    const payload = {
      format: "viewsync-validation-targets",
      version: 1,
      exported_at: new Date().toISOString(),
      datasets: readTargetOverrides(),
    };
    download("viewsync-validation-targets.json", JSON.stringify(payload, null, 2), "application/json");
  }

  updateTargetReadout() {
    const values = this.targetDraftDicom ?? [];
    [this.els.targetX, this.els.targetY, this.els.targetZ].forEach((output, index) => {
      output.textContent = Number.isFinite(values[index]) ? `${values[index].toFixed(2)} mm` : "-";
      output.title = output.textContent;
    });
  }

  close() {
    this.detachPicking();
    this.getScene()?.clearValidationMarkers();
    this.restoreViewer();
    document.body.classList.remove("validation-active");
    this.els.shell.hidden = true;
    this.els.shell.classList.remove("is-viewer-stage");
    this.els.shell.classList.remove("is-response-stage");
    this.els.shell.classList.remove("is-result-stage");
    this.targetDraftDicom = null;
    this.targetBaseDicom = null;
  }

  async start() {
    const participantId = this.isPractice
      ? `PRACTICE-${Date.now()}`
      : sanitizeId(this.els.participant.value);
    if (!this.isPractice && !participantId) {
      this.els.status.textContent = "研究用IDを入力してください。";
      return;
    }
    if (!this.isPractice && readSessions().some((session) => session.participant_id === participantId)) {
      this.els.status.textContent = "この研究用IDは既に使用されています。上書きはできません。";
      return;
    }
    this.els.status.textContent = "データと座標系を確認しています...";
    this.els.form.querySelector("button[type=submit]").disabled = true;
    try {
      const datasetId = this.els.dataset.value;
      const config = await this.loadConfig();
      await this.loadDataset(datasetId);
      const volume = this.getVolume();
      const scene = this.getScene();
      const mapper = this.getMapper();
      if (!volume || !scene?.getObjSummary?.() || !mapper) throw new Error("CT、モデル、座標変換を準備できませんでした。");
      const datasetConfig = config.datasets?.[datasetId];
      const targetDefinitions = this.isPractice ? datasetConfig?.practice_targets : datasetConfig?.targets;
      const minimumTargetCount = this.isPractice ? 1 : 2;
      if (!Array.isArray(targetDefinitions) || targetDefinitions.length < minimumTargetCount) {
        throw new Error(this.isPractice ? "練習用ターゲット定義が不足しています。" : "ターゲット定義が不足しています。");
      }
      const targets = buildTargets(volume.metadata, targetDefinitions, readTargetOverrides()[datasetId]);
      const order = this.els.allocation.value === "B" ? ["proposal", "mpr"] : ["mpr", "proposal"];
      this.session = {
        study_id: config.study_id,
        config_version: config.config_version,
        app_version: APP_VERSION,
        dataset_id: datasetId,
        participant_id: participantId,
        session_id: `${participantId}-${Date.now()}`,
        allocation_group: this.els.allocation.value,
        practice_mode: this.isPractice,
        started_at: new Date().toISOString(),
        status: "in_progress",
        trials: order.map((condition, index) => ({
          trial_id: `${participantId}-${index + 1}`,
          sequence_no: index + 1,
          condition,
          target_id: (targets[index] ?? targets[0]).target_id,
          target_xyz_mm: (targets[index] ?? targets[0]).xyz,
          practice_flag: this.isPractice,
          status: "ready",
        })),
        events: [],
      };
      this.log("session_started", { allocation_group: this.session.allocation_group });
      this.persist();
      this.els.participantBadge.textContent = this.isPractice ? "練習" : participantId;
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
    this.els.title.textContent = this.isPractice ? "操作練習" : "位置対応理解課題";
    trial.status = "observing";
    trial.observation_started_at = new Date().toISOString();
    trial.operation_count = 0;
    this.answerDicom = null;
    this.getScene().setValidationResponseMode(false);
    this.getScene().clearValidationMarkers();
    this.getScene().showValidationTarget(trial.target_xyz_mm, {
      attachToMpr: trial.condition === "proposal",
    });
    this.showOnly("task");
    this.els.stepBadge.textContent = `${this.isPractice ? "PRACTICE" : "TRIAL"} ${trial.sequence_no} / ${this.session.trials.length}`;
    this.els.condition.textContent = trial.condition === "mpr" ? "MPR条件" : "提案ビューア条件";
    this.els.instructionText.textContent = "表示されたターゲット位置を観察してください";
    this.els.responsePanel.hidden = true;
    this.els.endObservation.hidden = false;
    this.els.shell.classList.remove("is-response-stage");
    this.els.shell.classList.remove("is-result-stage");
    this.els.shell.classList.toggle("is-viewer-stage", trial.condition === "proposal");
    this.els.mprGrid.hidden = trial.condition !== "mpr";
    if (trial.condition === "mpr") this.renderThreePlaneMpr(trial.target_xyz_mm);
    else this.restoreViewer({ validation: true });
    this.log("observation_started", { condition: trial.condition, target_id: trial.target_id });
  }

  renderThreePlaneMpr(target) {
    const volume = this.getVolume();
    this.mprTarget = [...target];
    for (const definition of this.mprDefinitions) {
      const range = calculateSliceOffsetRange(volume.metadata, target, definition.normal);
      definition.slider.min = String(Math.ceil(range.min));
      definition.slider.max = String(Math.floor(range.max));
      definition.slider.value = "0";
      definition.zoom.value = "2";
    }
    this.renderAllMprPlanes();
  }

  renderAllMprPlanes() {
    for (const definition of this.mprDefinitions) this.renderMprPlane(definition);
  }

  renderMprPlane(definition) {
    const volume = this.getVolume();
    if (!volume || !this.mprTarget) return;
    const offsetMm = Number(definition.slider.value) || 0;
    const zoom = Math.max(1, Number(definition.zoom.value) || 1);
    const planeTarget = this.mprTarget.map((value, index) => value + definition.normal[index] * offsetMm);
    const baseRequest = this.getBaseMprRequest();
    const request = {
      ...baseRequest,
      width: 384,
      height: 384,
      spacingU: baseRequest.spacingU / zoom,
      spacingV: baseRequest.spacingV / zoom,
      target: planeTarget,
      eye: planeTarget.map((value, index) => value - definition.normal[index] * 500),
      up: definition.up,
      normalOffset: 0,
    };
    const result = this.renderMpr(volume, request);
    drawLinkedMprCrosshair(
      definition.canvas,
      result,
      request,
      this.getMprIntersectionPoint(),
      this.mprTarget,
      offsetMm,
      Math.max(3, (this.getScene()?.getValidationTargetRadiusDicom?.() ?? 6) * 0.5),
      definition,
      this.mprDefinitions,
    );
    definition.output.textContent = offsetMm === 0
      ? "基準断面"
      : `基準から ${offsetMm > 0 ? "+" : ""}${offsetMm} mm`;
    definition.zoomOutput.textContent = `${Math.round(zoom * 100)}%`;
  }

  getMprIntersectionPoint() {
    return this.mprDefinitions.reduce((point, definition) => {
      const offsetMm = Number(definition.slider.value) || 0;
      return point.map((value, index) => value + definition.normal[index] * offsetMm);
    }, [...this.mprTarget]);
  }

  beginResponse() {
    const trial = this.currentTrial();
    if (!trial || trial.status !== "observing") return;
    trial.observation_ended_at = new Date().toISOString();
    trial.observation_time_ms = calculateElapsedTimeMs(
      trial.observation_started_at,
      trial.observation_ended_at,
    );
    trial.final_camera_pose = this.getScene().getCameraPose();
    trial.status = "responding";
    this.getScene().clearValidationMarkers();
    this.getScene().setValidationResponseMode(true);
    // 観察時のカメラ姿勢は回答位置への手掛かりになるため引き継がず、
    // 条件や操作履歴に依存しない標準姿勢から回答を開始する。
    trial.response_camera_pose = this.getScene().resetValidationResponseCamera();
    this.els.shell.classList.add("is-viewer-stage");
    this.els.shell.classList.add("is-response-stage");
    this.els.mprGrid.hidden = true;
    this.els.condition.textContent = "回答";
    this.els.instructionText.textContent = "可視光モデル上で位置を指定してください";
    this.els.endObservation.hidden = true;
    this.els.responsePanel.hidden = false;
    this.initializeAnswerGizmo();
    this.responseStartedAt = performance.now();
    this.attachPicking("answer");
    this.log("response_started", { trial_id: trial.trial_id });
    this.persist();
  }

  attachPicking(purpose = "answer") {
    const canvas = this.getScene()?.renderer?.domElement;
    this.pickingPurpose = purpose;
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
          const dicomPoint = this.getMapper().threePointToDicom(point);
          if (this.pickingPurpose === "target") {
            this.targetDraftDicom = dicomPoint;
            this.updateTargetReadout();
            this.els.targetStatus.textContent = "座標を編集中です。「この座標を保存」で確定してください。";
          } else {
            const trial = this.currentTrial();
            this.answerDicom = dicomPoint;
            trial.reposition_count = (trial.reposition_count ?? 0) + 1;
            this.els.instructionText.textContent = "X・Y・Z軸で調整し、回答を確定してください";
            this.log("answer_gizmo_moved", { reposition_count: trial.reposition_count });
          }
        }
        return;
      }
      if (!this.pointerStart || Math.hypot(event.clientX - this.pointerStart[0], event.clientY - this.pointerStart[1]) > 5) return;
      const point = this.getScene().pickModelPoint(event.clientX, event.clientY);
      if (!point) {
        if (this.pickingPurpose === "target") {
          this.els.targetStatus.textContent = "モデル表面またはXYZギズモを選択してください。";
        } else {
          this.els.instructionText.textContent = "モデル表面を選択してください";
          this.log("invalid_pick", {});
        }
        return;
      }
      const dicomPoint = this.getMapper().threePointToDicom(point);
      if (this.pickingPurpose === "target") {
        this.targetDraftDicom = dicomPoint;
        this.getScene().showValidationAnswer(this.targetDraftDicom);
        this.updateTargetReadout();
        this.els.targetStatus.textContent = "モデル表面へ移動しました。XYZギズモで奥行きを調整できます。";
      } else {
        const trial = this.currentTrial();
        if (this.answerDicom) trial.reposition_count = (trial.reposition_count ?? 0) + 1;
        this.answerDicom = dicomPoint;
        this.getScene().showValidationAnswer(this.answerDicom);
        this.els.confirmAnswer.disabled = false;
        this.els.instructionText.textContent = "X・Y・Z軸で位置を調整できます";
        this.log("answer_placed", { reposition_count: trial.reposition_count ?? 0 });
      }
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
    this.pickingPurpose = null;
  }

  resetAnswer() {
    this.initializeAnswerGizmo();
    this.els.instructionText.textContent = "初期位置に戻しました。X・Y・Z軸で調整してください";
    this.log("answer_reset", {});
  }

  initializeAnswerGizmo() {
    const point = this.getScene()?.getValidationInitialAnswerPoint?.();
    if (!point || !this.getMapper()) {
      this.answerDicom = null;
      this.getScene()?.clearValidationAnswer();
      this.els.confirmAnswer.disabled = true;
      return;
    }
    this.answerDicom = this.getMapper().threePointToDicom(point);
    this.getScene().showValidationAnswer(this.answerDicom);
    this.els.confirmAnswer.disabled = false;
    this.els.instructionText.textContent = "中央のXYZギズモを動かすか、モデル表面をクリックしてください";
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
    this.detachPicking();
    this.getScene().clearValidationMarkers();
    this.getScene().setValidationResponseMode(true);
    this.session.status = "completed";
    this.session.completed_at = new Date().toISOString();
    this.log("session_completed", {});
    this.persist();
    this.els.shell.classList.add("is-viewer-stage");
    this.els.shell.classList.remove("is-response-stage");
    this.els.shell.classList.add("is-result-stage");
    this.els.title.textContent = this.isPractice ? "練習完了" : "検証完了";
    this.els.completeTitle.textContent = this.isPractice ? "練習が完了しました" : "検証が完了しました";
    this.renderResultReview();
    this.showOnly("complete");
    this.els.stepBadge.textContent = this.isPractice ? "PRACTICE DONE" : "COMPLETED";
    this.els.completeText.textContent = this.isPractice
      ? "練習結果を確認できます。結果は保存されません。確認後、通常モードへ戻って本番を開始できます。"
      : "結果を確認し、必要に応じてCSVまたはJSONとして保存してください。";
    this.els.csv.hidden = this.isPractice;
    this.els.json.hidden = this.isPractice;
  }

  renderResultReview() {
    const trials = this.session?.trials?.filter((trial) => trial.status === "completed") ?? [];
    const fragment = document.createDocumentFragment();
    for (const trial of trials) {
      const card = document.createElement("article");
      card.className = "validation-result-card";

      const header = document.createElement("header");
      const condition = document.createElement("strong");
      condition.textContent = trial.condition === "mpr" ? "MPR条件" : "提案ビューア条件";
      const sequence = document.createElement("span");
      sequence.textContent = `${this.isPractice ? "練習" : "試行"} ${trial.sequence_no}`;
      header.append(condition, sequence);

      const error = document.createElement("div");
      error.className = "validation-result-error";
      const errorLabel = document.createElement("span");
      errorLabel.textContent = "3D誤差";
      const errorValue = document.createElement("strong");
      errorValue.textContent = `${Number(trial.error_3d_mm).toFixed(1)} mm`;
      error.append(errorLabel, errorValue);

      const metrics = document.createElement("dl");
      metrics.className = "validation-result-metrics";
      const metricValues = [
        ["ΔX", `${signedFixed(trial.delta_xyz_mm[0])} mm`],
        ["ΔY", `${signedFixed(trial.delta_xyz_mm[1])} mm`],
        ["ΔZ", `${signedFixed(trial.delta_xyz_mm[2])} mm`],
        ["観察時間", `${(getObservationTimeMs(trial) / 1000).toFixed(1)} 秒`],
        ["回答時間", `${(trial.response_time_ms / 1000).toFixed(1)} 秒`],
      ];
      for (const [label, value] of metricValues) {
        const metric = document.createElement("div");
        metric.className = "validation-result-metric";
        const term = document.createElement("dt");
        term.textContent = label;
        const description = document.createElement("dd");
        description.textContent = value;
        metric.append(term, description);
        metrics.append(metric);
      }

      const details = document.createElement("details");
      details.className = "validation-result-coordinates";
      const summary = document.createElement("summary");
      summary.textContent = "座標詳細";
      const target = document.createElement("p");
      target.textContent = `正解: ${formatPoint(trial.target_xyz_mm)} mm`;
      const answer = document.createElement("p");
      answer.textContent = `回答: ${formatPoint(trial.answer_xyz_mm)} mm`;
      details.append(summary, target, answer);

      card.append(header, error, metrics, details);
      fragment.append(card);
    }
    this.els.resultList.replaceChildren(fragment);
    this.els.resultViewerCondition.textContent = "MPR・提案ビューアの2条件を同時表示";
    this.getScene().showValidationResults(trials);
  }

  showOnly(name) {
    this.els.setup.hidden = name !== "setup";
    this.els.task.hidden = name !== "task";
    this.els.targetEditor.hidden = name !== "targetEditor";
    this.els.complete.hidden = name !== "complete";
  }

  async loadConfig() {
    if (this.config) return this.config;
    const response = await fetch("./data/validation/study-config.phase1.json");
    if (!response.ok) throw new Error("研究設定を読み込めませんでした。");
    this.config = await response.json();
    return this.config;
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
    if (this.isPractice || !this.session) return;
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

export function buildTargets(metadata, definitions, overrides = {}) {
  const [width, height, depth] = metadata.dimensions;
  return definitions.map((definition) => {
    const override = overrides?.[definition.target_id];
    return {
      target_id: definition.target_id,
      xyz: isPoint3(override)
        ? override.map(Number)
        : voxelToDicom(metadata, [
          (width - 1) * definition.voxel_fraction[0],
          (height - 1) * definition.voxel_fraction[1],
          (depth - 1) * definition.voxel_fraction[2],
        ]),
    };
  });
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

function drawLinkedMprCrosshair(
  canvas,
  result,
  request,
  intersectionPoint,
  targetPoint,
  offsetMm,
  targetRadiusMm,
  currentDefinition,
  definitions,
) {
  canvas.width = result.width;
  canvas.height = result.height;
  const context = canvas.getContext("2d");
  context.putImageData(new ImageData(result.rgba, result.width, result.height), 0, 0);
  const projected = projectPointToMprPixel(result, request, intersectionPoint);
  for (const definition of definitions) {
    if (definition === currentDefinition) continue;
    const controlsHorizontalCoordinate =
      Math.abs(dot3(definition.normal, result.plane.u))
      >= Math.abs(dot3(definition.normal, result.plane.v));
    const coordinate = controlsHorizontalCoordinate ? projected.x : projected.y;
    const limit = controlsHorizontalCoordinate ? result.width : result.height;
    if (coordinate < 0 || coordinate > limit - 1) continue;
    drawCrosshairLine(context, {
      vertical: controlsHorizontalCoordinate,
      coordinate,
      width: result.width,
      height: result.height,
      color: definition.color,
    });
  }
  if (
    projected.x >= 0 && projected.x <= result.width - 1
    && projected.y >= 0 && projected.y <= result.height - 1
  ) {
    drawIntersectionPoint(context, projected);
  }
  drawMprTargetMarker(
    context,
    projectPointToMprPixel(result, request, targetPoint),
    offsetMm,
    targetRadiusMm,
    request,
    result.width,
    result.height,
  );
}

function drawCrosshairLine(context, { vertical, coordinate, width, height, color }) {
  const draw = () => {
    context.beginPath();
    if (vertical) {
      context.moveTo(coordinate, 0);
      context.lineTo(coordinate, height);
    } else {
      context.moveTo(0, coordinate);
      context.lineTo(width, coordinate);
    }
    context.stroke();
  };
  context.strokeStyle = "rgb(0 0 0 / 0.65)";
  context.lineWidth = 2.25;
  draw();
  context.strokeStyle = color;
  context.lineWidth = 0.85;
  draw();
}

function drawIntersectionPoint(context, point) {
  context.fillStyle = "rgb(255 255 255 / 0.9)";
  context.beginPath();
  context.arc(point.x, point.y, 2, 0, Math.PI * 2);
  context.fill();
}

function drawMprTargetMarker(context, point, offsetMm, sphereRadiusMm, request, width, height) {
  if (point.x < 0 || point.x > width - 1 || point.y < 0 || point.y > height - 1) return;
  const sectionRadiusMm = calculateSphereSectionRadius(sphereRadiusMm, offsetMm);
  if (sectionRadiusMm <= 0) return;
  const radiusX = Math.max(1, sectionRadiusMm / request.spacingU);
  const radiusY = Math.max(1, sectionRadiusMm / request.spacingV);
  const draw = () => {
    context.beginPath();
    context.ellipse(point.x, point.y, radiusX, radiusY, 0, 0, Math.PI * 2);
  };
  context.fillStyle = "rgb(98 240 167 / 0.62)";
  draw();
  context.fill();
  context.strokeStyle = "rgb(0 0 0 / 0.82)";
  context.lineWidth = 3.5;
  draw();
  context.stroke();
  context.strokeStyle = "rgb(170 255 207 / 1)";
  context.lineWidth = 2;
  draw();
  context.stroke();
  context.fillStyle = "rgb(220 255 236 / 0.95)";
  context.beginPath();
  context.arc(point.x - radiusX * 0.28, point.y - radiusY * 0.28, Math.max(0.8, Math.min(radiusX, radiusY) * 0.13), 0, Math.PI * 2);
  context.fill();
}

export function calculateSphereSectionRadius(sphereRadius, planeDistance) {
  const radius = Math.max(0, Number(sphereRadius) || 0);
  const distance = Math.abs(Number(planeDistance) || 0);
  if (radius === 0 || distance >= radius) return 0;
  return Math.sqrt(radius * radius - distance * distance);
}

export function projectPointToMprPixel(result, request, point) {
  const relative = point.map((value, index) => value - result.plane.center[index]);
  return {
    x: (result.width - 1) / 2 + dot3(relative, result.plane.u) / request.spacingU,
    y: (result.height - 1) / 2 + dot3(relative, result.plane.v) / request.spacingV,
  };
}

export function calculateSliceOffsetRange(metadata, target, normal) {
  if (!metadata?.dimensions || !metadata?.spacing || !metadata?.origin || !metadata?.direction) {
    return { min: -100, max: 100 };
  }
  const relative = target.map((value, index) => value - metadata.origin[index]);
  const voxel = [0, 1, 2].map((axis) => {
    const direction = [
      metadata.direction[axis],
      metadata.direction[3 + axis],
      metadata.direction[6 + axis],
    ];
    return dot3(relative, direction) / metadata.spacing[axis];
  });
  const rate = [0, 1, 2].map((axis) => {
    const direction = [
      metadata.direction[axis],
      metadata.direction[3 + axis],
      metadata.direction[6 + axis],
    ];
    return dot3(normal, direction) / metadata.spacing[axis];
  });
  let min = -Infinity;
  let max = Infinity;
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(rate[axis]) < 1e-9) continue;
    const first = (0 - voxel[axis]) / rate[axis];
    const last = (metadata.dimensions[axis] - 1 - voxel[axis]) / rate[axis];
    min = Math.max(min, Math.min(first, last));
    max = Math.min(max, Math.max(first, last));
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return { min: -100, max: 100 };
  return { min, max };
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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
    observation_time_ms: getObservationTimeMs(trial),
    response_time_ms: trial.response_time_ms,
    reposition_count: trial.reposition_count,
    status: trial.status,
  };
}

function getObservationTimeMs(trial) {
  const stored = Number(trial?.observation_time_ms);
  if (Number.isFinite(stored) && stored >= 0) return Math.round(stored);
  return calculateElapsedTimeMs(trial?.observation_started_at, trial?.observation_ended_at);
}

function calculateElapsedTimeMs(startedAt, endedAt) {
  const start = Date.parse(startedAt);
  const end = Date.parse(endedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return 0;
  return Math.max(0, Math.round(end - start));
}

function readSessions() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function readTargetOverrides() {
  try {
    const value = JSON.parse(localStorage.getItem(TARGETS_KEY) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

function isPoint3(value) {
  return Array.isArray(value) && value.length === 3 && value.every((item) => Number.isFinite(Number(item)));
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

function signedFixed(value) {
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(1)}`;
}

function formatPoint(point) {
  return point.map((value) => Number(value).toFixed(1)).join(", ");
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
